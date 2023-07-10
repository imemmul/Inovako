import pypylon.pylon as py
import time
import sys
import cv2
import numpy as np
import torch
import argparse
from .models import TRTModule
from .config import CLASSES, COLORS, MASK_COLORS, ALPHA
from .models.torch_utils import seg_postprocess
from .utils import MockInstantCamera, MockCameraArray
from .models.utils import blob, letterbox
import multiprocessing
from multiprocessing import Queue, Process, set_start_method
import os
from itertools import zip_longest
from PyQt6.QtWidgets import QMessageBox
from .utils import count_images
#from ..app import pop_up_call
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import traceback

DEFAULT_EXPOSURE = 10000
# TODO ProcessPoolExecutor
# TODO some optimizations needed, cam queue cam might wait other cam's to complete
# TODO parallel inference (max delay 0.01 = 10ms) acceptable
# TODO design of seperating the functions for master and the others is faulty, no need to do that, a simple Cam class might be better

class BaslerCamera():
    def __init__(self, cam:py.InstantCamera, is_master, cam_id) -> None:
        self.cam = cam
        self.cam_id = cam_id
        self.is_master = is_master

class BaslerCameraArray():
    def __init__(self, num_cams, master_id) -> None:
        self.tlf_object = py.TlFactory.GetInstance()
        self.cam_array = py.InstantCameraArray(num_cams)
        self.baslercam_array = []
        self.master_id = master_id
        self.devs = self.tlf_object.EnumerateDevices()
    def init_array(self, h, w, fps):
        for idx, cam in enumerate(self.cam_array):
            cam.Attach(self.tlf_object.CreateDevice(self.devs[idx]))
        self.cam_array.Open()
        self.configure_cams(h=h, w=w, fps=fps)
        return self.cam_array
    def configure_cams(self, h, w, fps):
        for idx, cam in enumerate(self.cam_array):
            camera_serial = cam.DeviceInfo.GetSerialNumber()
            print(f"set context {idx} for camera {camera_serial}")
            cam.SetCameraContext(idx)
            cam.ExposureTime.SetValue(DEFAULT_EXPOSURE)
            cam.PixelFormat.SetValue('Mono8')
            #cam.Width.SetValue(2600)
            #cam.Height.SetValue(2128)
            cam.Width.SetValue(w)
            cam.Height.SetValue(h)
            cam.TriggerSelector = "FrameStart"
            cam.TriggerMode.SetValue('On')
            cam.TriggerSource.SetValue('Software')
            # below 3 lines run the flashes on cameras TODO what to do here ?
            cam.Gamma.SetValue(0.7)
            cam.LineSelector.SetValue("Line2")
            cam.LineMode.SetValue("Output")
            cam.LineSource.SetValue("ExposureActive")
            cam.AcquisitionFrameRateEnable.SetValue(True)
            cam.AcquisitionFrameRate.SetValue(fps)
            cam.LineInverter.SetValue(True)
            self.baslercam_array.append(BaslerCamera(cam=cam, is_master=(idx==self.master_id), cam_id=idx))
    def get_cam(self, index):
        return self.baslercam_array[index], self.cam_array[index] # returns tuple of BaslerCam obj and original InstantCameraAray
        

def delete_files(args, limit):
    if len(os.listdir(args.out_dir)) > limit:
        for out in os.listdir(args.out_dir):
            os.remove(f"{args.out_dir}{out}")
        print(f"Deleted all files in out-dir")

def run_inference(q:Queue, id, args, running, devices):
    try:
        engine, device, H, W  = load_engine(args)
        run_id = len(os.listdir(args.out_dir))
        print(f"running inference group: {id}")
        while running.is_set() or q.qsize() > 0:
            try:
                start_time = time.time()
                # count = len(os.listdir(args.out_dir))
                # print(f"QUEUE SIZE OF cam:{cam_id}: {q.qsize()}")
                # print(f"what is q_length = {q.qsize()}")
                image, cam_id, exp_time, capture_id, capture_time = q.get()
                print(f"image taken from cam: {cam_id}, processing")
                # print(f"cam id {cam_id} captured in {capture_time}")
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                draw = bgr.copy()
                bgr, ratio, dwdh = letterbox(bgr, (W, H))
                dw, dh = int(dwdh[0]), int(dwdh[1])
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                tensor, seg_img = blob(rgb, return_seg=True)
                dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
                tensor = torch.asarray(tensor, device=device)
                data = engine(tensor)
                seg_img = torch.asarray(seg_img[dh:H - dh, dw:W - dw, [2, 1, 0]],
                                        device=device)
                bboxes, scores, labels, masks = seg_postprocess(
                    data, bgr.shape[:2], args.conf_thres, args.iou_thres)
                
                if len(bboxes) == 0:
                    # print(f"trying to save no_det here: {args.out_dir}run_{run_id}/{devices[cam_id]}/{exp_time}/NO_DET/output_{capture_id}.jpg")
                    cv2.imwrite(filename=f"{args.out_dir}run_{run_id}/{devices[cam_id]}/{exp_time}/NO_DET/output_{capture_id}.jpg", img=image)
                    #print(f"nothing detected")
                else:
                    masks = masks[:, dh:H - dh, dw:W - dw, :]
                    indices = (labels % len(MASK_COLORS)).long()
                    mask_colors = torch.asarray(MASK_COLORS, device=device)[indices]
                    mask_colors = mask_colors.view(-1, 1, 1, 3) * ALPHA
                    mask_colors = masks @ mask_colors
                    inv_alph_masks = (1 - masks * 0.5).cumprod(0)
                    mcs = (mask_colors * inv_alph_masks).sum(0) * 2
                    seg_img = (seg_img * inv_alph_masks[-1] + mcs) * 255
                    draw = cv2.resize(seg_img.cpu().numpy().astype(np.uint8),
                                    draw.shape[:2][::-1])

                    bboxes -= dwdh
                    bboxes /= ratio

                    for (bbox, score, label) in zip(bboxes, scores, labels):
                        bbox = bbox.round().int().tolist()
                        cls_id = int(label)
                        cls = CLASSES[cls_id]
                        color = COLORS[cls]
                        cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
                        cv2.putText(draw,
                                    f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.75, [225, 255, 255],
                                    thickness=2)
                    # print(f"image saved")
                    #print(f"trying to save det here: {args.out_dir}run_{run_id}/{devices[cam_id]}/{exp_time}/DET/output_{capture_id}.jpg")
                    cv2.imwrite(filename=f"{args.out_dir}run_{run_id}/{devices[cam_id]}/{exp_time}/DET/output_{capture_id}.jpg", img=draw)
                end_time = time.time()
                print(f"Its been {end_time-start_time} seconds to process cam: {cam_id}")
            except Exception as e:
                print(f"Some error occured in run_inference with cam_id:{cam_id}: {e}")
    except Exception as e:
        print(f"Error in inference: {e}")
    print("stopping inference")

def part_detection(img, threshold):
    gray_value = np.mean(img)
    # below is selecting a 400x400
    # height, width = img.shape[:2]
    # start_x = width // 2 - 200
    # end_x = width // 2 + 200
    # start_y = 0
    # end_y = 400
    # selected_area = cv2.cvtColor(img[start_y:end_y, start_x:end_x], cv2.COLOR_RGB2GRAY)
    # gray_value = np.mean(selected_area)
    print(f"gray_value: {gray_value}, and {gray_value > threshold}")
    return gray_value > threshold

def run_devices(bca, cam_group, q, nums_cams, args):
    # Define a function to be run in each thread
    delay_dict = {}
    for group in cam_group:
        with ThreadPoolExecutor(max_workers=nums_cams) as executor:
            try:
                for cam_id, cam in enumerate(cam_group):
                    # print(f"capture amount: {capture_amount}")
                    if cam_id != args.master:
                        executor.submit(trigger_and_capture, args, group, cam_id, running, q, delay_dict, capture_test) 
                    else:
                        print(f"master cam thread loaded")
                        executor.submit(trigger_master, group, cam, cam_id, running, q, delay_dict, capture_test)
            except Exception as e:
                print(f"some error occured in thread pool: {e}")
    executor.shutdown(wait=False)  # Stop the executor
    print(f"executor shutdowned")
    # Get the start times in a list
    start_times = list(delay_dict.values())

    # Sort the start times
    start_times.sort()

    # Calculate the delays
    delays = [start_times[i] - start_times[i - 1] for i in range(1, len(start_times))]
    count_images(args)
    # Print the delays
    for i in range(len(delays)):
        print(f"Delay between camera {i} and camera {i + 1}: {delays[i]} seconds")

def trigger_master(args, group, cam_id, running, q, delay_dict, capture_test):
    # TODO convert this into for cam in group run cam 
    try:
        capture_amount = 0
        while running.is_set():
            # print(f"master is running")
            for cam in group:
                cam.ExposureTime.SetValue(int(args.exposure_time))
                cam.ExecuteSoftwareTrigger()
                grabResult = cam.RetrieveResult(1000, py.TimeoutHandling_ThrowException)
                # print(f"grabbed something")
                if grabResult.GrabSucceeded():
                    print(f"image captured from cam:{cam_id} with exp_time: {args.exposure_time}")
                    img = grabResult.GetArray()
                    if part_detection(img, args.gray_thres):
                        #print(f"part detected running all other camereas")
                        capture_test.set() # run other cameras
                        capture_amount += 1
                        capture_time = time.time()
                        q.put((img, cam_id, args.exposure_time, capture_amount, capture_time))
                    else:
                        print(f"No Part detected, checking in every {args.check_interval} seconds, master cam: {cam_id}")
                        capture_test.clear()
                        time.sleep(args.check_interval)
                time.sleep(args.interval)
            delay_dict[cam_id] = time.time()
            cam.Close()
        print(f"Thread with cam_id {cam_id} stopped")
    except Exception as e:
        print(f"some error occured in trigger_master: {e}")
        traceback.print_exc()
        # print(f"camera {cam_id} captured image in {time.time()}")

def trigger_and_capture(args, cam, cam_id, running, q, delay_dict, capture_test):
        capture_amount = 0
        try:
            while running.is_set():
                # print(f"running ?: {running.is_set()}")
                # print(f"i am trying cam: {cam_id}")
                if capture_test.is_set():
                    # print(f"i am running cam: {cam_id}")
                    # for _ in range(1):
                    cam.ExposureTime.SetValue(int(args.exposure_time))
                    cam.ExecuteSoftwareTrigger()
                    grabResult = cam.RetrieveResult(1000, py.TimeoutHandling_ThrowException)
                    # print(f"grabbed something")
                    if grabResult.GrabSucceeded():
                        print(f"image captured from cam:{cam_id} with exp_time: {args.exposure_time}")
                        img = grabResult.GetArray()
                        # print(f"image put in queue")
                        capture_amount += 1
                        capture_time = time.time()
                        #print(f"image got with shape: {img.shape} from cam: {cam_id}")
                        q.put((img, cam_id, args.exposure_time, capture_amount, capture_time))
                        time.sleep(args.interval)
                    delay_dict[cam_id] = time.time()
                    # print(f"camera {cam_id} captured image in {time.time()}")
            cam.Close()
        except Exception as e:
            print(f"some bugs in trigger_and_capture: {e}")
        print(f"Thread with cam_id {cam_id} stopped")

def load_engine(args):
    try:
        device = torch.device(args.device)
        # print(f"what is device {args.engine}")
        Engine = TRTModule(args.engine, device)
        H, W = Engine.inp_info[0].shape[-2:]
        Engine.set_desired(['outputs', 'proto'])
        return Engine, device, H, W
    except Exception as e:
        print(f"An error occured in load_engine: {e}")

def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def load_devices(args):
    tlf = py.TlFactory.GetInstance()
    devs = tlf.EnumerateDevices()
    print(f"args master: {args.master}")
    #try:
    if len(devs) > 0:
        num_cams = len(devs)
        print(f"num cams: {num_cams}")
        bca = BaslerCameraArray(num_cams=num_cams, master_id=args.master)
        cam_array = bca.init_array(h=1080, w=1920, fps=60)
        print(type(cam_array))
        return bca, cam_array, num_cams
    else:
        print(f"No devices found")
    # except Exception as e:
    #     print(f"An error in load_devices: {e}")
        #pop_up_call(error_name="Another Client is Running", error_text="Please check other clients")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default="/home/emir/Desktop/dev/Inovako/tensorrt_engines/tofas_model.engine")
    parser.add_argument('--out-dir', type=str, default='../output/')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gray-thres', type=int, default=30)
    parser.add_argument('--exposure-time', type=int, default=850)
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.65)
    parser.add_argument('--interval', type=float, default=0.1)
    parser.add_argument('--check-interval', type=int, default=5)
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--test-engine', action="store_true")
    parser.add_argument('--filter-cam', type=str)
    parser.add_argument('--filter-expo', type=str)
    parser.add_argument('--master', type=int, default=1)
    parser.add_argument('--group-size', type=int, default=2)
    args = parser.parse_args()
    return args

def stop_engine():
    global running
    running.clear()
    print(f"stopped the engine")

def run_engine(args):
    # if len(os.listdir(args.out_dir)) > 0:
    #     for out in os.listdir(args.out_dir):
    #         os.remove(f"{args.out_dir}{out}")
    #     print(f"Deleted all files in out-dir")
    global running
    running = multiprocessing.Event()
    running.set()
    global capture_test
    capture_test = multiprocessing.Event()
    bca, cam_array, num_cams = load_devices(args)
    cam_groups = grouper(args.group_size, cam_array)
    print(f"Devices are loaded, running")
    cam_array.StartGrabbing(py.GrabStrategy_LatestImageOnly)
    # Creating thread for each group
    # queues = []
    # processes = []
    with ThreadPoolExecutor(max_workers=args.group_size) as executor:
        for id, group in enumerate(cam_groups):
            q = Queue()
            p = Process(target=run_inference, args=(q, id, args, running, list_devices(args))) # background listening
            p.start()
            print(f"running group: {id}")
            executor.submit(run_devices(bca=bca, cam_group=group, q=q, nums_cams=len(group), args=args))
    executor.shutdown(wait=False)  # Stop the executor
    print(f"executor shutdowned group")
    if p.is_alive(): # If the process is still running, terminate it
        p.terminate()
    p.join()
    cam_array.StopGrabbing()

def list_devices(args):
    try:
        if args.test:
            return [str(i) for i in range(8)]
        else:
            devs = py.TlFactory.GetInstance().EnumerateDevices()
            return [device.GetFriendlyName() for device in devs]
    except Exception as e:
        print(f"Error in list_devices")



if __name__ == "__main__":
    set_start_method('spawn') # this is temp
    args = parse_args()
    run_engine(args)