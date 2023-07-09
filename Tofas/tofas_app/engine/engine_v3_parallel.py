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
from PyQt6.QtWidgets import QMessageBox
from .utils import count_images
#from ..app import pop_up_call
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

DEFAULT_EXPOSURE = 10000
# TODO ProcessPoolExecutor
# TODO some optimizations needed, cam queue cam might wait other cam's to complete
# TODO parallel inference (max delay 0.01 = 10ms) acceptable


def delete_files(args, limit):
    if len(os.listdir(args.out_dir)) > limit:
        for out in os.listdir(args.out_dir):
            os.remove(f"{args.out_dir}{out}")
        print(f"Deleted all files in out-dir")

def run_inference(q:Queue, args, running, devices):
    try:
        engine, device, H, W  = load_engine(args)
        run_id = len(os.listdir(args.out_dir))
        while running.is_set() or q.qsize() > 0:
            try:
                start_time = time.time()
                # count = len(os.listdir(args.out_dir))
                print(f"QUEUE SIZE OF cam:{cam_id}: {q.qsize()}")
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

def run_devices(cam_array, nums_cams, args):
    """
    this function stands for the testing 8 multiple cameras without 8 devices. ONLY FOR TEST PURPOSES
    """
    cam_array.StartGrabbing(py.GrabStrategy_LatestImageOnly) # ??
    # Define a function to be run in each thread
    def trigger_and_capture(cam, cam_id, running, delay_dict, capture_test):
        q = Queue()
        p = Process(target=run_inference, args=(q, args, running, list_devices(args))) # background listening
        p.start()
        capture_amount = 0
        try:
            while running.is_set():
                # print(f"running ?: {running.is_set()}")
                # print(f"i am trying cam: {cam_id}")
                if capture_test.is_set():
                    # print(f"i am running cam: {cam_id}")
                    # for _ in range(1):
                    exp_time = args.exposure_time[int(capture_amount % 2)]
                    cam.ExposureTime.SetValue(int(exp_time))
                    cam.ExecuteSoftwareTrigger()
                    grabResult = cam.RetrieveResult(1000, py.TimeoutHandling_ThrowException)
                    # print(f"grabbed something")
                    if grabResult.GrabSucceeded():
                        print(f"image captured from cam:{cam_id} with exp_time: {exp_time}")
                        img = grabResult.GetArray()
                        # print(f"image put in queue")
                        capture_amount += 1
                        capture_time = time.time()
                        #print(f"image got with shape: {img.shape} from cam: {cam_id}")
                        q.put((img, cam_id, exp_time, capture_amount, capture_time))
                        time.sleep(args.interval)
                    delay_dict[cam_id] = time.time()
                    # print(f"camera {cam_id} captured image in {time.time()}")
            cam.Close()
            if p.is_alive(): # If the process is still running, terminate it
                p.terminate()
            p.join()
        except Exception as e:
            print(f"some bugs in trigger_and_capture: {e}")
        print(f"Thread with cam_id {cam_id} stopped")

    delay_dict = {}
    with ThreadPoolExecutor(max_workers=nums_cams) as executor:
        try:
            for cam_id, cam in enumerate(cam_array):
                # print(f"capture amount: {capture_amount}")
                if cam_id != args.master:
                    executor.submit(trigger_and_capture, cam, cam_id, running, delay_dict, capture_test)
                else:
                    print(f"master cam thread loaded")
                    executor.submit(trigger_master, args, cam, cam_id, running, delay_dict, capture_test)
        except Exception as e:
            print(f"some error occured in thread pool: {e}")
    cam_array.StopGrabbing()
    print(f"executor shutdown")
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

def trigger_master(args, cam, cam_id, running, delay_dict, capture_test):
    try:
        q = Queue()
        p = Process(target=run_inference, args=(q, args, running, list_devices(args))) # background listening
        p.start()
        capture_amount = 0
        while running.is_set():
            # print(f"master is running")
            exp_time = args.exposure_time[int(capture_amount % 2)]
            cam.ExposureTime.SetValue(int(exp_time))
            cam.ExecuteSoftwareTrigger()
            grabResult = cam.RetrieveResult(1000, py.TimeoutHandling_ThrowException)
            # print(f"grabbed something")
            if grabResult.GrabSucceeded():
                print(f"image captured from cam:{cam_id} with exp_time: {exp_time}")
                img = grabResult.GetArray()
                if part_detection(img, args.gray_thres):
                    #print(f"part detected running all other camereas")
                    capture_test.set() # run other cameras
                    capture_amount += 1
                    capture_time = time.time()
                    q.put((img, cam_id, exp_time, capture_amount, capture_time))
                    time.sleep(args.interval)
                else:
                    print(f"No Part detected, checking in every {args.check_interval} seconds, master cam: {cam_id}")
                    capture_test.clear()
                    time.sleep(args.check_interval)
            else:
                print(f"couldn't capture master")
        if p.is_alive(): # If the process is still running, terminate it
            p.terminate()
        p.join()
        delay_dict[cam_id] = time.time()
        cam.Close()
        print(f"Thread with cam_id {cam_id} stopped")
    except Exception as e:
        print(f"some error occured in trigger_master: {e}")
        # print(f"camera {cam_id} captured image in {time.time()}")

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

def load_devices(args):
    tlf = py.TlFactory.GetInstance()
    devs = tlf.EnumerateDevices()
    try:
        if len(devs) > 0:
            num_cams = len(devs)
            print(f"num cams: {num_cams}")
            cam_array = py.InstantCameraArray(num_cams)
            for idx, cam in enumerate(cam_array):
                cam.Attach(tlf.CreateDevice(devs[idx]))
            cam_array.Open()
            for idx, cam in enumerate(cam_array):
                camera_serial = cam.DeviceInfo.GetSerialNumber()
                print(f"set context {idx} for camera {camera_serial}")
                cam.SetCameraContext(idx)
                cam.ExposureTime.SetValue(DEFAULT_EXPOSURE)
                cam.PixelFormat.SetValue('Mono8')
                #cam.Width.SetValue(2600)
                #cam.Height.SetValue(2128)
                cam.Width.SetValue(1920)
                cam.Height.SetValue(1080)
                cam.TriggerSelector = "FrameStart"
                cam.TriggerMode.SetValue('On')
                cam.TriggerSource.SetValue('Software')
                # below 3 lines run the flashes on cameras TODO what to do here ?
                cam.Gamma.SetValue(0.7)
                cam.LineSelector.SetValue("Line2")
                cam.LineMode.SetValue("Output")
                cam.LineSource.SetValue("ExposureActive")
                cam.AcquisitionFrameRateEnable.SetValue(True)
                cam.AcquisitionFrameRate.SetValue(60.0)
                cam.LineInverter.SetValue(True)
            return cam_array, num_cams
        else:
            print(f"No devices found")
    except Exception as e:
        print(f"another client is open")
        #pop_up_call(error_name="Another Client is Running", error_text="Please check other clients")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default="/home/emir/Desktop/dev/Inovako/tensorrt_engines/tofas_model.engine")
    parser.add_argument('--out-dir', type=str, default='../output/')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gray-thres', type=int, default=0)
    parser.add_argument('--exposure-time', type=list, default=[10000, 20000])
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.65)
    parser.add_argument('--interval', type=float, default=1.0)
    parser.add_argument('--check-interval', type=int, default=5)
    parser.add_argument('--test', action="store_true")
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
    cam_array, num_cams = load_devices(args)
    print(f"Devices are loaded, running")
    run_devices(cam_array=cam_array, nums_cams=num_cams, args=args)

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
    if args.test:
        print(f"Running test")
        run_engine(args)
    else:
        run_engine(args)