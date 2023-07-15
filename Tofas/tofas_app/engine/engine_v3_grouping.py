# The following import statements include necessary libraries for the operation of the code. 
# The libraries are used for image processing, parallel processing, time manipulation, 
# error handling, argument parsing, etc.

# TOFAS MAIN ENGINE // last updated in 07.15.2023 E.Ulurak emirulurak@mgmail.com

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
# from ..app import pop_up_call
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import traceback
from threading import Barrier


# The following class represents an array of Basler cameras and is used to initialize, configure and manage the cameras.
class BaslerCameraArray():
    # The constructor method initializes the BaslerCameraArray object with a given number of cameras and a master camera id.
    def __init__(self, num_cams, master_id) -> None:
        self.tlf_object = py.TlFactory.GetInstance()  # This line gets an instance of the Pylon Transport Layer Factory.
        self.cam_array = py.InstantCameraArray(num_cams)  # Creates an array of InstantCamera objects.
        self.master_id = master_id  # The ID of the master camera.
        self.devs = self.tlf_object.EnumerateDevices()  # List of all available pylon devices.
    
    # This method initializes the camera array by attaching each camera in the array to a device and opening it. Then it configures the cameras.
    def init_array(self, args,  h, w, fps):
        for idx, cam in enumerate(self.cam_array):
            cam.Attach(self.tlf_object.CreateDevice(self.devs[idx]))
        self.cam_array.Open()
        self.configure_cams(args=args, h=h, w=w, fps=fps)
        return self.cam_array

    # This method configures each camera in the array based on the arguments provided.  
    def configure_cams(self, args, h, w, fps):
        for idx, cam in enumerate(self.cam_array):
            camera_serial = cam.DeviceInfo.GetSerialNumber()
            print(f"set context {idx} for camera {camera_serial}")
            # Various camera settings are adjusted here.
            # Note: This is an example configuration, the exact settings will depend on your use case.
            cam.SetCameraContext(idx)
            cam.ExposureTime.SetValue(int(args.exposure_time))  # Set exposure time for the camera.
            cam.PixelFormat.SetValue('Mono8')  # Set pixel format to Mono8.
            cam.TriggerSelector = "FrameStart"  # Selects the kind of trigger to configure.
            cam.TriggerMode.SetValue('On')  # Turns triggering mode on.
            cam.TriggerSource.SetValue('Software')  # The source that triggers the acquisition.
            cam.Gamma.SetValue(0.7)  # The gamma correction value.
            cam.LineSelector.SetValue("Line2")  # Selects the input or output line.
            cam.LineMode.SetValue("Output")  # Sets the mode of the selected line.
            cam.LineSource.SetValue("ExposureActive")  # The signal source for the selected line.
            cam.AcquisitionFrameRateEnable.SetValue(True)  # Enables setting the acquisition frame rate.
            cam.AcquisitionFrameRate.SetValue(fps)  # The acquisition frame rate to set.
            cam.LineInverter.SetValue(True)  # Inverts the logic level of the selected line.

    # This method returns the BaslerCam object and original InstantCameraArray for the provided index.
    def get_cam(self, index):
        return self.baslercam_array[index], self.cam_array[index]

    # This method creates groups of cameras based on the provided group size.
    def create_groups(self, group_size):
        return self.grouper(group_size, self.cam_array)

    # This method groups a list of cameras into smaller groups of a specified size.
    def grouper(self, group_size, cam_array):
        """
        returns group of given list with group_size
        """
        cam_list = list(cam_array)
        return [cam_list[i:i+group_size] for i in range(0, len(cam_list), group_size)]

# This function runs inference on images fetched from the queue.
# It runs the inference in a loop, as long as there are images in the queue and the running flag is set.
# If no object is detected in the image, it is saved to a "NO_DET" directory.
# If an object is detected, it is marked in the image and the image is saved to a "DET" directory.
def run_inference(q:Queue, group_id, args, running, devices):
    """
    Creates TensorRT engines with assigned groups and inference each image captured from that group cameras.
    """
    try:
        engine, device, H, W  = load_engine(args)
        run_id = len(os.listdir(args.out_dir))
        print(f"running inference group: {id}")
        while True:
            try:
                # print(f"QUEUE SIZE OF cam:{cam_id}: {q.qsize()}")
                image, cam_id, exp_time, capture_id, capture_time = q.get()
                print(f"image taken from cam: {cam_id}, processing in group {group_id}")
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
                # print(f"Its been {end_time-start_time} seconds to process cam: {cam_id}")
                if not running.is_set() and q.qsize() == 0:
                    break
            except Exception as e:
                print(f"Some error occured in run_inference with cam_id:{cam_id}: {e}")
    except Exception as e:
        print(f"Error in inference: {e}")
    print(f"stopping inference for group: {id}")

def part_detection(img, threshold):
    gray_value = np.mean(img)
    print(f"gray_value: {gray_value}, and {gray_value > threshold}")
    return gray_value > threshold

def run_devices(cam_groups, nums_cams, args):
    """
    Starts running the connected devices and assigns threads to each device.
    Uses multiprocessing to create processes and assigns a separate thread to each camera device within that process.
    """
    cam_id = 0
    barrier = Barrier(nums_cams)
    inference_processes = []
    delay_dict = {}
    with ThreadPoolExecutor(max_workers=nums_cams) as executor:
        for group_id, group in enumerate(cam_groups):
            print(f"group_id: {group_id}")
            q = Queue()
            p = Process(target=run_inference, args=(q, group_id, args, running, list_devices(args))) # background listening (Queue)
            p.start()
            inference_processes.append(p)
            try:
                for cam in group:
                    # print(f"cam: {cam}, cam_id {cam_id}")
                    # print(f"capture amount: {capture_amount}")
                    if cam_id != args.master:
                        executor.submit(trigger_and_capture, args, cam, cam_id, running, q, delay_dict, capture_all, barrier) 
                    else:
                        print(f"master cam thread loaded")
                        executor.submit(trigger_master, args, cam, cam_id, running, q, delay_dict, capture_all, barrier)
                    cam_id += 1
                for i, p in enumerate(inference_processes):
                    print(f"Process {i} is {'alive' if p.is_alive() else 'not alive'}")
            except Exception as e:
                print(f"some error occured in thread pool: {e}")
                traceback.print_exc()
    executor.shutdown(wait=True)  # Stop the executor
    for p in inference_processes:
        if p.is_alive(): # If the process is still running, terminate it
            p.terminate()
        p.join()
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

def trigger_master(args, cam, cam_id, running, q, delay_dict, capture_all, barrier:Barrier):
    
    """
    Master thread that controls the master camera. It checks if the master camera is able to capture images,
    if it does it will set a flag to trigger other camera to start capturing as well.
    """
    capture_amount = 0
    try:
        current_wait_time = 0
        while True:
            # print(f"master is running")
            # cam.ExposureTime.SetValue(int(args.exposure_time))
            try:
                cam.ExecuteSoftwareTrigger()
                grabResult = cam.RetrieveResult(1000, py.TimeoutHandling_ThrowException)
                # print(f"grabbed something")
                if grabResult.GrabSucceeded():
                    print(f"image captured from cam:{cam_id} with exp_time: {args.exposure_time}")
                    img = grabResult.GetArray()
                    if part_detection(img, args.gray_thres):
                        #print(f"part detected running all other camereas")
                        current_wait_time = 0
                        capture_all.set() # run other cameras
                        capture_amount += 1
                        capture_time = time.time()
                        q.put((img, cam_id, args.exposure_time, capture_amount, capture_time))
                        time.sleep(args.interval)
                    else:
                        current_wait_time += 1
                        print(f"No Part detected, checking in every {args.check_interval} seconds, master cam: {cam_id}")
                        capture_all.clear()
                        if current_wait_time == args.wait_time:
                            print(f"Waiting limit reached stopping.")
                            stop_engine()
                        time.sleep(args.check_interval)
                else:
                    print(f"couldn't capture master")
                    time.sleep(args.interval)
                if not running.is_set():
                    break
            except Exception as e:
                print(f"some error occured in trigger_master capturing: {e}")
                traceback.print_exc()
            finally:
                print("waitting")
                barrier.wait()
        delay_dict[cam_id] = time.time()
        cam.Close()
        print(f"Thread with cam_id {cam_id} stopped")
    except Exception as e:
        print(f"some error occured in trigger_master: {e}")
        traceback.print_exc()
        # print(f"camera {cam_id} captured image in {time.time()}")

def trigger_and_capture(args, cam, cam_id, running, q, delay_dict, capture_all, barrier:Barrier):
    """
    Function to capture images from the camera. It gets triggered by the master thread to start capturing.
    """
    capture_amount = 0
    try:
        while True:
            # print(f"running ?: {running.is_set()}")
            # print(f"i am trying cam: {cam_id}")
            try:
                if capture_all.is_set():
                    # print(f"i am running cam: {cam_id}")
                    # for _ in range(1):
                    # cam.ExposureTime.SetValue(int(args.exposure_time))
                    cam.ExecuteSoftwareTrigger()
                    grabResult = cam.RetrieveResult(1000, py.TimeoutHandling_ThrowException)
                    # print(f"grabbed something")
                    if grabResult.GrabSucceeded():
                        print(f"image captured from cam:{cam_id} with exp_time: {args.exposure_time}")
                        img = grabResult.GetArray()
                        # print(f"image put in queue")
                        capture_time = time.time()
                        #print(f"image got with shape: {img.shape} from cam: {cam_id}")
                        capture_amount += 1
                        q.put((img, cam_id, args.exposure_time, capture_amount, capture_time))
                        time.sleep(args.interval)
                    delay_dict[cam_id] = time.time()
                    # print(f"camera {cam_id} captured image in {time.time()}")
                if not running.is_set():
                    break
            except Exception as e:
                print(f"some bugs in trigger_and_capture capturing: {e}")
            finally:
                barrier.wait()
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

def load_devices(args):
    tlf = py.TlFactory.GetInstance()
    devs = tlf.EnumerateDevices()
    print(f"args master: {args.master}")
    #try:
    if len(devs) > 0:
        num_cams = len(devs)
        print(f"num cams: {num_cams}")
        bca = BaslerCameraArray(num_cams=num_cams, master_id=args.master)
        cam_array = bca.init_array(args=args, w=1920, h=1080, fps=60)
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

def update_status(engine_stopped):
    with open('./status.txt', 'w') as f:
        f.write(str(int(engine_stopped)))

def stop_engine():
    global running
    running.clear()
    update_status(True)
    print(f"stopped the engine")

def run_engine(args):
    global running
    running = multiprocessing.Event()
    running.set()
    global capture_all
    capture_all = multiprocessing.Event()
    bca, cam_array, num_cams = load_devices(args)
    cam_groups = bca.create_groups(args.group_size) # this is [(baslercam_array[0], py.InstantCameraArray[0])]'
    print(f"Devices are loaded, running")
    update_status(False) # engine started
    cam_array.StartGrabbing(py.GrabStrategy_LatestImageOnly)
    run_devices(cam_groups=cam_groups, nums_cams=num_cams, args=args)
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