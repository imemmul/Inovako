import pypylon.pylon as py
import time
import sys
import cv2
import numpy as np
import torch
import argparse
from .models import TRTModule
from .config import CLASSES, COLORS, MASK_COLORS
from .models.torch_utils import seg_postprocess
from .models.utils import blob, letterbox
import multiprocessing
from multiprocessing import Queue, Process, set_start_method
import os
from concurrent.futures import ThreadPoolExecutor
from PyQt6.QtWidgets import QMessageBox

DEFAULT_EXPOSURE = 10000
TEST_DIR = "/home/emir/Desktop/dev/Inovako/Inovako/Tofas/tofas_app/engine/mock_images/"

# below classes is for mocking the pypylon InstantCamera object
class MockCameraArray:
    def __init__(self, num_cams):
        self.cameras = [MockInstantCamera(id=i) for i in range(num_cams)]
        self.is_grabbing = False

    def StartGrabbing(self, strategy=None):
        self.is_grabbing = True
        for camera in self.cameras:
            # Here you might want to initiate the grabbing process for each camera
            # but for this mock class, we'll just change a flag
            camera.is_grabbing = True

    def StopGrabbing(self):
        self.is_grabbing = False
        for camera in self.cameras:
            # Here you might want to stop the grabbing process for each camera
            # but for this mock class, we'll just change a flag
            camera.is_grabbing = False

    def __getitem__(self, index):
        return self.cameras[index]

    def __len__(self):
        return len(self.cameras)

    def Open(self):
        for camera in self.cameras:
            camera.Open()

    def Attach(self, device):
        for camera in self.cameras:
            camera.Attach(device)

class MockInstantCamera:
    def __init__(self, image_size=(720, 1280), pixel_range=(0, 255), id=0):
        self.image_size = image_size
        self.pixel_range = pixel_range
        self.PixelFormat = MockAttribute('Mono8')
        self.Width = MockAttribute(1280)
        self.Height = MockAttribute(720)
        self.TriggerSelector = "FrameStart"
        self.TriggerMode = MockAttribute('On')
        self.TriggerSource = MockAttribute('Software')
        self.ExposureTime = MockAttribute(10000)
        self.DeviceInfo = MockDeviceInfo(id)
        # print(f"grabbed image {os.path.join(TEST_DIR, sorted(os.listdir(TEST_DIR))[id])} for cam: {id}")
        self.grab_result = cv2.imread(os.path.join(TEST_DIR, sorted(os.listdir(TEST_DIR))[id]))

    def Attach(self, device):
        pass

    def ExecuteSoftwareTrigger(self):
        pass

    def RetrieveResult(self, timeout, timeout_handling):
        return self.grab_result 

    def Open(self):
        pass

    def SetCameraContext(self, idx):
        pass

class MockAttribute:
    def __init__(self, value):
        self.value = value

    def SetValue(self, value):
        self.value = value

    def GetValue(self):
        return self.value

class MockDeviceInfo:
    def __init__(self, id):
        self.serial_number = f'Mock{id}'

    def GetSerialNumber(self):
        return self.serial_number

def run_inference(q:Queue, args, running):
    try:
        engine, device, H, W  = load_engine(args)
    except Exception as e:
        print(f"Failed to load engine: {e}")
        return
    while running.is_set() or q.qsize() > 0:
        try:
            count = len(os.listdir(args.out_dir))
            # print(f"what is q_length = {q.qsize()}")
            image, cam_id, exp_time, capture_id = q.get()
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            draw = bgr.copy()
            # print(f"what is shape: {draw.shape}")
            bgr, ratio, dwdh = letterbox(bgr, (W, H))
            dw, dh = int(dwdh[0]), int(dwdh[1])
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor, seg_img = blob(rgb, return_seg=True)
            dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
            tensor = torch.asarray(tensor, device=device)

            data = engine(tensor)

            seg_img = torch.asarray(seg_img[dh:H - dh, dw:W - dw, [2, 1, 0]], device=device)
            bboxes, scores, labels, masks = seg_postprocess(data, bgr.shape[:2], args.conf_thres, args.iou_thres)
            print(f"running inference on captured images from cam: {cam_id}")
            if len(bboxes) == 0:
                print("Nothing detected.")
            else:
                # print(f"something detected")
                masks = masks[:, dh:H - dh, dw:W - dw, :]
                indices = (labels % len(MASK_COLORS)).long()
                mask_colors = torch.asarray(MASK_COLORS, device=device)[indices]
                mask_colors = mask_colors.view(-1, 1, 1, 3)
                mask_colors = masks @ mask_colors
                inv_alph_masks = (1 - masks * 0.5).cumprod(0)
                mcs = (mask_colors * inv_alph_masks).sum(0) * 2
                seg_img = (seg_img * inv_alph_masks[-1] + mcs) * 255
                draw = cv2.resize(seg_img.cpu().numpy().astype(np.uint8), draw.shape[:2][::-1])

                bboxes -= dwdh
                bboxes /= ratio

                for (bbox, score, label) in zip(bboxes, scores, labels):
                    bbox = bbox.round().int().tolist()
                    cls_id = int(label)
                    cls = CLASSES[cls_id]
                    color = COLORS[cls]
                    cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
                    cv2.putText(draw, f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)
                count += 1
                print(f"image saved")
                cv2.imwrite(filename=f"{args.out_dir}output_{count}_{cam_id}_{capture_id}_{exp_time}.jpg", img=draw)
        except Exception as e:
            print(f"Some error occured in run_inference: {e}")

def part_detection(img, threshold):
    gray_value = np.mean(img)
    print(f"gray_value: {gray_value}")
    return gray_value > threshold

def run_devices(cam_array, nums_cams, args):
    q = Queue()
    p = Process(target=run_inference, args=(q, args, running))
    p.start()
    # TODO differ the file nameing count and capture amount ? DONE
    # file_count = len(os.listdir(args.out_dir))
    exp_time = 0
    cam_array.StartGrabbing(py.GrabStrategy_LatestImageOnly) # ??

    # Define a function to be run in each thread
    def trigger_and_capture(cam, cam_id, exp_time, running, delay_dict):
        """
        each cam will run this function and capture the images
        """
        capture_amount = 0
        print(f"which cam is trying run {cam_id}")
        while running.is_set():
            try:
                exp_time = args.exposure_time[int(capture_amount % 2)]
                cam.ExposureTime.SetValue(int(exp_time))
                cam.ExecuteSoftwareTrigger()
                grabResult = cam.RetrieveResult(1000, py.TimeoutHandling_ThrowException)
                # print(f"grabbed something")
                if grabResult.GrabSucceeded():
                    print(f"image captured from cam:{cam_id} with exp_time: {exp_time}")
                    img = grabResult.GetArray()
                    grabResult.Release()
                    if part_detection(img, args.gray_thres):
                        print(f"image put in queue")
                        capture_amount += 1
                        q.put((img, cam_id, exp_time, capture_amount))
                        time.sleep(args.interval)
                    else:
                        print(f"No Part detected, checking in every {args.check_interval} seconds")
                        time.sleep(args.check_interval)
                delay_dict[cam_id] = time.time()
            except Exception as e:
                print(f"Some Error occured in trigger_and_capture, {e}")
        print(f"Thread with cam_id {cam_id} stopped")

    delay_dict = {}
    try:
        with ThreadPoolExecutor(max_workers=nums_cams) as executor:
            for cam_id, cam in enumerate(cam_array):
                print(f"cam id creating thread: {cam_id}")
                executor.submit(trigger_and_capture, cam, cam_id, exp_time, running, delay_dict)
    except Exception as e:
        print(f"An error occured during the initialization of threads for cams: {e}")
    cam_array.StopGrabbing()
    print(f"executor shutdown")
    executor.shutdown(wait=True)  # Stop the executor
    print(f"executor shutdowned")
    start_times = list(delay_dict.values())

    # Sort the start times
    start_times.sort()

    # Calculate the delays
    delays = [start_times[i] - start_times[i - 1] for i in range(1, len(start_times))]

    # Print the delays
    for i in range(len(delays)):
        print(f"Delay between camera {i} and camera {i + 1}: {delays[i]} seconds")
    if p.is_alive(): # If the process is still running, terminate it
        p.terminate()
    p.join()


def run_devices_test(cam_array, nums_cams, args):
    """
    this function stands for the testing 8 multiple cameras without 8 devices. ONLY FOR TEST PURPOSES
    """
    q = Queue()
    p = Process(target=run_inference, args=(q, args, running))
    p.start()
    exp_time = 0
    cam_array.StartGrabbing(py.GrabStrategy_LatestImageOnly) # ??

    # Define a function to be run in each thread
    def trigger_and_capture(cam, cam_id, exp_time, running, delay_dict):
        capture_amount = 0
        while running.is_set():
        # for _ in range(1):
            exp_time = args.exposure_time[int(capture_amount % 2)]
            cam.ExposureTime.SetValue(int(exp_time))
            cam.ExecuteSoftwareTrigger()
            grabResult = cam.RetrieveResult(1000, py.TimeoutHandling_ThrowException)
            # print(f"grabbed something")
            if True:
                print(f"image captured from cam:{cam_id} with exp_time: {exp_time}")
                img = grabResult
                if part_detection(img, args.gray_thres):
                    print(f"image put in queue")
                    capture_amount += 1
                    q.put((img, cam_id, exp_time, capture_amount))
                    time.sleep(args.interval)
                else:
                    print(f"No Part detected, checking in every {args.check_interval} seconds")
                    time.sleep(args.check_interval)
            delay_dict[cam_id] = time.time()
            # print(f"camera {cam_id} captured image in {time.time()}")
        print(f"Thread with cam_id {cam_id} stopped")
    delay_dict = {}
    with ThreadPoolExecutor(max_workers=nums_cams) as executor:
        for cam_id, cam in enumerate(cam_array):
            # print(f"capture amount: {capture_amount}")
            executor.submit(trigger_and_capture, cam, cam_id, exp_time, running, delay_dict)
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

    # Print the delays
    for i in range(len(delays)):
        print(f"Delay between camera {i} and camera {i + 1}: {delays[i]} seconds")
    if p.is_alive(): # If the process is still running, terminate it
        p.terminate()
    p.join()

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
    di = py.DeviceInfo()
    devs = tlf.EnumerateDevices()
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
            cam.Width.SetValue(1280)
            cam.Height.SetValue(720)
            cam.TriggerSelector = "FrameStart"
            cam.TriggerMode.SetValue('On')
            cam.TriggerSource.SetValue('Software')
            # below 3 lines run the flashes on cameras
            cam.LineSelector.SetValue("Line2")
            cam.LineMode.SetValue("Output")
            cam.LineSource.SetValue("ExposureActive")
        return cam_array, num_cams
    else:
        print(f"No devices found")

    

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
    if len(os.listdir(args.out_dir)) > 0:
        for out in os.listdir(args.out_dir):
            os.remove(f"{args.out_dir}{out}")
        print(f"Deleted all files in out-dir")
    global running
    running = multiprocessing.Event()
    running.set()
    cam_array, num_cams = load_devices(args)
    print(f"Devices are loaded, running")
    run_devices(cam_array=cam_array, nums_cams=num_cams, args=args)

def run_engine_test(args):
    # TODO run_this
    global running
    running = multiprocessing.Event()
    running.set()
    cam_array = MockCameraArray(8)
    print(f"Devices are loaded, running")
    run_devices_test(cam_array=cam_array, nums_cams=8, args=args)


def run_test(args):
    if len(os.listdir(args.out_dir)) > 0:
        for out in os.listdir(args.out_dir):
            os.remove(f"{args.out_dir}{out}")
        print(f"Deleted all files in out-dir")
    run_engine_test(args)

if __name__ == "__main__":
    set_start_method('spawn') # this is temp
    args = parse_args()
    if args.test:
        print(f"Running test")
        run_test(args)
    else:
        run_engine(args)
