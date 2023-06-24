import pypylon.pylon as py
import time
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

# TODO write a no camera handler
# TODO handle the issue about set_start_method('spawn')
DEFAULT_EXPOSURE = 10000


def run_inference(q:Queue, args, running):
    engine, device, H, W  = load_engine(args)
    count = len(os.listdir(args.out_dir))
    while running.is_set():
        image, cam_id, exp_time = q.get()
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        draw = bgr.copy()
        print(f"what is shape: {draw.shape}")
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        dw, dh = int(dwdh[0]), int(dwdh[1])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor, seg_img = blob(rgb, return_seg=True)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)

        data = engine(tensor)

        seg_img = torch.asarray(seg_img[dh:H - dh, dw:W - dw, [2, 1, 0]], device=device)
        bboxes, scores, labels, masks = seg_postprocess(data, bgr.shape[:2], args.conf_thres, args.iou_thres)

        if len(bboxes) == 0:
            print("Nothing detected.")
        else:
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
            cv2.imwrite(filename=f"{args.out_dir}output_{cam_id}_{count}_{exp_time}.jpg", img=draw)

def part_detection(img, threshold):
    gray_value = np.mean(img)
    print(f"gray_value: {gray_value}")
    return gray_value > threshold

def run_devices(cam_array, nums_cams, args):
    q = Queue()
    p = Process(target=run_inference, args=(q, args, running))
    p.start()
    capture_amount = 0
    exp_time = 0
    cam_array.StartGrabbing(py.GrabStrategy_LatestImageOnly)
    while running.is_set():
        for cam in cam_array:
            exp_time = args.exposure_time[capture_amount % 2]
            print(f"what is exp_time: {int(exp_time)}")
            cam.ExposureTime.SetValue(int(exp_time))  # Set new exposure time
            # print(f"cam exposureTime: {cam.ExposureTime}")
        with cam_array.RetrieveResult(1000) as res:
            # print(f"image taken")
            if res.GrabSucceeded():
                print(f"what is image {res.Array.shape}")
                if part_detection(res.Array, args.gray_thres):
                    print(f"Part detected starting to capture")
                    cam_id = res.GetCameraContext()
                    # Run inference
                    print(f"captured image size of {res.Array.shape}")
                    q.put((res.Array, cam_id, exp_time))
                    capture_amount += 1
                    time.sleep(args.interval)
                else:
                    print(f"No Part detected, checking in every {args.check_interval} seconds")
                    time.sleep(args.check_interval)
    cam_array.StopGrabbing()
    p.terminate()
    q.put(None)  # signal the inference process to end

def load_engine(args):
    device = torch.device(args.device)
    # print(f"what is device {args.engine}")
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]
    Engine.set_desired(['outputs', 'proto'])
    print(f"engine is loaded")
    return Engine, device, H, W

def load_devices(args):
    tlf = py.TlFactory.GetInstance()
    di = py.DeviceInfo()
    devs = tlf.EnumerateDevices()
    num_cams = len(devs)
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
    return cam_array, num_cams

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default="/home/inovako/Inovako/emir_workspace/tensorrt_engines/tofas_engine/tofas_model.engine")
    parser.add_argument('--out-dir', type=str, default='../output/')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gray-thres', type=int, default=35)
    parser.add_argument('--exposure-time', type=list, default=[10000, 50000])
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.65)
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--check-interval', type=int, default=5)
    parser.add_argument('--test', action="store_true")
    args = parser.parse_args()
    return args

def stop_engine():
    global running
    running.clear()
    print(f"stopped the engine")

def run_engine(args):
    global running
    running = multiprocessing.Event()
    running.set()
    cam_array, num_cams = load_devices(args)
    print(f"Devices are loaded, running")
    run_devices(cam_array=cam_array, nums_cams=num_cams, args=args)

def run_test(args):
    if len(os.listdir(args.out_dir)) > 0:
        for out in os.listdir(args.out_dir):
            os.remove(f"{args.out_dir}{out}")
        print(f"Deleted all files in out dir")
    run_engine(args)

if __name__ == "__main__":
    set_start_method('spawn') # this is temp
    args = parse_args()
    if args.test:
        run_test(args)
    else:
        run_engine(args)
