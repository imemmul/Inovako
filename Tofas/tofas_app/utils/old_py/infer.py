from pypylon import pylon
import time
import datetime
import cv2
import numpy as np
import torch
import argparse
from pathlib import Path
from models import TRTModule
from Inovako.Tofas.tofas_app.engine.config import CLASSES, COLORS, MASK_COLORS
from models.torch_utils import seg_postprocess
from models.utils import blob, letterbox
import threading
import queue

class BaslerCamera:
    def __init__(self, exposure_time, device, event) -> None:
        self.camera = self.open_camera(exposure_time, device)
        self.event = event
        self.device_id = device.GetFriendlyName()

    def open_camera(self, exposure_time, device):
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device))
        camera.Open()
    
        camera.LineSelector.SetValue("Line2")
        camera.LineMode.SetValue("Output")
        camera.LineSource.SetValue("ExposureActive")
        camera.LineInverter.SetValue(True)
        camera.ExposureTime.SetValue(int(exposure_time))
        camera.MaxNumBuffer = 1

        return camera

    def set_exposure_time(self, new_value):
        self.camera.ExposureTime.SetValue(int(new_value))

    def take_images(self, count, image_queue):
        print(f"ID: {self.device_id}")
        # FIXME
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_RGB8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        result = self.camera.GrabOne(500)
        image = converter.Convert(result)
        result.Release()

        image_queue.put((image.Array, count))

        self.event.set()

def capture_images(exposure_t, image_q, interval):
    devices = pylon.TlFactory.GetInstance().EnumerateDevices()
    threads = []
    count = 0
    event = threading.Event()
    capture_flag = threading.Event()

    def capture_thread(camera):
        while running.is_set():
            event.wait()
            if not capture_flag.is_set():
                break
            camera.take_images(count, image_q) #3000, 5000
            event.clear()
            time.sleep(interval)

    for index, device in enumerate(devices):
        print(f"what is device: {index}: {device.GetFriendlyName()}")
        camera = BaslerCamera(exposure_t, device=device, event=event)
        thread = threading.Thread(target=capture_thread, args=(camera,))
        count += 1
        threads.append(thread)
        thread.start()

    while running.is_set():
        capture_flag.set()
        event.set()
        time.sleep(interval)
        event.clear()
        capture_flag.clear()

def process_images(image_queue, result_queue, device, Engine, save_path, W, H, args):
    while running.is_set():
        image, count = image_queue.get()

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        draw = bgr.copy()
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        dw, dh = int(dwdh[0]), int(dwdh[1])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor, seg_img = blob(rgb, return_seg=True)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.asarray(tensor, device=device)

        data = Engine(tensor)

        seg_img = torch.asarray(seg_img[dh:H - dh, dw:W - dw, [2, 1, 0]], device=device)
        bboxes, scores, labels, masks = seg_postprocess(data, bgr.shape[:2], args.conf_thres, args.iou_thres)

        if len(bboxes) == 0:
            print("No objects detected.")
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

        result_queue.put((draw, count))

def save_images(result_queue, save_path):
    count_dict = {}

    while running.is_set():
        image, thread_id = result_queue.get()

        if thread_id not in count_dict:
            count_dict[thread_id] = 0

        count = count_dict[thread_id]
        count_dict[thread_id] += 1

        cv2.imwrite(str(save_path / f"{count + 1}.jpg"), image)

def main(args):
    global running
    running = threading.Event()
    running.set()

    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    Engine.set_desired(['outputs', 'proto'])

    save_path = Path(args.out_dir)

    if not args.show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    exposure_time = args.exposure_time

    image_queue = queue.Queue()
    result_queue = queue.Queue()
    process_thread = threading.Thread(target=process_images, args=(image_queue, result_queue, device, Engine, save_path, W, H, args))
    process_thread.start()

    save_thread = threading.Thread(target=save_images, args=(result_queue, save_path))
    save_thread.start()

    capture_thread = threading.Thread(target=capture_images, args=(exposure_time,image_queue,args.interval/1000))  # Interval'ı saniye cinsinden uygun hale getir
    capture_thread.start()

def stop_threads():
    global running
    running.clear()
    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='./tofas_model.engine')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--out-dir', type=str, default='./output')
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.65)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exposure-time', type=int, default=3000)
    parser.add_argument('--interval', type=int, default=1000)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
