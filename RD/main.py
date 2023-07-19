import cv2
import torch
from models import TRTModule
from config import CLASSES, COLORS, MASK_COLORS, ALPHA
from models.torch_utils import seg_postprocess
from models.utils import blob, letterbox
import numpy as np
import os
import traceback
# 13 14 is a good example to apply matching algoritm
# Basler_a2A2600-64umBAS__40359001__20230713_185427266_0012.tiff
# Basler_a2A2600-64umBAS__40359001__20230713_185427266_0013.tiff
# random select a bbox crop and match with other image ?
dataset_dir = ["/home/emir/Desktop/dev/Inovako/dataset_no_detect/Emin_NoDetect/1/Basler_a2A2600-64umBAS__40359001__20230713_185427266_0012.tiff", "/home/emir/Desktop/dev/Inovako/dataset_no_detect/Emin_NoDetect/1/Basler_a2A2600-64umBAS__40359001__20230713_185427266_0013.tiff"]
# dataset_dir =  "/home/emir/Desktop/dev/Inovako/dataset_no_detect/Emin_NoDetect/1/"
engine_dir = "/home/emir/Desktop/dev/Inovako/tensorrt_engines/tofas_model.engine"

def load_engine():
    try:
        device = torch.device("cuda:0")
        # print(f"what is device {args.engine}")
        Engine = TRTModule(engine_dir, device)
        H, W = Engine.inp_info[0].shape[-2:]
        Engine.set_desired(['outputs', 'proto'])
        return Engine, device, H, W
    except Exception as e:
        print(f"An error occured in load_engine: {e}")


def run_inference(conf_thres, iou_thres):
    count = 0
    try:
        engine, device, H, W  = load_engine()
        for image_dir in dataset_dir:
            image = cv2.imread(filename=image_dir)
            try:
                
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
                    data, bgr.shape[:2], conf_thres, iou_thres)
                
                if len(bboxes) == 0:
                    # print(f"trying to save no_det here: {args.out_dir}run_{run_id}/{devices[cam_id]}/{exp_time}/NO_DET/output_{capture_id}.jpg")
                    cv2.imwrite(filename=f"./output/output_{count}_NO_DET.jpg", img=image)
                    #print(f"nothing detected")
                else:
                    print(f"what is bboxes type: {type(bboxes)}")
                    random_idx = np.random.randint(len(bboxes))
                    updated_bbox = torch.Tensor(list(bboxes[random_idx])).to(device=device)
                    updated_scores = torch.Tensor([scores[random_idx]]).to(device=device)
                    updated_masks =  masks[random_idx].to(device=device)
                    updated_masks = updated_masks.unsqueeze(-1).expand(-1, -1, 3)
                    updated_labels = torch.Tensor([labels[random_idx]]).to(device=device)
                    updated_masks = updated_masks[:, dh:H - dh, dw:W - dw]
                    indices = (updated_labels % len(MASK_COLORS)).long()
                    mask_colors = torch.asarray(MASK_COLORS, device=device)[indices]
                    mask_colors = mask_colors.view(-1, 1, 1, 3) * ALPHA
                    mask_colors = updated_masks @ mask_colors
                    inv_alph_masks = (1 - updated_masks * 0.5).cumprod(0)
                    
                    mcs = (mask_colors * inv_alph_masks).sum(0) * 2
                    print(inv_alph_masks.shape)
                    print(mcs.shape)
                    print(seg_img.shape)
                    seg_img = (seg_img * inv_alph_masks[-1] + mcs) * 255
                    draw = cv2.resize(seg_img.cpu().numpy().astype(np.uint8),
                                    draw.shape[:2][::-1])

                    updated_bbox -= dwdh
                    updated_bbox /= ratio

                    for (bbox, score, label) in zip(updated_bbox, updated_scores, updated_labels):
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
                    count += 1
                    #print(f"trying to save det here: {args.out_dir}run_{run_id}/{devices[cam_id]}/{exp_time}/DET/output_{capture_id}.jpg")
                    cv2.imwrite(filename=f"./output/output_{count}.jpg", img=draw)
                # print(f"Its been {end_time-start_time} seconds to process cam: {cam_id}")
            except Exception as e:
                print(f"Some error occured in run_inference {e}")
                traceback.print_exc()
    except Exception as e:
        print(f"Error in inference: {e}")


if __name__ == "__main__":
    run_inference(conf_thres=0.25, iou_thres=0.65)