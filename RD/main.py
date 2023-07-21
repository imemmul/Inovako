import cv2
import torch
from models import TRTModule
from config import CLASSES, COLORS, MASK_COLORS, ALPHA
from models.torch_utils import seg_postprocess
from models.utils import blob, letterbox
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


def find_match(img1, img2):
    # Initialize the SURF detector algorithm
    orb = cv2.SIFT_create()

    # Find keypoints and descriptors with SURF
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw only "good" matches (i.e., whose distance is less than 0.7*min_dist )
    # and return keypoints in the second image that are matched
    good = []
    for m in matches:
        good.append(m)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Draw matches
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,flags=2)

    return img3, dst_pts


def run_inference(conf_thres, iou_thres):
    count = 0
    try:
        engine, device, H, W  = load_engine()
        for image_dir in range(1):
            image = cv2.imread(dataset_dir[0])
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
                    # cv2.imwrite(filename=f"./output/output_{count}_NO_DET.jpg", img=image)
                    print(f"nothing detected")
                else:
                    print(f"what is bboxes type: {type(bboxes)}")
                    random_index = np.random.randint(0, len(bboxes) - 1)
                    masks = masks[random_index][None, dh:H - dh, dw:W - dw, :]
                    indices = (labels[random_index] % len(MASK_COLORS)).long()
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

                    bbox = bboxes[random_index].round().int().tolist()
                    score = scores[random_index]
                    label = labels[random_index]
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
                    img2 = cv2.imread(dataset_dir[1])
                    margin = 100
                    margine_bbox = bbox.copy()
                    # Ensure bbox is within image boundaries
                    margine_bbox[0] = max(0, bbox[0] - margin)  # x1
                    margine_bbox[1] = max(0, bbox[1] - margin)  # y1
                    margine_bbox[2] = min(image.shape[1], bbox[2] + margin)  # x2
                    margine_bbox[3] = min(image.shape[0], bbox[3] + margin)  # y2
                    
                    # Cropping the detected hole from the original image
                    cropped = image[margine_bbox[1]:margine_bbox[3], margine_bbox[0]:margine_bbox[2]]
                    cv2.imwrite(f"./output/cropped_{count}.jpg", cropped)
                    #print(f"trying to save det here: {args.out_dir}run_{run_id}/{devices[cam_id]}/{exp_time}/DET/output_{capture_id}.jpg")
                    cv2.imwrite(filename=f"./output/output_{count}.jpg", img=draw)
                    result, dst_pts = find_match(img1=cropped, img2=img2)
                    x_coords = dst_pts[:,0,0]
                    y_coords = dst_pts[:,0,1]
                    bbox = [np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords)]

                    # Draw the bounding box and segmentation mask in the second image
                    # You might need to adjust the parameters to fit your case
                    bbox = [int(coord) for coord in bbox]
                    cv2.rectangle(img2, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 3)
                    cv2.putText(img2, 'Hole', (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                    # Save the result
                    cv2.imwrite('macthed_images.jpg', img2)
                    count += 1
                # print(f"Its been {end_time-start_time} seconds to process cam: {cam_id}")
            except Exception as e:
                print(f"Some error occured in run_inference {e}")
                traceback.print_exc()
    except Exception as e:
        print(f"Error in inference: {e}")


def find_keypoints_and_matches(img1, img2):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    return pts1, pts2, good

def calculate_pose(img1, img2):
    pts1, pts2, matches = find_keypoints_and_matches(img1, img2)

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([pts1], [pts2], img1.shape[::-1], None, None)

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(mtx, dist, mtx, dist, img1.shape[::-1], R1, T, flags=0)

    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, R1, P1, img1.shape[::-1], cv2.CV_32FC1)

    img1_rectified = cv2.remap(img1, map1, map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(img2, map1, map2, cv2.INTER_LINEAR)

    disparity = cv2.StereoBM_create(numDisparities=16, blockSize=15).compute(img1_rectified, img2_rectified)

    points = cv2.reprojectImageTo3D(disparity, Q)
    
    return points

def plot_3d_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = points[:, :, 0].flatten()
    y = points[:, :, 1].flatten()
    z = points[:, :, 2].flatten()

    ax.scatter(x, y, z)
    plt.show()



if __name__ == "__main__":
    # run_inference(conf_thres=0.25, iou_thres=0.65)
    # read images
    img1 = cv2.imread(dataset_dir[0], 0)
    img2 = cv2.imread(dataset_dir[1], 0)
    points = calculate_pose(img1, img2)