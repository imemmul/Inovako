import cv2
import matplotlib.pyplot as plt
import numpy as np
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
import imgaug.augmenters as iaa
import imgaug as ia
import cv2
import os
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
from imgaug.parameters import Uniform

OUTPUT_DIR = "/home/emir/Desktop/dev/Inovako/Inovako/dataset_tofas_augmented/"
INPUT_DIR = "/home/emir/Desktop/dev/Inovako/dataset_tofas/tofas_main/"

# TODO iaa.RandomCrop 
def convert_dataset_class(annot_dir):
    with open(annot_dir) as f:
        lines = f.read().splitlines()
    new_lines = []
    for line in lines:
        info = line.split()
        # print(not bool(int(info[0])))
        if not bool(int(info[0])):
        #    print(f"old class is {info[0]}")
           info[0] = str(1)
        #    print(f"old class is {str(1)}")
        # print(info)
        new_line = ' '.join(info)
        new_lines.append(new_line + '\n')
        # print(info)
   
    with open(annot_dir, 'w') as f:
        f.writelines(new_lines)




def draw_annotations(img_dir, annot_dir, aug:bool):

    print(img_dir)
    img = cv2.imread(img_dir, 0)

    print(f"img_shape: {img.shape}")
    img_height, img_width = img.shape[:2]
    print(f"img_height: {img_height}")
    print(f"img_width: {img_width}")
    # Open your text file

    with open(annot_dir) as f:
        lines = f.read().splitlines()

    # Parse the text file and draw bounding boxes
    # polys = []
    clr = (0, 0, 0)
    if aug:
        for line in lines:
            parts = line.split()
            cls = int(parts[0])
            print(f"what is class: {cls}")
            points = np.array(parts[1:], dtype=np.float32)
            points = [(float(points[i]), float(points[i+1])) for i in range(0, len(points), 2)] # making them x,y pairs
            # print(f"points {points}")
            poly = [Polygon(points, label=cls)]
            # print(f" what is class {cls}")
            if bool(cls): # cls == 1 hole
                clr = (0, 255, 0)
            else: # cls == 0  crack
                clr(255, 0, 0)
            # print(f"what is color {clr}")
            polys_oi = PolygonsOnImage(poly, shape=img.shape)
            
            img = polys_oi.draw_on_image(img, alpha_points=0, alpha_lines=1, alpha_face=0.5, color=clr)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        # print(f"img shape {img.shape}")
    else:
        for line in lines:
            parts = line.split()
            cls = int(parts[0])
            points = np.array(parts[1:], dtype=np.float32)
            points = [(float(points[i])*img_width, float(points[i+1])*img_height) for i in range(0, len(points), 2)] # making them x,y pairs
            # print(f"points {points}")
            poly = [Polygon(points, label=cls)]
            # print(f" what is class {cls}")
            if 2 == cls:
                clr = (255, 0, 0)
            elif 1 == cls:
                clr = (0, 255, 0)
            else:
                clr = (0, 0, 255)
            # print(f"what is color {clr}")
            polys_oi = PolygonsOnImage(poly, shape=img.shape)
            
            img = polys_oi.draw_on_image(img, alpha_points=0, alpha_lines=1, alpha_face=0.5, color=clr)
        print(f"img shape {img.shape}")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

import os
def augment_data(dir, target_dir):
    aug = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.2),
        iaa.Affine(scale=Uniform(0.5, 1.5)),
        iaa.Multiply((0.8, 1.2)),
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.1))
        ),  # Gaussian blur for 10% of the images
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
        iaa.Affine(rotate=(-25, 25))
        
    ], random_order=True)

    # Specify the paths to your images and masks
    path_to_images = f"{dir}images/"
    path_to_polygons = f"{dir}labels/"
    # Process each image
    for filename in os.listdir(path_to_images):
    # filename = "v2-2-2-_bmp.rf.a6319e97b2af489206fd056d9f7f6c15.jpg"
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Read image
            img = cv2.imread(os.path.join(path_to_images, filename))
            img_height, img_width = img.shape[:2]
            # Read corresponding polygon points
            with open(os.path.join(path_to_polygons, filename.rsplit(".", 1)[0] + ".txt"), 'r') as file:
                lines = file.readlines()
                polygons = [Polygon([(float(coord.split()[i])*img_height, float(coord.split()[i+1])*img_width) 
                                        for i in range(1, len(coord.split()), 2)], label=coord.split()[0]) 
                                for coord in lines]

            # Augment image and polygons 400 times for each image
            for i in range(40):
                img_aug, polygons_aug = aug(image=img, polygons=PolygonsOnImage(polygons, shape=img.shape))
                # polygons_aug = polygons_aug.clip_out_of_image()

                # Filter out polygons that are out-of-bound
                valid_polygons = []
                valid_labels = []
                for polygon in polygons_aug.polygons:
                    coords = polygon.exterior
                    # print(f"what is polygons: {coords}")
                    label = polygon.label
                    if all(0 <= point[0] < img.shape[1] and 0 <= point[1] < img.shape[0] for point in coords):
                        # print(f"what is label {label}")
                        valid_polygons.append(polygon)
                        valid_labels.append(label)
                polygons_aug.polygons = valid_polygons

                # Clip the coordinates of the augmented polygons
                # print(f"polygons before for {filename} {polygons}")
                # print(f"polygons after for {filename} {polygons_aug}")
                print(f"images/aug_{i}_")
                # Save the augmented image
                cv2.imwrite(os.path.join(target_dir, f"images/" + f"{filename[:-4]}_aug_{i}.jpg"), img_aug)
                print(f"labels/aug_{i}_")
                # Convert PolygonsOnImage object to polygons and save
                polygons_aug_txt = [f"{cls} {' '.join([str(float(coord[0]))+' '+str(float(coord[1])) for coord in polygons_aug.exterior])}"
                                    for polygons_aug, cls in zip(polygons_aug.polygons, valid_labels)]
                with open(os.path.join(target_dir, f"labels/" + filename.rsplit(".", 1)[0] + f"_aug_{i}.txt"), 'w') as file:
                    file.write('\n'.join(polygons_aug_txt))



if __name__ == "__main__":
    train_dir = f"{INPUT_DIR}train/"
    valid_dir = f"{INPUT_DIR}valid/"
    augment_data(train_dir, target_dir=OUTPUT_DIR+"train/")
    augment_data(valid_dir, target_dir=OUTPUT_DIR+"valid/")