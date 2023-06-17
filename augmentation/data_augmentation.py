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

OUTPUT_DIR = "/home/emir/Desktop/dev/Inovako/projet_secret/augmented_dataset/"
INPUT_DIR = "../dataset/"

def draw_annotations(img_dir, annot_dir, aug:bool):

    # Load your image
    img = cv2.imread(img_dir)
    img_height, img_width = img.shape[:2]


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
            points = np.array(parts[1:], dtype=np.float32)
            points = [(float(points[i]), float(points[i+1])) for i in range(0, len(points), 2)] # making them x,y pairs
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
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    else:
        for line in lines:
            parts = line.split()
            cls = int(parts[0])
            points = np.array(parts[1:], dtype=np.float32)
            points = [(float(points[i])*img_height, float(points[i+1])*img_width) for i in range(0, len(points), 2)] # making them x,y pairs
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
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    

import os
def augment_data(dir, target_dir):
    aug = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.2),
        iaa.Affine(rotate=(-10, 10)),
        iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
        iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
        iaa.Affine(scale=Uniform(0.8, 1.2)),
        iaa.Affine(shear=(-8,8)),
        iaa.Multiply((0.8, 1.2)),
        iaa.GaussianBlur(sigma=(0.0, 0.01)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.005*255)),
        iaa.ElasticTransformation(alpha=10, sigma=5),
        iaa.Cutout(nb_iterations=(1, 3), size=0.01, squared=False),
        iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
        iaa.ChannelShuffle(1.0)
    ])

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
            for i in range(300):
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
                cv2.imwrite(os.path.join(target_dir, f"images/aug_{i}_" + filename), img_aug)
                print(f"labels/aug_{i}_")
                # Convert PolygonsOnImage object to polygons and save
                polygons_aug_txt = [f"{cls} {' '.join([str(float(coord[0]))+' '+str(float(coord[1])) for coord in polygons_aug.exterior])}"
                                    for polygons_aug, cls in zip(polygons_aug.polygons, valid_labels)]
                with open(os.path.join(target_dir, f"labels/aug_{i}_" + filename.rsplit(".", 1)[0] + ".txt"), 'w') as file:
                    file.write('\n'.join(polygons_aug_txt))



if __name__ == "__main__":
    train_dir = f"{INPUT_DIR}train/"
    valid_dir = f"{INPUT_DIR}valid/"
    augment_data(train_dir, target_dir=OUTPUT_DIR+"train/")
    augment_data(valid_dir, target_dir=OUTPUT_DIR+"valid/")