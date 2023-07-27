import cv2
import matplotlib.pyplot as plt
import numpy as np
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
import imgaug.augmenters as iaa
import imgaug as ia
import cv2
import os
import imageio
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
from imgaug.parameters import Uniform

OUTPUT_DIR = "/Users/emirulurak/Desktop/dev/Inovako_folders/augmented_dataset_tofas/"
INPUT_DIR = "/Users/emirulurak/Desktop/dev/Inovako_folders/dataset_tofas/"

def read_annotation_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    annotations = []
    for line in lines:
        # Split the line into numbers
        numbers = list(map(float, line.split()))
        
        # The first number is the class ID, the rest are coordinates
        class_id = int(numbers[0])
        coordinates = numbers[1:]

        annotations.append([class_id] + coordinates)
    
    return annotations

def adjust_annotations(original_image_width, original_image_height, original_annotations, crop_box, crop_size):
    adjusted_annotations = []

    for annotation in original_annotations:
        class_id = annotation[0]
        polygon_points = annotation[1:]

        adjusted_polygon = [class_id]

        for i in range(0, len(polygon_points), 2):
            # Convert to pixel coordinates
            x_point = polygon_points[i] * original_image_width
            y_point = polygon_points[i+1] * original_image_height

            # Adjust coordinates
            x_point -= crop_box[0]
            y_point -= crop_box[1]

            # Check if point is within the crop
            if 0 < x_point < crop_size[0] and 0 < y_point < crop_size[1]:
                # Normalize coordinates
                x_point /= crop_size[0]
                y_point /= crop_size[1]
                
                adjusted_polygon.extend([x_point, y_point])
        
        if len(adjusted_polygon) > 3:  # Only include polygons with at least one point inside the crop
            adjusted_annotations.append(adjusted_polygon)

    return adjusted_annotations

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

from PIL import Image
def crop_images_and_annotations(img_dir, annotation_dir, output_dir):
    # Open an image file
    with Image.open(img_dir) as img:
        width, height = img.size
        print(f"width: {width}")
        print(f"heights: {height}")

        # Define your step sizes
        x_step = 640 - 150
        y_step = 640 - 144

        # Define your crop box size
        box_size = (640, 640)

        # Read the original annotations
        original_annotations = read_annotation_file(annotation_dir)

        fig, axs = plt.subplots(4, 5, figsize=(20, 20))

        x_count = 0
        for i in range(0, width-150, x_step):
            y_count = 0
            for j in range(0, height-144, y_step):
                box = (i, j, i + box_size[0], j + box_size[1])
                crop_img = img.crop(box)
                axs[y_count, x_count].imshow(crop_img)
                axs[y_count, x_count].axis('off')
                axs[y_count, x_count].set_title(f'Cropped Image: {y_count}_{x_count}')

                # Define the output paths for the cropped image and annotation file
                img_output_path = os.path.join(output_dir, f'crop_{y_count}_{x_count}.jpg')
                annotation_output_path = os.path.join(output_dir, f'crop_{y_count}_{x_count}.txt')

                # Save the cropped image
                crop_img.save(img_output_path)

                # Adjust the annotations for the crop
                adjusted_annotations = adjust_annotations(original_image_height=height, original_image_width=width, original_annotations=original_annotations, crop_box=box, crop_size=box_size)

                # Write the adjusted annotations to a new file
                with open(annotation_output_path, 'w') as file:
                    for annotation in adjusted_annotations:
                        # Join the numbers in the annotation into a string, with spaces in between
                        line = ' '.join(map(str, annotation))
                        file.write(line + '\n')

                y_count += 1
            x_count += 1

def draw_annotations(img_dir, annot_dir, aug:bool):

    print(img_dir)
    img = cv2.imread(img_dir)

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
def augment_data(dataset_dir, augment_dir):
    aug = iaa.Sequential([
        iaa.Resize({"height": 864, "width": 864}),
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Flipud(0.2),  # vertical flips
        iaa.GaussianBlur(sigma=(0.0, 1.0)),  # Apply Gaussian Blur
        iaa.AdditiveGaussianNoise(scale=0.005*255),  # Add gaussian noise
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # zoom in & zoom out with scale of 0.8 to 1.2 times.
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45),  # rotate by -45 to +45 degrees
        ),  
    ], random_order=True)

    
    splits = ["train", "valid"]
    # Process each image
    for split in splits:
        for filename in os.listdir(f"{dataset_dir}{split}/images/"):
        # filename = "v2-2-2-_bmp.rf.a6319e97b2af489206fd056d9f7f6c15.jpg"
            if filename.endswith(".jpg"):
                # Read image
                img = cv2.imread(os.path.join(f"{dataset_dir}{split}/images/", filename))
                img = cv2.resize(img, (864, 864))
                img_height, img_width = img.shape[:2]
                print(img_height)
                print(img_width)
                path_to_polygons = os.path.join(f"{dataset_dir}{split}/labels/", f"{filename[:-3]}txt")
                with open(path_to_polygons, 'r') as file:
                    lines = file.readlines()
                    polygons = [Polygon([(float(coord.split()[i])*img_width, float(coord.split()[i+1])*img_height) 
                                            for i in range(1, len(coord.split()), 2)], label=coord.split()[0]) 
                                    for coord in lines]
                augment_count = 1  # default count
                for polygon in polygons:
                    if polygon.label == "0":  # or use '0' if your labels are strings
                        augment_count = 10  # the count for the desired class
                        break
                print(f"augment count is : {augment_count}.")
                for i in range(augment_count):
                    img_aug, polygons_aug = aug(image=img, polygons=PolygonsOnImage(polygons, shape=img.shape))
                    # polygons_aug = polygons_aug.clip_out_of_image()

                    # Filter out polygons that are out-of-bound
                    valid_polygons = []
                    valid_labels = []
                    for polygon in polygons_aug.polygons:
                        coords = polygon.exterior
                        # print(f"what is polygons: {coords}")
                        label = polygon.label
                        if all(0 <= point[0] < 864 and 0 <= point[1] < 864 for point in coords):
                            # print(f"what is label {label}")
                            valid_polygons.append(polygon)
                            valid_labels.append(label)
                    polygons_aug.polygons = valid_polygons

                    # Clip the coordinates of the augmented polygons
                    # print(f"polygons before for {filename} {polygons}")
                    # print(f"polygons after for {filename} {polygons_aug}")
                    
                    # Save the augmented image
                    img_save_dir = os.path.join(f"{augment_dir}{split}/images/", f"{filename[:-4]}_aug_{i}.jpg")
                    cv2.imwrite(os.path.join(f"{augment_dir}{split}/images/", f"{filename[:-4]}_aug_{i}.jpg"), img_aug)
                    print(f"saving image to {img_save_dir}")
                    # Convert PolygonsOnImage object to polygons and save
                    polygons_aug_txt = [f"{cls} {' '.join([str(float(coord[0])/864)+' '+str(float(coord[1])/864) for coord in polygons_aug.exterior])}"
                                        for polygons_aug, cls in zip(polygons_aug.polygons, valid_labels)]
                    annot_save_dir = os.path.join(f"{augment_dir}{split}/labels/", f"{filename[:-4]}_aug_{i}.txt")
                    with open(os.path.join(f"{augment_dir}{split}/labels/", f"{filename[:-4]}_aug_{i}.txt"), 'w') as file:
                        print(f"saving image to {annot_save_dir}")
                        file.write('\n'.join(polygons_aug_txt))
                    # draw_annotations(img_dir=img_save_dir, annot_dir=annot_save_dir, aug=True)
    
if __name__ == "__main__":
    dataset_dir = "/Users/emirulurak/Desktop/dev/Inovako_folders/dataset_tofas_v2/"
    splits = ["train", "valid"]
    augment_dataset_dir = "/Users/emirulurak/Desktop/dev/Inovako_folders/augmented_dataset_tofas_v2/"
    augment_data(dataset_dir=dataset_dir, augment_dir=augment_dataset_dir)
