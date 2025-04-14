import os
import cv2
import numpy as np
import albumentations as A
import shutil
from tqdm import tqdm
import argparse  # Import argparse

def create_augmentation_pipeline():
    """set up a pipeline of simple transformations that won’t mess up our bounding boxes."""
    transform = A.Compose([
        # basic transformations: randomly rotate or flip the image
        A.RandomRotate90(p=0.3),  # rotate the image by 90 degrees 30% of the time
        A.HorizontalFlip(p=0.5),   # flip the image horizontally 50% of the time
        
        # modify the lighting: adjust brightness, contrast, and hue
        A.RandomBrightnessContrast(p=0.5),  # randomly tweak brightness and contrast
        A.HueSaturationValue(p=0.3),        # adjust hue, saturation, and value a little
        
        # add some noise or blur to make it more realistic
        A.GaussNoise(p=0.2),     # add some Gaussian noise to the image
        A.MotionBlur(p=0.2),     # apply motion blur to simulate camera movement
        
        # small shifts, scaling, and rotation to avoid over-distortion
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),  # small random shifts, rotations, and scaling
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
    
    return transform

def fix_bboxes(bboxes):
    """make sure our bounding boxes are within valid ranges and not too small or large."""
    fixed_bboxes = []
    
    for bbox in bboxes:
        if len(bbox) != 4:  # if the bounding box isn't in the correct format, we skip it
            continue
            
        x_center, y_center, width, height = bbox
        
        # keep the center of the bounding box within the image (between 0 and 1)
        x_center = max(0.001, min(0.999, x_center))
        y_center = max(0.001, min(0.999, y_center))
        
        # make sure the width and height don’t go out of bounds
        width = min(width, 2 * min(x_center, 1 - x_center))
        height = min(height, 2 * min(y_center, 1 - y_center))
        
        # ensure the width and height aren’t too small
        width = max(0.01, width)
        height = max(0.01, height)
        
        fixed_bboxes.append([x_center, y_center, width, height])
    
    return fixed_bboxes

def apply_augmentations(image_path, label_path, output_img_dir, output_label_dir, transform, aug_index):
    """apply augmentations to the image and bounding boxes, with some error handling just in case."""
    try:
        # read the image from file
        image = cv2.imread(image_path)
        if image is None:
            print(f"oops! couldn’t load the image: {image_path}")
            return False
        
        # read the bounding box labels from the file
        bboxes = []
        classes = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:  # if the line doesn't have enough info, we skip it
                    continue
                
                class_id = int(parts[0])
                try:
                    x_center, y_center, width, height = map(float, parts[1:5])
                    bboxes.append([x_center, y_center, width, height])
                    classes.append(class_id)
                except ValueError:
                    print(f"looks like there’s something wrong with the values in {label_path}")
                    continue
        
        if not bboxes:
            print(f"oops, no valid bounding boxes found in {label_path}")
            return False
        
        # fix any invalid bounding boxes
        bboxes = fix_bboxes(bboxes)
        
        if not bboxes:
            print(f"still no valid bounding boxes after fixing them in {label_path}")
            return False
        
        # apply the transformations to the image and bounding boxes
        try:
            transformed = transform(
                image=image,
                bboxes=bboxes,
                class_labels=classes
            )
            
            augmented_image = transformed['image']
            augmented_bboxes = transformed['bboxes']
            augmented_classes = transformed['class_labels']
        except Exception as e:
            print(f"something went wrong during augmentation for {image_path}: {str(e)}")
            return False
        
        if len(augmented_bboxes) == 0:
            print(f"uh-oh, all bounding boxes were removed after augmentation for {image_path}")
            return False
        
        # save the augmented image to disk
        base_name = os.path.basename(image_path)
        aug_img_name = f"{os.path.splitext(base_name)[0]}_aug_{aug_index}{os.path.splitext(base_name)[1]}"
        output_image_path = os.path.join(output_img_dir, aug_img_name)
        cv2.imwrite(output_image_path, augmented_image)
        
        # save the updated bounding boxes in the same format
        base_label = os.path.basename(label_path)
        aug_label_name = f"{os.path.splitext(base_label)[0]}_aug_{aug_index}.txt"
        output_label_path = os.path.join(output_label_dir, aug_label_name)
        
        with open(output_label_path, 'w') as f:
            for i, bbox in enumerate(augmented_bboxes):
                class_id = augmented_classes[i]
                x_center, y_center, width, height = bbox
                
                # make sure the bounding box values are still within range
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0.01, min(1, width))
                height = max(0.01, min(1, height))
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"created augmentation {aug_index} for {os.path.basename(image_path)}")
        return True
        
    except Exception as e:
        print(f"something went wrong while augmenting {image_path}: {str(e)}")
        return False

def augment_dataset(data_dir, output_dir, augmentation_factor=3):
    """augment the dataset by applying transformations and saving the new files."""
    # set up the augmentation pipeline
    transform = create_augmentation_pipeline()
    
    # make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # process the training and validation datasets
    for split in ['train', 'val']:
        print(f"starting with the {split} split...")
        
        # define the paths for images and labels
        images_dir = os.path.join(data_dir, split, 'images')
        labels_dir = os.path.join(data_dir, split, 'labels')
        
        output_images_dir = os.path.join(output_dir, split, 'images')
        output_labels_dir = os.path.join(output_dir, split, 'labels')
        
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)
        
        # copy over the original files first
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        for img_file in image_files:
            image_path = os.path.join(images_dir, img_file)
            label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')
            
            # copy the original images and labels to the new directory
            shutil.copy(image_path, os.path.join(output_images_dir, img_file))
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(output_labels_dir, os.path.splitext(img_file)[0] + '.txt'))
                print(f"copied the original file: {img_file}")
        
        # now, let’s apply the augmentations and create new versions
        for img_file in tqdm(image_files, desc=f"augmenting {split}"):
            image_path = os.path.join(images_dir, img_file)
            label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')
            
            if not os.path.exists(label_path):
                continue
            
            # generate multiple augmented versions of the image
            for i in range(augmentation_factor):
                apply_augmentations(
                    image_path, 
                    label_path,
                    output_images_dir,
                    output_labels_dir,
                    transform,
                    i + 1
                )
    
    # copy the test dataset without any changes
    test_images_dir = os.path.join(data_dir, 'test', 'images')
    test_labels_dir = os.path.join(data_dir, 'test', 'labels')
    
    output_test_images_dir = os.path.join(output_dir, 'test', 'images')
    output_test_labels_dir = os.path.join(output_dir, 'test', 'labels')
    
    os.makedirs(output_test_images_dir, exist_ok=True)
    os.makedirs(output_test_labels_dir, exist_ok=True)
    
    # just copy over all the test files
    for img_file in os.listdir(test_images_dir):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            shutil.copy(
                os.path.join(test_images_dir, img_file),
                os.path.join(output_test_images_dir, img_file)
            )
            
            label_file = os.path.splitext(img_file)[0] + '.txt'
            if os.path.exists(os.path.join(test_labels_dir, label_file)):
                shutil.copy(
                    os.path.join(test_labels_dir, label_file),
                    os.path.join(output_test_labels_dir, label_file)
                )

def create_yaml_files(output_dir):
    """generate the yaml configuration files for yolo training (detection and segmentation)."""
    
    # detection yaml (for object detection tasks)
    det_yaml_content = f"""
# yolo v8 detection dataset config
path: {output_dir}
train: train/images
val: val/images
test: test/images

# classes
nc: 2
names:
  0: floor
  1: pallet
"""
    
    # segmentation yaml (for segmentation tasks)
    seg_yaml_content = f"""
# yolo v8 segmentation dataset config
path: {output_dir}
train: train/images
val: val/images
test: test/images

# classes
nc: 2
names:
  0: floor
  1: pallet
"""
    
    # write the yaml config files
    with open(os.path.join(output_dir, "dataset_augmented.yaml"), 'w') as f:
        f.write(det_yaml_content)
        
    with open(os.path.join(output_dir, "segmentation_augmented.yaml"), 'w') as f:
        f.write(seg_yaml_content)
    
    print(f"yaml files have been created in {output_dir}")

if __name__ == "__main__":
    # Set up argparse to take input arguments
    parser = argparse.ArgumentParser(description="Augment YOLO dataset")
    parser.add_argument("data_dir", help="Path to the dataset")
    parser.add_argument("output_dir", help="Directory to save the augmented dataset")
    parser.add_argument("--augmentation_factor", type=int, default=3, help="Number of augmentations to generate per image (default is 3)")
    
    args = parser.parse_args()
    
    # start the dataset augmentation process
    augment_dataset(args.data_dir, args.output_dir, augmentation_factor=args.augmentation_factor)
    
    # generate the yaml files
    create_yaml_files(args.output_dir)
    
    print("\ndata augmentation is complete!")
    print(f"your augmented dataset is saved to: {args.output_dir}")
    print(f"use {os.path.join(args.output_dir, 'dataset_augmented.yaml')} for detection tasks")
    print(f"use {os.path.join(args.output_dir, 'segmentation_augmented.yaml')} for segmentation tasks")
