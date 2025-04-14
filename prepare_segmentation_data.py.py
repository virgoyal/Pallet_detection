import os
import glob
import shutil
import numpy as np

def convert_detection_to_segmentation(txt_file_path, output_dir, img_width=640, img_height=640):
    """
    Convert YOLO detection format to YOLO segmentation format.
    Since we don't have the original polygons, we'll just make rectangular masks from the bounding boxes.
    """
    try:
        # make the output dir if it doesn't exist already
        os.makedirs(output_dir, exist_ok=True)
        
        # get the file name without the path
        base_name = os.path.basename(txt_file_path)
        output_file = os.path.join(output_dir, base_name)
        
        # open the detection file and read lines
        with open(txt_file_path, 'r') as f:
            lines = f.readlines()
        
        # create the segmentation file and write polygons
        with open(output_file, 'w') as f:
            for line in lines:
                parts = line.strip().split()
                
                if len(parts) < 5:
                    print(f"Warning: Line format is incorrect in {txt_file_path}, skipping")
                    continue
                
                # extract the class id and bounding box values
                class_id = parts[0]
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # calculate rectangle corners in normalized coordinates
                x_min = max(0, x_center - width/2)
                y_min = max(0, y_center - height/2)
                x_max = min(1, x_center + width/2)
                y_max = min(1, y_center + height/2)
                
                # create polygon points for rectangle (clockwise order)
                polygon_points = [
                    (x_min, y_min),  # top-left
                    (x_max, y_min),  # top-right
                    (x_max, y_max),  # bottom-right
                    (x_min, y_max)   # bottom-left
                ]
                
                # format the line for the segmentation
                seg_line = f"{class_id}"
                
                # add the polygon points to the line
                for x, y in polygon_points:
                    seg_line += f" {x:.6f} {y:.6f}"
                
                # write the formatted line to the file
                f.write(seg_line + "\n")
                
        print(f"Created segmentation file: {output_file}")
        
    except Exception as e:
        print(f"Error processing {txt_file_path}: {str(e)}")

def create_segmentation_dataset(src_data_dir, dst_data_dir):
    """
    Create a segmentation dataset from a YOLO detection dataset.
    - src_data_dir: Root directory of the detection dataset
    - dst_data_dir: Root directory where to save the segmentation dataset
    """
    # make the destination directory if it doesn't exist
    os.makedirs(dst_data_dir, exist_ok=True)
    
    # process the train, val, and test splits
    for split in ['train', 'val', 'test']:
        # define source dirs for images and labels
        src_images_dir = os.path.join(src_data_dir, split, 'images')
        src_labels_dir = os.path.join(src_data_dir, split, 'labels')
        
        # define destination dirs for images and labels
        dst_split_dir = os.path.join(dst_data_dir, split)
        dst_images_dir = os.path.join(dst_split_dir, 'images')
        dst_labels_dir = os.path.join(dst_split_dir, 'labels')
        
        # create destination dirs
        os.makedirs(dst_images_dir, exist_ok=True)
        os.makedirs(dst_labels_dir, exist_ok=True)
        
        # copy over the images
        image_files = glob.glob(os.path.join(src_images_dir, '*.*'))
        for img_file in image_files:
            dst_img = os.path.join(dst_images_dir, os.path.basename(img_file))
            shutil.copy(img_file, dst_img)
            print(f"Copied image: {os.path.basename(img_file)}")
        
        # convert the label files
        label_files = glob.glob(os.path.join(src_labels_dir, '*.txt'))
        for txt_file in label_files:
            convert_detection_to_segmentation(txt_file, dst_labels_dir)

def main():
    # define paths - update these based on where your data is
    src_data_dir = "C:/Users/vgoyal/Desktop/peer/data_augmented"
    dst_data_dir = "C:/Users/vgoyal/Desktop/peer/seg_data_augmented"
    
    # create the segmentation dataset
    create_segmentation_dataset(src_data_dir, dst_data_dir)
    
    # create a YAML file for the dataset
    yaml_content = f"""
# Segmentation dataset config
path: {dst_data_dir}  # path to the segmentation data
train: train/images
val: val/images
test: test/images

# Number of classes
nc: 2

# Class names
names:
  0: floor
  1: pallet
"""
    
    # write the YAML config to file
    with open(os.path.join(dst_data_dir, "segmentation.yaml"), 'w') as f:
        f.write(yaml_content)
    
    print("Created segmentation.yaml")

if __name__ == "__main__":
    main()
