# Pallet and Ground Detection-Segmentation Pipeline

## Overview

This repository contains a pipeline for detecting and segmenting pallets and ground surfaces in industrial environments. The pipeline includes:

1. Dataset preparation and augmentation  
2. Object detection and semantic segmentation model development  

---

## Dataset Preparation

I used **DINO** to annotate 20 images from the provided pallet dataset. A simple script was written to split the annotated `.png` and `.json` files into training, validation, and test folders.

I supplemented this dataset with data from [RoboFlow](https://universe.roboflow.com/david-akhihiero-pvxdr/pallet_and_ground)

---

## Data Augmentation

To improve model robustness, data augmentation was applied using the following script:

```bash
python augment.py <path_to_data_dir> <path_to_output_dir> --augmentation_factor 5
```

- `<path_to_data_dir>`: Directory containing the raw images  
- `<path_to_output_dir>`: Directory where augmented images will be saved  
- `--augmentation_factor`: Number of augmented versions generated per image

---

## Prepare Segmentation Dataset

The next step converts the detection-style dataset into a format suitable for segmentation model training:

For the provided data I used - 
```bash
python prepare_segmentation_data.py <source_detection_dataset_path> <output_segmentation_dataset_path>
```

For the supplementary data I used - 
```bash
python coco_yolo.py
```

---

## Train the Segmentation Model

A YOLOv8-compatible segmentation model was trained using the prepared dataset:

```bash
python train_segmentation.py path/to/data.yaml --epochs 100 --imgsz 640 --batch 4 --patience 15
```

**Arguments:**

- `--epochs`: Number of training epochs  
- `--imgsz`: Input image resolution (default: 640)  
- `--batch`: Batch size  
- `--patience`: Early stopping after N epochs with no improvement  

The final trained weights (model_best.pt) from the segmentation model are included in the repository and should be used for inference during testing.

---

## Test the Model

Inference was run on the full set of 150 images provided in the pallet dataset:

```bash
python test_model.py <model_path> <image_dir> <output_dir>
```

The model produced encouraging results on the test dataset.

---

## ROS2 Node Development
The ROS 2 node was developed to perform real-time inference using the trained YOLOv8 segmentation model. The node subscribes to an image topic, runs segmentation inference, and publishes or displays the result using OpenCV.

To simplify launching the node, a ROS 2 launch file is included:

```bash
ros2 launch pall_detect pallet_launch.py model_path:= <model_path> 
```
The model path can be passed as an argument using model_path:=...

Inside the node, you can also adjust the confidence threshold to filter predictions (default: 0.5). 


---

peer\_vision/ ├── coco/                         # Supplementary dataset │   ├── data.yaml │   └── pallet\_and\_ground.v9i.coco-segmentation.zip ├── pall\_detect/                  # ROS 2 package for real-time detection │   ├── launch/                   # Launch files for running the node │   ├── models/                   # Trained YOLOv8 model weights (e.g., best.pt) │   ├── pall\_detect/              # ROS 2 node implementation (Python) │   ├── requirements.txt          # Python dependencies │   ├── setup.py / setup.cfg      # ROS 2 Python package setup │   ├── package.xml               # ROS 2 package metadata │   └── resource/                 # ROS 2 resource directory (e.g., index files) ├── scripts/                      # Utility scripts for training and data prep │   ├── augment.py │   ├── coco\_yolo.py │   ├── prepare\_segmentation\_data.py │   └── train\_segmentation.py ├── README.md                     # Project documentation&#x20;

