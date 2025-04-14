# Pallet and Ground Detection-Segmentation Pipeline

## Overview

This repository contains a pipeline for detecting and segmenting pallets and ground surfaces in industrial environments. The pipeline includes:

1. Dataset preparation and augmentation  
2. Object detection and semantic segmentation model development  

---

## Dataset Preparation

I used **DINO** to annotate 18 images from the provided pallet dataset. A simple script was written to split the annotated `.png` and `.json` files into training, validation, and test folders.

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

```bash
python prepare_segmentation_data.py <source_detection_dataset_path> <output_segmentation_dataset_path>
```

---

## Train the Segmentation Model

A YOLOv8-compatible segmentation model was trained using the prepared dataset:

```bash
python train_segmentation.py path/to/segmentation.yaml --epochs 100 --imgsz 640 --batch 4 --patience 15
```

**Arguments:**

- `--epochs`: Number of training epochs  
- `--imgsz`: Input image resolution (default: 640)  
- `--batch`: Batch size  
- `--patience`: Early stopping after N epochs with no improvement  

The final trained weights (best.pt) from the segmentation model are included in the repository and should be used for inference during testing.

---

## Test the Model

Inference was run on the full set of 150 images provided in the pallet dataset:

```bash
python test_model.py <model_path> <image_dir> <output_dir>
```

Despite training on just 18 images, the model produced encouraging results on the full dataset.

---

## ROS2 Node

ROS2 Node Development
I started working on the pallet_detection_node, but couldn’t complete the full ROS2 integration due to compatibility issues on macOS. Some ROS2 Humble dependencies and tools required for real-time testing weren’t fully supported. I’ve uploaded the code I wrote so far, which includes the basic structure for subscribing to image topics and running model inference. This can be further developed and tested on a Linux system.
---

## Notes

- Dataset annotations were done using DINO  
- Model was trained and tested locally  
- Results and sample outputs can be found in the `results/` folder  

