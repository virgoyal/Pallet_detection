import cv2
import os
import time
from ultralytics import YOLO
import numpy as np

def test_model_with_debug(model_path, image_dir, output_dir):
    """
    Run inference on a folder of images with enhanced debugging (no GUI).

    Args:
        model_path: Path to the YOLOv8 model weights
        image_dir: Directory containing test images
        output_dir: Directory to save annotated images
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Does model file exist? {os.path.exists(model_path)}")
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)

    print(f"Model task: {model.task}")
    print(f"Model names (classes): {model.names}")
    print(f"Model stride: {model.stride}")

    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Found {len(image_files)} images")

    for image_path in image_files:
        print(f"\nProcessing {image_path}")
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to read image: {image_path}")
            continue

        print(f"Image dimensions: {image.shape}")

        for conf_threshold in [0.5, 0.25, 0.1]:
            print(f"\nTrying with confidence threshold: {conf_threshold}")
            start_time = time.time()
            results = model(image, conf=conf_threshold, imgsz=640)
            inference_time = time.time() - start_time
            print(f"Inference time: {inference_time:.4f} seconds")

            result = results[0]

            # Masks
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks
                print(f"Number of masks: {len(masks)}")
                if len(masks) > 0:
                    print(f"Mask shape: {masks.data.shape}")
            else:
                print("No masks found")

            # Boxes
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                print(f"Number of boxes: {len(boxes)}")
                if len(boxes) > 0:
                    print(f"Confidence scores: {boxes.conf.tolist()}")
                    print(f"Class IDs: {boxes.cls.tolist()}")
                    for i, box in enumerate(boxes):
                        class_id = int(box.cls.item())
                        conf = box.conf.item()
                        cls_name = model.names[class_id]
                        print(f"  Detection #{i+1}: {cls_name} (ID: {class_id}), confidence: {conf:.4f}")
                        if hasattr(box, 'xyxy'):
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            print(f"    Bounding box: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")

            # Save annotated image if detections found
            if (hasattr(result, 'boxes') and len(result.boxes) > 0) or \
               (hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0):
                print(f"Detections found at threshold {conf_threshold}, saving results")

                # Save original
                orig_path = os.path.join(output_dir, f"original_{os.path.basename(image_path)}")
                cv2.imwrite(orig_path, image)

                # Get annotated image
                annotated_img = result.plot()

                # Save annotated image
                output_path = os.path.join(output_dir, f"detected_{conf_threshold}_{os.path.basename(image_path)}")
                cv2.imwrite(output_path, annotated_img)

            else:
                print(f"No detections at threshold {conf_threshold}")
                if conf_threshold == 0.1:
                    # Save original image once if no detection at lowest threshold
                    orig_path = os.path.join(output_dir, f"original_{os.path.basename(image_path)}")
                    cv2.imwrite(orig_path, image)

    print("\nProcessing complete!")

if __name__ == "__main__":
    # Adjust these paths to match your environment
    model_path = "C:\\Users\\vgoyal\\Desktop\\peer\\runs\\segment\\train8\\weights\\best.pt"
    image_dir = "C:\\Users\\vgoyal\\Desktop\\peer\\data\\test\\images\\test_hard"
    output_dir = "C:/Users/vgoyal/Desktop/peer/results"

    test_model_with_debug(model_path, image_dir, output_dir)
