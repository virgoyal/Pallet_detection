import argparse
from ultralytics import YOLO

def train_segmentation_model(data_path, epochs, imgsz, batch, patience):
    # Load the pre-trained YOLOv8 segmentation model
    model = YOLO('yolov8n-seg.pt')
    
    # Train the model using the segmentation dataset
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        verbose=True
    )

    # Validate the model after training
    results = model.val()
    print(f"Segmentation model validation results:\n{results}")

def main():
    parser = argparse.ArgumentParser(description="Train a YOLOv8 segmentation model on a custom dataset.")
    
    parser.add_argument("data_path", type=str, help="Path to the segmentation dataset YAML file.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100).")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640).")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (default: 4).")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15).")
    
    args = parser.parse_args()
    
    train_segmentation_model(
        data_path=args.data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience
    )

if __name__ == "__main__":
    main()
