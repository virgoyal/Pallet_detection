#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import torch
import torchvision.ops as ops
from ultralytics import YOLO
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class PalletDetectionNode(Node):
    def __init__(self):
        super().__init__('pallet_detection_node')

        # Parameters
        self.declare_parameter('model_path', '../model/best.pt')
        self.declare_parameter('camera_topic', '/robot1/zed2i/left/image_rect_color')
        self.declare_parameter('min_confidence', 0.55)
        self.conf_threshold = self.get_parameter('min_confidence').get_parameter_value().double_value

        # Load parameters
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value

        # Resolve model path
        full_model_path = os.path.join(os.path.dirname(__file__), model_path)
        if not os.path.exists(full_model_path):
            self.get_logger().error(f'Model file not found: {full_model_path}')
            raise FileNotFoundError(f'Model file not found: {full_model_path}')

        self.get_logger().info(f'Loading YOLO model from {full_model_path}')
        self.model = YOLO(full_model_path)

        self.bridge = CvBridge()
        self.debug_view = True

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.image_subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            qos_profile
        )

        self.annotated_image_publisher = self.create_publisher(
            Image,
            '/pallet_detection/annotated_image',
            10
        )

        self.get_logger().info('Pallet Detection Node initialized')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            annotated_image = cv_image.copy()

            results = self.model(cv_image)
            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                self.get_logger().info("No boxes detected.")
                return

            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            masks = result.masks.data.cpu().numpy() if result.masks is not None else None

            # Filter by confidence threshold
            filtered_indices = [i for i, conf in enumerate(confidences) if conf >= self.conf_threshold]
            if not filtered_indices:
                self.get_logger().info("All detections below confidence threshold.")
                return

            boxes_filtered = boxes_xyxy[filtered_indices]
            scores_filtered = confidences[filtered_indices]
            classes_filtered = classes[filtered_indices]
            masks_filtered = masks[filtered_indices] if masks is not None else None

            # Apply NMS
            keep = ops.nms(torch.tensor(boxes_filtered), torch.tensor(scores_filtered), iou_threshold=0.5).numpy()

            self.get_logger().info(f"Visualizing {len(keep)} filtered detections")

            for i in keep:
                x1, y1, x2, y2 = boxes_filtered[i]
                conf = scores_filtered[i]
                cls_id = classes_filtered[i]
                label = f"{'Ground' if cls_id == 0 else 'Pallet'}: {conf:.2f}"
                color = [0, 255, 0] if cls_id == 0 else [0, 0, 255]
                alpha = 0.3 if cls_id == 0 else 0.5

                # Draw mask if available
                if masks_filtered is not None:
                    mask = masks_filtered[i]
                    mask_resized = cv2.resize(mask, (cv_image.shape[1], cv_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    overlay = np.zeros_like(cv_image, dtype=np.uint8)
                    overlay[mask_binary == 1] = color
                    mask_3ch = np.stack([mask_binary]*3, axis=-1)
                    annotated_image = np.where(mask_3ch,
                                               (alpha * overlay + (1 - alpha) * annotated_image).astype(np.uint8),
                                               annotated_image)

                # Draw bounding box and label
                cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(annotated_image, label, (int(x1), max(0, int(y1) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Publish the annotated image
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image.astype(np.uint8), encoding='bgr8')
            annotated_msg.header = msg.header
            self.annotated_image_publisher.publish(annotated_msg)

            if self.debug_view:
                cv2.imshow("Pallet Detection - Annotated", annotated_image)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def destroy_node(self):
        if self.debug_view:
            cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = PalletDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down due to KeyboardInterrupt")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()