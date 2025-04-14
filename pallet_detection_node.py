#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from ultralytics import YOLO
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class PalletDetectionNode(Node):
    def __init__(self):
        super().__init__('pallet_detection_node')
        
        # Declare parameters
        self.declare_parameter('model_path', '../model/best.pt')
        self.declare_parameter('confidence_threshold', 0.1)
        self.declare_parameter('camera_topic', '/robot1/zed2i/left/image_rect_color')
        
        # Get parameters
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        
        # Calculate absolute path for model
        full_model_path = os.path.join(os.path.dirname(__file__), model_path)
        
        self.get_logger().info(f'Loading model from: {full_model_path}')
        
        # Check if model file exists
        if not os.path.exists(full_model_path):
            self.get_logger().error(f'Model file not found: {full_model_path}')
            raise FileNotFoundError(f'Model file not found: {full_model_path}')
            
        # Load YOLO model
        self.model = YOLO(full_model_path)
        
        self.get_logger().info(f'Model loaded successfully. Task: {self.model.task}')
        
        # Initialize CV bridge to convert between ROS and OpenCV images
        self.bridge = CvBridge()
        
        # Create QoS profile that's compatible with bag files
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Create subscription to camera images with the compatible QoS profile
        self.image_subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            qos_profile)
            
        # Create publisher for annotated images
        self.annotated_image_publisher = self.create_publisher(
            Image,
            '/pallet_detection/annotated_image',
            10)
            
        self.get_logger().info('Pallet Detection Node initialized')
        
    def image_callback(self, msg):
        # Log that we received an image
        self.get_logger().debug(f'Received image with timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}')
        
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Run inference with YOLO
            results = self.model(cv_image, conf=self.confidence_threshold)
            
            # Get the first result
            result = results[0]
            
            # Create a copy of the image to draw on
            annotated_image = cv_image.copy()
            
            # Process segmentation masks and detected objects
            if hasattr(result, 'masks') and result.masks is not None:
                # Get segmentation masks
                for seg, box, cls in zip(result.masks.data, result.boxes.data, result.boxes.cls):
                    # Get class ID (0: ground, 1: pallet)
                    class_id = int(cls.item())
                    
                    # Create mask
                    seg = seg.cpu().numpy()
                    
                    # Convert mask to binary image
                    mask = (seg > 0.5).astype(np.uint8)
                    mask = np.transpose(mask, (1, 2, 0))  # (H, W, 1)
                    mask = cv2.resize(mask, (cv_image.shape[1], cv_image.shape[0]))
                    
                    # Create color overlay based on class
                    if class_id == 0:  # ground
                        color = [0, 255, 0]  # Green for ground
                        alpha = 0.3
                    else:  # pallet
                        color = [0, 0, 255]  # Red for pallet
                        alpha = 0.5
                    
                    # Apply mask as overlay
                    mask_colored = np.zeros_like(cv_image)
                    mask_colored[mask > 0] = color
                    
                    # Blend the mask with the original image
                    cv2.addWeighted(mask_colored, alpha, annotated_image, 1 - alpha, 0, annotated_image)
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"Pallet: {box[4]:.2f}" if class_id == 1 else f"Ground: {box[4]:.2f}"
                    cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Convert the annotated image back to ROS message
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            annotated_msg.header = msg.header  # Copy the header
            
            # Publish the annotated image
            self.annotated_image_publisher.publish(annotated_msg)
            
            self.get_logger().debug('Published annotated image')
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = PalletDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
