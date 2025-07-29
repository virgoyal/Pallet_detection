from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value='/home/akxhay/Desktop/Pallet_detection-main/best.pt',
            description='Path to the YOLOv8 .pt model file'
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/robot1/zed2i/left/image_rect_color',
            description='Image topic to subscribe to'
        ),
        Node(
            package='pallet_detector',
            executable='pallet_node',
            name='pallet_detector',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'image_topic': LaunchConfiguration('image_topic'),
            }]
        )
    ])
