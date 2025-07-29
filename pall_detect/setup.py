from setuptools import setup

package_name = 'pall_detect'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/pallet_launch.py']),
        ('share/' + package_name + '/models', ['models/best_model.pt']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='akxhay',
    maintainer_email='akxhaykannan@gmail.com',
    description='YOLOv8 pallet detection ROS2 node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pallet_node = pall_detect.pallet_detection_node:main'
        ],
    },
)

