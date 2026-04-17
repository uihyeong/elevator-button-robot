from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # world → open_manipulator_x (MoveIt2 플래닝 프레임 연결)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='world_to_robot_pub',
            arguments=['0', '0', '0', '0', '0', '0',
                       'world', 'open_manipulator_x'],
        ),
        # camera_link → camera_color_frame
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_color_frame_pub',
            arguments=['0', '0.015', '0', '0', '0', '0',
                       'camera_link', 'camera_color_frame'],
        ),
        # camera_color_frame → camera_color_optical_frame
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_color_optical_frame_pub',
            arguments=['0', '0', '0', '-1.5707963', '0', '-1.5707963',
                       'camera_color_frame', 'camera_color_optical_frame'],
        ),
    ])
