#!/usr/bin/env python3
"""
배달 미션 통합 런치파일
========================

엘리베이터 자율 배달 전체 시퀀스에 필요한 노드를 한 번에 실행합니다.

실행되는 노드
  - arm_delivery     : 픽업(14스텝) / 배달(15스텝) 팔 제어
  - arm_elevator     : 엘리베이터 UP/DOWN · 층수 버튼 인식/누르기
  - aruco_servoing   : 전체 시퀀스 오케스트레이터 (ArUco 정렬 + 주행 + 단계 조율)
  - contact_detector : (선택) 버튼 접촉 모니터링 — 기본 OFF, contact:=true 로 켬
                       ※ 차량 전진 시 덜컹임을 충돌로 오인해 팔이 움츠러드는
                         문제가 있어 기본 비활성화. 필요 시에만 켤 것.

선행 조건 (이 런치파일은 포함하지 않음 — 별도 터미널에서 먼저 실행)
  ros2 launch open_manipulator_x_bringup hardware.launch.py
  ros2 launch realsense2_camera rs_launch.py rgb_camera.color_profile:=1920,1080,30 align_depth.enable:=true
  ros2 run tf2_ros static_transform_publisher --x 0.12 --y 0.01 --z 0.062 \
      --roll 0.0 --pitch 0.0 --yaw 0.0 --frame-id link5 --child-frame-id camera_link
  (Scout Mini 베이스 드라이버도 주행을 위해 필요)

미션 트리거 (런치 후 운영자가 발행)
  ros2 topic pub --once /mission_floor std_msgs/Int32 "{data: -2}"
  ros2 topic pub --once /start_pickup  std_msgs/Bool  "{data: true}"

사용법
  ros2 launch delivery_mission.launch.py               # 핵심 3노드 (접촉 모니터 OFF)
  ros2 launch delivery_mission.launch.py contact:=true # 접촉 모니터까지 함께 실행
  (또는 패키지화 전이라면) python3 -m launch delivery_mission.launch.py
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


# 이 런치파일이 있는 디렉터리 (= nodes/real_robot) 를 기준으로 스크립트 경로 해석
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _node(filename):
    """nodes/real_robot/<filename> 을 python3 로 실행하는 ExecuteProcess 생성."""
    path = os.path.join(THIS_DIR, filename)
    return ExecuteProcess(
        cmd=['python3', path],
        name=os.path.splitext(filename)[0],
        output='screen',
        emulate_tty=True,
    )


def generate_launch_description():
    contact = LaunchConfiguration('contact')

    return LaunchDescription([
        DeclareLaunchArgument(
            'contact', default_value='false',
            description='버튼 접촉 모니터(contact_detector) 동시 실행 여부. '
                        '차량 전진 덜컹임을 충돌로 오인하는 문제로 기본 OFF.',
        ),

        # --- 핵심 미션 노드 3종 ---
        _node('arm_delivery.py'),     # 팔: 픽업 / 배달
        _node('arm_elevator.py'),     # 팔: 엘리베이터 버튼
        _node('aruco_servoing.py'),   # 주행 + 시퀀스 오케스트레이터

        # --- 선택: 접촉 모니터 ---
        ExecuteProcess(
            cmd=['python3', os.path.join(THIS_DIR, 'contact_detector.py')],
            name='contact_detector',
            output='screen',
            emulate_tty=True,
            condition=IfCondition(contact),
        ),
    ])
