# open_manipulator 패치 파일

[ROBOTIS open_manipulator](https://github.com/ROBOTIS-GIT/open_manipulator) 패키지에 추가/수정한 파일들입니다.

## 용도별 필요 패치

| 사용 목적 | 필요한 패치 |
|---|---|
| `real_robot_unified.py` (권장) | 패치 불필요 |
| MoveIt2 노드 (`real_robot_yolo_moveit.py`) | `kinematics.yaml` 필수 |
| Isaac Sim 시뮬레이션 | `isaac_sim_tf.launch.py` 필수 |

## 파일 목록

| 파일 | 설명 |
|------|------|
| `open_manipulator_x_description/launch/isaac_sim_tf.launch.py` | Isaac Sim용 Static TF 발행 launch 파일 |
| `open_manipulator_x_description/urdf/open_manipulator_x_with_camera.urdf.xacro` | D435 카메라 마운트 포함 URDF xacro |
| `open_manipulator_x_description/urdf/open_manipulator_x_with_camera.urdf` | D435 카메라 마운트 포함 URDF (변환 완료) |
| `open_manipulator_x_description/urdf/stand_rs-d435_s01.stl` | D435 카메라 마운트 STL 메시 |
| `open_manipulator_x_moveit_config/config/kinematics.yaml` | KDL kinematics plugin으로 변경한 설정 |

## 적용 방법

```bash
cd ~/elevator-button-robot/ros2_packages/open_manipulator_patches

# MoveIt2 사용 시
cp open_manipulator_x_moveit_config/config/kinematics.yaml \
   ~/colcon_ws/src/open_manipulator/open_manipulator_x_moveit_config/config/kinematics.yaml

# Isaac Sim 시뮬레이션 사용 시
cp open_manipulator_x_description/launch/isaac_sim_tf.launch.py \
   ~/colcon_ws/src/open_manipulator/open_manipulator_x_description/launch/
cp open_manipulator_x_description/urdf/open_manipulator_x_with_camera.urdf.xacro \
   ~/colcon_ws/src/open_manipulator/open_manipulator_x_description/urdf/
cp open_manipulator_x_description/urdf/open_manipulator_x_with_camera.urdf \
   ~/colcon_ws/src/open_manipulator/open_manipulator_x_description/urdf/
cp open_manipulator_x_description/urdf/stand_rs-d435_s01.stl \
   ~/colcon_ws/src/open_manipulator/open_manipulator_x_description/urdf/

# 패치 적용 후 재빌드
cd ~/colcon_ws && colcon build --symlink-install
```

## kinematics.yaml 변경 내용

`lma_kinematics_plugin` → `kdl_kinematics_plugin` 으로 변경.

lma 플러그인이 설치되지 않은 환경에서 IK 서비스 타임아웃이 발생하는 문제를 해결합니다.
