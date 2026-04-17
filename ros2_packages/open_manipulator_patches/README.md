# open_manipulator 패치 파일

[ROBOTIS open_manipulator](https://github.com/ROBOTIS-GIT/open_manipulator) 패키지에 추가/수정한 파일들입니다.

`~/colcon_ws/src/open_manipulator/` 아래 동일한 경로에 복사하면 됩니다.

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
cd ~/colcon_ws/src/open_manipulator

# launch 파일
cp open_manipulator_patches/open_manipulator_x_description/launch/isaac_sim_tf.launch.py \
   open_manipulator_x_description/launch/

# URDF 파일
cp open_manipulator_patches/open_manipulator_x_description/urdf/open_manipulator_x_with_camera.urdf.xacro \
   open_manipulator_x_description/urdf/
cp open_manipulator_patches/open_manipulator_x_description/urdf/open_manipulator_x_with_camera.urdf \
   open_manipulator_x_description/urdf/
cp open_manipulator_patches/open_manipulator_x_description/urdf/stand_rs-d435_s01.stl \
   open_manipulator_x_description/urdf/

# kinematics 설정 (기존 파일 덮어쓰기)
cp open_manipulator_patches/open_manipulator_x_moveit_config/config/kinematics.yaml \
   open_manipulator_x_moveit_config/config/kinematics.yaml
```

## kinematics.yaml 변경 내용

`lma_kinematics_plugin` → `kdl_kinematics_plugin` 으로 변경.

lma 플러그인이 설치되지 않은 환경에서 IK 서비스 타임아웃이 발생하는 문제를 해결합니다.
