# OpenMANIPULATOR-X URDF (Isaac Sim 임포트용)

Isaac Sim에 임포트해서 사용한 로봇 URDF와 메시 파일 모음. 임포트 절차는
[`../isaac_sim_openmanipulator_import.md`](../isaac_sim_openmanipulator_import.md) 참고.

## 파일
| 파일 | 설명 |
|------|------|
| `open_manipulator_x.urdf` | OpenMANIPULATOR-X 기본 (팔 + 그리퍼) |
| `open_manipulator_x_with_camera.urdf` | link5에 D435 카메라 장착 버전 |
| `meshes/*.stl` | 링크/그리퍼 메시 |
| `meshes/d435.dae` | D435 카메라 메시 (카메라 버전용) |

## 참고
- 메시 경로는 이 폴더만으로 바로 임포트되도록 `package://...` → 상대경로(`meshes/...`)로
  변경해 두었다. (원본은 ROS2 패키지 `open_manipulator_x_description` / `realsense2_description` 기준)
- 단위: meters (Isaac Sim meters per unit = 1.0과 일치)
- 원본 위치: `~/colcon_ws/src/open_manipulator/open_manipulator_x_description/urdf/`
