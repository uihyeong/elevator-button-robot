# 자율주행 택배 로봇 - 엘리베이터 버튼 자동 인식 및 조작 시스템

캡스톤디자인 프로젝트 — 자율주행 택배 로봇이 엘리베이터를 스스로 탑승할 수 있도록,
로봇팔이 카메라로 버튼을 인식하고 자동으로 누르는 시스템입니다.

## 시스템 구성

```
택배기사 앱 (층수 입력)
        ↓
자율주행 로봇 (엘리베이터 앞으로 이동)
        ↓
로봇팔 (버튼 인식 → 누르기)  ← 이 저장소
        ↓
엘리베이터 탑승 → 배달 → 복귀
```

## 담당 역할 (로봇팔 시스템)

- **YOLOv8** 으로 UP/DOWN 버튼 실시간 인식
- **D435 RGB-D 카메라** 로 버튼 3D 좌표 추출
- **MoveIt2 IK** 로 로봇팔 경로 계획 및 실행
- **Isaac Sim** 시뮬레이션으로 전체 흐름 검증

## 기술 스택

| 분야 | 기술 |
|------|------|
| 로봇 플랫폼 | OpenMANIPULATOR-X |
| 카메라 | Intel RealSense D435 |
| AI/인식 | YOLOv8 (mAP50: 98.7%) |
| 로봇 미들웨어 | ROS2 Humble, MoveIt2 |
| 시뮬레이션 | Isaac Sim 5.1.0 |
| 언어 | Python 3.10 |

## 주요 흐름

```
카메라 영상 수신
    ↓
YOLOv8 버튼 감지 (UP / DOWN)
    ↓
Depth 이미지로 버튼 3D 좌표 계산
    ↓
TF 변환 (카메라 프레임 → 로봇 베이스 프레임)
    ↓
MoveIt2 IK로 관절 각도 계산
    ↓
로봇팔 이동 및 버튼 누르기
```

## 파일 구조

```
elevator-button-robot/
├── nodes/
│   ├── isaac_sim_yolo_moveit.py       # 메인 노드 (YOLO + MoveIt2 IK)
│   ├── isaac_sim_analytical_ik_moveit.py  # 해석적 IK 실험 노드
│   ├── isaac_sim_yolo_depth.py        # 뎁스 인식 테스트
│   ├── isaac_sim_yolo_tf.py           # TF 변환 테스트
│   └── isaac_sim_yolo_test.py         # YOLO 인식 테스트
└── ros2_packages/
    └── isaac_moveit_bridge/           # Isaac Sim ↔ MoveIt2 브릿지 패키지
```

## 실행 방법 (시뮬레이션)

```bash
# 1. Isaac Sim 실행 후 Play ▶️
# 2. 브릿지 노드 실행
ros2 run isaac_moveit_bridge bridge

# 3. MoveIt2 실행
ros2 launch open_manipulator_x_moveit_config moveit_core.launch.py

# 4. 메인 노드 실행
python3 nodes/isaac_sim_yolo_moveit.py
```

## 토픽 인터페이스

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/target_floor` | `std_msgs/Int32` | 목표 층수 입력 |
| `/robot_status` | `std_msgs/String` | 현재 상태 (MOVING / BUTTON_PRESSED / SUCCESS / FAILED) |

## 개발 환경

- OS: Ubuntu 22.04
- ROS2: Humble
- Python: 3.10
- Isaac Sim: 5.1.0
