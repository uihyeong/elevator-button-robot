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
- **PID 제어기** 로 관절 위치 피드백 제어
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
PID 제어기로 관절 위치 추종
    ↓
로봇팔 이동 및 버튼 누르기
```

## 파일 구조

```
elevator-button-robot/
├── nodes/
│   ├── isaac_sim_yolo_moveit.py           # 메인 노드 (YOLO + MoveIt2 IK)
│   ├── pid_joint_controller.py            # PID 관절 제어기 (50Hz)
│   ├── isaac_sim_analytical_ik_moveit.py  # 해석적 IK 실험 노드
│   ├── isaac_sim_yolo_depth.py            # 뎁스 인식 테스트
│   ├── isaac_sim_yolo_tf.py               # TF 변환 테스트
│   └── isaac_sim_yolo_test.py             # YOLO 인식 테스트
├── ros2_packages/
│   └── isaac_moveit_bridge/               # Isaac Sim ↔ MoveIt2 브릿지 패키지
└── yolo/
    ├── weights/best.pt                    # 학습된 YOLO 모델
    └── dataset/                           # 학습 데이터셋
```

## 실행 방법 (시뮬레이션)

```bash
# 1. Isaac Sim 실행 후 Play ▶️

# 2. Static TF 발행
ros2 launch open_manipulator_x_description isaac_sim_tf.launch.py

# 3. 브릿지 노드 실행
ros2 run isaac_moveit_bridge bridge

# 4. MoveIt2 실행
ros2 launch open_manipulator_x_moveit_config moveit_core.launch.py

# 5. PID 제어기 실행
python3 nodes/pid_joint_controller.py

# 6. 메인 노드 실행
python3 nodes/isaac_sim_yolo_moveit.py
```

## PID 제어기

`nodes/pid_joint_controller.py`

MoveIt2가 계획한 궤적을 실제 관절 위치 피드백으로 추종합니다.

```
bridge → /joint_target → PID 제어기 → /joint_command → Isaac Sim
                              ↑
                        /joint_states (피드백)
```

| 파라미터 | 값 | 설명 |
|---|---|---|
| 주기 | 50Hz | 제어 루프 주기 |
| Dead band | 0.02 rad | 이 안에 들어오면 보정 중지 |
| Deriv filter | alpha=0.1 | derivative 저역통과 필터 |
| MAX_VELOCITY | 1.0 rad/s | 출력 클램프 |

관절별 게인 (Kp, Ki, Kd):

| 관절 | Kp | Ki | Kd |
|---|---|---|---|
| joint1 (좌우) | 100 | 2.0 | 300 |
| joint2 (어깨) | 100 | 2.0 | 300 |
| joint3 (팔꿈치) | 80 | 1.5 | 200 |
| joint4 (손목) | 60 | 1.0 | 100 |

## 토픽 인터페이스

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/target_floor` | `std_msgs/Int32` | 목표 층수 입력 |
| `/robot_status` | `std_msgs/String` | 현재 상태 (MOVING / BUTTON_PRESSED / SUCCESS / FAILED) |
| `/joint_target` | `sensor_msgs/JointState` | PID 목표 관절 위치 |
| `/joint_command` | `sensor_msgs/JointState` | Isaac Sim 관절 명령 |

## 개발 환경

- OS: Ubuntu 22.04
- ROS2: Humble
- Python: 3.10
- Isaac Sim: 5.1.0
