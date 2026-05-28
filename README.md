# 자율주행 택배 로봇 — 엘리베이터 버튼 인식 및 조작 시스템


캡스톤디자인 프로젝트 — 자율주행 택배 로봇이 엘리베이터를 스스로 탑승하고 목적지까지 배달할 수 있도록,  
로봇팔이 카메라로 버튼·호수를 인식하고 픽업·배달·엘리베이터 조작을 자동으로 수행하는 시스템입니다.

## 시스템 구성

```
택배기사 앱 (층수 입력)
        ↓  ROS2 /target_floor
자율주행 로봇 Scout Mini (출발점 대기)

[픽업]
로봇팔 OpenMANIPULATOR-X  ← 이 저장소
  └─ 책상 위 박스 집기 → Scout Mini 바구니에 내려놓기
        ↓  ROS2 /robot_status  (PICKUP_DONE)

[엘리베이터 탑승]
자율주행 로봇 Scout Mini (엘리베이터 앞으로 이동)
        ↓  ROS2 /elevator_ready
로봇팔 OpenMANIPULATOR-X
  ├─ UP/DOWN 버튼 인식 → 누르기
  ├─ 버튼 점등 확인 → 엘리베이터 도착 대기
  └─ 목표 층수 버튼 누르기
        ↓  ROS2 /robot_status  (NUMBER_PRESSED)
자율주행 로봇 Scout Mini (엘리베이터 탑승 → 목표 층 이동)

[배달]
로봇팔 OpenMANIPULATOR-X
  └─ 호수 인식 (EasyOCR) → /room_number 발행
        ↓  ROS2 /room_number
자율주행 로봇 Scout Mini (해당 호실 앞 정렬)
        ↓  ROS2 /aligned_ready
로봇팔 OpenMANIPULATOR-X
  └─ 바구니에서 박스 집기 → 목적지에 내려놓기
        ↓  ROS2 /robot_status  (DELIVERY_DONE)

[복귀]
자율주행 로봇 Scout Mini (엘리베이터 앞으로 이동)
        ↓  ROS2 /elevator_ready
로봇팔 OpenMANIPULATOR-X
  └─ DOWN 버튼 → 1층 버튼 누르기
자율주행 로봇 Scout Mini (1층 복귀)
```

---

## 기술 스택

| 분야 | 기술 |
|------|------|
| 로봇 플랫폼 | OpenMANIPULATOR-X (4-DOF) |
| 자율주행 | Scout Mini |
| 카메라 | Intel RealSense D435 (RGB-D) |
| 버튼 인식 | YOLOv8 (mAP50: 98.7%), YOLO-seg + EasyOCR |
| 호수 인식 | EasyOCR |
| IK | 해석적 IK (수식 직접 계산, MoveIt2 불필요) |
| 접촉 감지 | SVM (RBF 커널, F1: 0.812) |
| 로봇 미들웨어 | ROS2 Humble |
| 시뮬레이션 | Isaac Sim 5.1.0 |
| 언어 | Python 3.10 |

---

## 전체 동작 흐름

```
[픽업]
책상 위 박스 감지 (YOLOv8n + Depth)
        ↓
해석적 IK → 박스 집기 → Scout Mini 바구니에 내려놓기
        ↓  /robot_status: PICKUP_DONE

[엘리베이터 탑승]
Scout Mini 엘리베이터 앞 도착 → /elevator_ready 수신
        ↓
YOLOv8 → UP/DOWN 버튼 감지
        ↓
Depth → 3D 좌표 추출 → 해석적 IK → 버튼 누르기
        ↓
홈 복귀 → 버튼 점등 확인 (HSV 초록 비율)
        ↓
버튼 소등 대기 → 엘리베이터 도착
        ↓
YOLO-seg + EasyOCR → 목표 층수 버튼 감지 → 누르기
        ↓  /robot_status: NUMBER_PRESSED

[배달]
Scout Mini 목표 층 도착
        ↓
팔 위로 이동 → EasyOCR → 호수 인식 (531호 등)
        ↓  /room_number 발행
Scout Mini 해당 호실 앞 정렬 → /aligned_ready 수신
        ↓
바구니에서 박스 집기 → 목적지 책상에 내려놓기
        ↓  /robot_status: DELIVERY_DONE

[복귀]
Scout Mini 엘리베이터 앞 도착 → /elevator_ready 수신
        ↓
DOWN 버튼 → 1층 버튼 누르기 → Scout Mini 1층 복귀
```

엘리베이터 버튼 상태 머신 (`real_robot_unified.py`):

```
IDLE → UPDOWN_READY → UPDOWN_PRESS → WAIT → NUMBER_READY → NUMBER_PRESS → DONE
```

---

## 픽앤플레이스 데모 (Pick & Place)

책상 위 박스를 Scout Mini 바구니에 픽업한 뒤, 목적지 책상까지 배달하는 데모입니다.

### 데모 영상

<p align="center">
  <img src="media/demo_pickup.gif" width="48%"/>
  <img src="media/demo_delivery.gif" width="48%"/>
</p>
<p align="center">
  <em>좌: 픽업 (책상 → 바구니) &nbsp;|&nbsp; 우: 배달 (바구니 → 목적지)</em>
</p>

### 시퀀스

각 스텝은 **Joint 직접 지령**(절대 관절각)과 **XYZ→IK**(좌표 입력, IK가 관절각 자동 계산) 두 방식을 혼용합니다.  
위치가 바뀔 때: Joint 스텝은 재실측 필요, XYZ 스텝은 좌표값만 수정.

| # | 픽업 | 방식 | 배달 | 방식 |
|---|------|------|------|------|
| 1 | 홈 | Joint `HOME_JOINTS` | 홈 | Joint `HOME_JOINTS` |
| 2 | 책상 방향 확인 | Joint `TABLE_LOOK_JOINTS` | 바구니 확인 | Joint `BASKET_LOOK_JOINTS` |
| 3 | 그리퍼 열기 | Gripper | 그리퍼 열기 | Gripper |
| 4 | 박스 위 호버 | XYZ `TABLE_HOVER` | 바구니 박스 잡기 | Joint `BASKET_GRIP_JOINTS` |
| 5 | 박스 잡기 위치 | XYZ `TABLE_GRIP` | 그리퍼 닫기 | Gripper |
| 6 | 그리퍼 닫기 | Gripper | 박스 들어올리기 | XYZ `BASKET_HOVER` |
| 7 | 바구니에 내려놓기 | Joint `BASKET_PLACE_JOINTS` | 목적지 방향 확인 | Joint `TABLE_LOOK_JOINTS` |
| 8 | 그리퍼 열기 | Gripper | 목적지 호버 | XYZ `DEST_HOVER` |
| 9 | 홈 복귀 | Joint `HOME_JOINTS` | 목적지에 내려놓기 | XYZ `DEST_PLACE` |
| 10 | | | 그리퍼 열기 | Gripper |
| 11 | | | 위로 호버 | XYZ `DEST_HOVER` |
| 12 | | | 홈 복귀 | Joint `HOME_JOINTS` |

### 웨이포인트 값

**Joint 직접 지령** — 절대 관절각 [rad], 위치 바뀌면 재실측 필요

| 상수 | joint1 | joint2 | joint3 | joint4 | 용도 |
|------|--------|--------|--------|--------|------|
| `HOME_JOINTS` | 3.141 | -1.3963 | 1.2217 | 0.5236 | 홈 포지션 |
| `TABLE_LOOK_JOINTS` | 1.571 | -1.3963 | 1.2217 | 0.5236 | 책상·목적지 방향 확인 |
| `BASKET_LOOK_JOINTS` | -3.116 | -0.387 | 0.755 | 1.164 | 바구니 확인 (배달) |
| `BASKET_PLACE_JOINTS` | 3.1032 | 0.00767 | 1.41126 | -1.41433 | 바구니에 내려놓기 (픽업) |
| `BASKET_GRIP_JOINTS` | 3.122 | 0.457 | 0.831 | 0.305 | 바구니 박스 잡기 (배달) |

**XYZ → IK** — world 프레임 좌표 [m], 좌표값만 수정하면 IK가 관절각 자동 계산

| 상수 | X | Y | Z | 용도 |
|------|---|---|---|------|
| `TABLE_HOVER` | 0.013 | 0.298 | 0.100 | 박스 위 호버 (픽업) |
| `TABLE_GRIP` | 0.013 | 0.298 | 0.040 | 박스 잡기 위치 (픽업) |
| `BASKET_HOVER` | -0.165 | 0.009 | 0.123 | 박스 들어올리기 (배달) |
| `DEST_HOVER` | 0.013 | 0.298 | 0.100 | 목적지 호버·위로 후퇴 (배달) |
| `DEST_PLACE` | 0.013 | 0.298 | 0.040 | 목적지에 내려놓기 (배달) |

### 실행

```bash
# 터미널 1 — 하드웨어 컨트롤러
ros2 launch open_manipulator_x_bringup hardware.launch.py

# 터미널 2 — 자동 실행 (스텝 간 1.5초 자동 진행)
python3 nodes/real_robot/real_robot_delivery.py

# 또는 수동 실행 (Enter로 한 스텝씩 진행)
python3 nodes/real_robot/test_delivery_motion.py
```

| 키 | 동작 |
|----|------|
| `Enter` | 현재 스텝 실행 (수동 모드) |
| `q` | 즉시 종료 (홈 복귀) |
| `r` | 처음부터 다시 |

---

## 해석적 IK (Analytical Inverse Kinematics)

MoveIt2 없이 수식을 직접 유도하여 관절각을 계산합니다.

<p align="center">
  <img src="https://emanual.robotis.com/assets/images/platform/openmanipulator_x/rviz_om_view.png" width="40%"/>
</p>

### 링크 파라미터

| 기호 | 값 | 설명 |
|------|-----|------|
| $L_1$ | 0.0595 m | base → joint2 (수직) |
| $L_2$ | $\sqrt{0.024^2 + 0.128^2} \approx 0.1302$ m | joint2 → joint3 유효 길이 |
| $\alpha$ | $\text{atan2}(0.128,\ 0.024) \approx 79.4°$ | link2 수평 기준각 |
| $L_3$ | 0.124 m | joint3 → joint4 |
| $L_4$ | 0.126 m | joint4 → end-effector |

### 수식 유도

목표 위치 $(X, Y, Z)$ 가 주어졌을 때:

**① θ₁ — 베이스 회전**

$$\theta_1 = \text{atan2}(Y,\ X)$$

**② 손목 위치 계산** (end-effector에서 $L_4$ 제거)

$$r_w = \sqrt{X^2 + Y^2} - L_4, \qquad z_w = Z$$

$$d_r = r_w, \quad d_z = z_w - L_1, \quad D = \sqrt{d_r^2 + d_z^2}$$

**③ 코사인 법칙** → ψ (L₂, L₃ 사이각, 두 해: elbow-up / elbow-down)

$$\cos\psi = \frac{D^2 - L_2^2 - L_3^2}{2 L_2 L_3}, \qquad \psi = \pm\arccos(\cos\psi)$$

**④ θ₂, θ₃**

$$\gamma = \text{atan2}(L_3 \sin\psi,\ L_2 + L_3 \cos\psi)$$

$$\theta_2 = \alpha - \bigl(\text{atan2}(d_z,\ d_r) - \gamma\bigr)$$

$$\theta_3 = -\psi - \alpha$$

**⑤ θ₄ — 수평 접근 구속 조건** (end-effector가 버튼에 수평으로 접근)

$$\theta_4 = -(\theta_2 + \theta_3)$$

elbow-up / elbow-down 두 해를 모두 계산한 뒤, 관절 한계를 통과하는 첫 번째 해를 사용합니다.

---

## YOLO 학습 결과 (UP/DOWN 버튼)

<p align="center">
  <img src="yolo/results/results.png" width="32%"/>
  <img src="yolo/results/confusion_matrix_normalized.png" width="32%"/>
  <img src="yolo/results/val_batch0_pred.jpg" width="32%"/>
</p>

mAP50: **98.7%**

---

## SW 블록 다이어그램

<p align="center">
  <img src="media/sw_block_diagram.svg" width="90%"/>
</p>

---

## 설치

### 요구 사항

- Ubuntu 22.04 + ROS2 Humble
- Python 3.10
- OpenMANIPULATOR-X + U2D2 (실제 로봇)
- Intel RealSense D435 (실제 로봇)

### 1. 이 레포 클론

```bash
git clone https://github.com/uihyeong/elevator-button-robot.git
cd elevator-button-robot
pip install -r requirements.txt
```

> **주의**: `numpy < 2.0.0` 필요. cv_bridge가 NumPy 1.x 기준으로 컴파일되어 있습니다.

### 2. colcon_ws — ROS2 하드웨어 패키지 빌드

```bash
# 의존 패키지 설치
sudo apt install \
  ros-humble-ros2-control \
  ros-humble-moveit* \
  ros-humble-gazebo-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-controller-manager \
  ros-humble-position-controllers \
  ros-humble-joint-state-broadcaster \
  ros-humble-joint-trajectory-controller \
  ros-humble-gripper-controllers \
  ros-humble-hardware-interface \
  ros-humble-xacro

# open_manipulator 소스 빌드
mkdir -p ~/colcon_ws/src && cd ~/colcon_ws/src
git clone -b humble https://github.com/ROBOTIS-GIT/DynamixelSDK.git
git clone -b humble https://github.com/ROBOTIS-GIT/open_manipulator.git
git clone -b humble https://github.com/ROBOTIS-GIT/dynamixel_hardware_interface.git
git clone -b humble https://github.com/ROBOTIS-GIT/dynamixel_interfaces.git
cd ~/colcon_ws && colcon build --symlink-install

# ROS 환경 설정
echo 'source /usr/share/gazebo/setup.sh' >> ~/.bashrc
echo 'source ~/colcon_ws/install/local_setup.bash' >> ~/.bashrc
source ~/.bashrc

# RealSense 드라이버
sudo apt install ros-humble-realsense2-camera
```

> 전체 설치 가이드 (U2D2 통신 설정 등): [ROBOTIS e-Manual — OpenMANIPULATOR-X Quick Start Guide](https://emanual.robotis.com/docs/en/platform/openmanipulator_x/quick_start_guide/)

### 3. open_manipulator_patches 적용 (선택)

MoveIt2 노드 또는 Isaac Sim 시뮬레이션 사용 시에만 필요합니다.  
→ [적용 방법 보기](ros2_packages/open_manipulator_patches/README.md)

---

## 실행 방법 (실제 로봇)

U2D2 연결 후 아래 공통 준비를 먼저 실행합니다.

```bash
# 터미널 1 — 하드웨어 컨트롤러
ros2 launch open_manipulator_x_bringup hardware.launch.py

# 터미널 2 — D435 카메라
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true

# 터미널 3 — 카메라 TF 연결 (유지)
ros2 run tf2_ros static_transform_publisher \
    --x 0.12 --y 0.01 --z 0.062 \
    --roll 0.0 --pitch 0.0 --yaw 0.0 \
    --frame-id link5 --child-frame-id camera_link
```

---

### ★ 픽업/배달 노드 (`real_robot_delivery.py`)

책상 위 박스를 바구니에 픽업하고, 목적지 책상까지 배달합니다.

```bash
# 터미널 4 — 픽업/배달 자동 실행 노드
python3 nodes/real_robot/real_robot_delivery.py
```

메뉴에서 `1` (픽업) 또는 `2` (배달) 선택 후 스텝 간 1.5초 딜레이로 자동 진행됩니다.

---

### ★ 엘리베이터 버튼 노드 (`real_robot_unified.py`)

YOLOv8 + YOLO-seg + EasyOCR로 버튼을 인식하고 누릅니다. YOLO 모델(`yolo/weights/`)이 레포에 포함되어 있어 추가 학습 없이 바로 실행 가능합니다.

```bash
# 터미널 4 — 엘리베이터 버튼 노드
python3 nodes/real_robot/real_robot_unified.py

# 터미널 5 — 층수 입력 (3층 예시)
ros2 topic pub --once /target_floor std_msgs/Int32 "{data: 3}"
```

<p align="center">
  <img src="media/demo1.gif" width="48%"/>
  <img src="media/demo2.gif" width="48%"/>
</p>

---

### 접촉 감지 — 선택, 병렬 실행

두 가지 방식을 제공합니다.

#### 방법 A — SVM 기반 (`contact_detector_svm.py`) ★ 권장

FSR406 센서로 수집한 실측 데이터(총 19,664 슬라이딩 윈도우)로 학습한  
SVM(RBF 커널) 모델이 접촉 여부를 분류합니다.

**특징 벡터 (160차원)**:  
최근 10샘플 × (joint velocity 4 + effort_delta 4 + 프레임 간 diff 8)

**학습 결과**: 5-fold CV F1 **0.812 ± 0.006** (sklearn 1.7.2, StandardScaler 정규화)

**오인식 방지**:
- 버튼 누르는 중 (`MOVING` 상태): 감지 완전 차단
- 홈 복귀 중 (`MOVING` 종료 후 6초): 감지 억제
- 연속 3윈도우 × prob ≥ 0.80 조건 충족 시에만 접촉 확정

```bash
python3 nodes/real_robot/contact_detector_svm.py
```

모델 재학습 (로그 파일 추가 후):

```bash
python3 nodes/real_robot/train_svm_model.py
```

#### 방법 B — Effort Threshold 기반 (`contact_detector.py`)

joint3 effort 편차가 threshold를 초과하면 접촉으로 판정하는 단순 방식입니다.

```bash
python3 nodes/real_robot/contact_detector.py
```

---

## 실행 방법 (Isaac Sim 시뮬레이션)

Isaac Sim 실행 후 Play ▶️ 를 누른 뒤:

```bash
# 터미널 1 — Static TF 발행
ros2 launch open_manipulator_x_description isaac_sim_tf.launch.py

# 터미널 2 — 브릿지 노드 (Isaac Sim ↔ ROS2)
ros2 run isaac_moveit_bridge bridge

# 터미널 3 — 메인 노드 (MoveIt2 없이)
python3 nodes/simulation/isaac_sim_direct_ik.py
```

MoveIt2 사용 시:

```bash
ros2 launch open_manipulator_x_moveit_config moveit_core.launch.py
python3 nodes/simulation/pid_joint_controller.py
python3 nodes/simulation/isaac_sim_yolo_moveit.py
```

<p align="center">
  <img src="media/sim_full.gif" width="48%"/>
  <img src="media/sim_robot.gif" width="48%"/>
</p>

---

## 파일 구조

```
elevator-button-robot/
├── nodes/
│   ├── real_robot/
│   │   ├── real_robot_unified.py       # ★ 엘리베이터 버튼 노드 (UP/DOWN → 숫자 전체 시퀀스)
│   │   ├── real_robot_delivery.py      # ★ 픽업/배달 자동 실행 노드 (스텝 간 자동 진행)
│   │   ├── test_delivery_motion.py     # 픽업/배달 수동 데모 (Enter로 스텝별 진행)
│   │   ├── contact_detector.py         # 접촉 감지 — Effort Threshold 방식
│   │   ├── contact_detector_svm.py     # ★ 접촉 감지 — SVM 방식 (권장)
│   │   ├── train_svm_model.py          # SVM 모델 재학습 스크립트
│   │   ├── fsr_effort_logger.py        # FSR + effort 동시 로깅 (학습 데이터 수집)
│   │   ├── test_button_lit.py          # 버튼 점등 HSV 튜닝 도구
│   │   ├── real_robot_direct_ik.py     # UP/DOWN 단독 노드
│   │   ├── real_robot_num_ocr_ik.py    # 숫자 버튼 단독 노드
│   │   └── real_robot_yolo_moveit.py   # MoveIt2 IK 노드 (참고용)
│   └── simulation/
│       ├── isaac_sim_yolo_moveit.py    # YOLO + MoveIt2
│       ├── isaac_sim_direct_ik.py      # YOLO + 해석적 IK
│       ├── pid_joint_controller.py     # PID 관절 제어기 (50Hz)
│       └── setup_mobile_manipulator.py # Scout Mini + 로봇팔 합체 씬 구성
├── ros2_packages/
│   ├── isaac_moveit_bridge/            # Isaac Sim ↔ ROS2 브릿지 패키지
│   └── open_manipulator_patches/       # open_manipulator 커스텀 수정 파일
├── yolo/
│   ├── weights/
│   │   ├── best.pt                     # UP/DOWN 버튼 인식 모델 (mAP50: 98.7%)
│   │   └── best_num.pt                 # 숫자 버튼 분할 모델
│   ├── dataset/                        # 학습 데이터셋 정보
│   └── results/                        # 학습 결과 그래프
├── media/                              # 데모 GIF
└── requirements.txt
```

---

## 토픽 인터페이스

### 로봇팔 ↔ Scout Mini

| 토픽 | 방향 | 타입 | 설명 |
|------|------|------|------|
| `/target_floor` | Scout Mini → 팔 | `std_msgs/Int32` | 목표 층수 (음수=지하, 예: -1=B1) |
| `/elevator_ready` | Scout Mini → 팔 | `std_msgs/Bool` | 엘리베이터 앞 도착 신호 |
| `/aligned_ready` | Scout Mini → 팔 | `std_msgs/Bool` | 호실 앞 정렬 완료 신호 |
| `/robot_status` | 팔 → Scout Mini | `std_msgs/String` | 아래 상태값 참고 |
| `/room_number` | 팔 → Scout Mini | `std_msgs/String` | 인식된 호수 (예: "531") |

### 내부 토픽

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/target_point` | `geometry_msgs/PointStamped` | 수동 테스트용 world 좌표 직접 입력 |
| `/contact_detected` | `std_msgs/Bool` | 접촉 감지 신호 |
| `/contact_status` | `std_msgs/String` | CONTACT_DETECTED / CONTACT_RESOLVED |

### `/robot_status` 상태값

| 값 | 의미 |
|---|---|
| `MOVING` | 관절 이동 중 |
| `PICKUP_DONE` | 픽업 완료 |
| `BUTTON_PRESSED` | UP/DOWN 버튼 점등 확인 완료 |
| `ELEVATOR_ARRIVED` | 버튼 소등 감지 (엘리베이터 도착) |
| `NUMBER_PRESSED` | 층수 버튼 누르기 완료 |
| `DELIVERY_DONE` | 배달 완료 |
| `NEED_REPOSITION` | 연속 3회 실패 → Scout Mini 재정렬 요청 |
| `FAILED` | 오류 |

층수 입력 예시:

```bash
ros2 topic pub --once /target_floor std_msgs/Int32 "{data: 3}"   # 3층 (up)
ros2 topic pub --once /target_floor std_msgs/Int32 "{data: -1}"  # B1층 (down)
```

---

## 개발 환경

- OS: Ubuntu 22.04
- ROS2: Humble
- Python: 3.10
- Isaac Sim: 5.1.0
