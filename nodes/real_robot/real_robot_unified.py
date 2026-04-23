"""
통합 엘리베이터 버튼 인식 노드.

Phase 1 (UPDOWN):  YOLOv8 → UP/DOWN 버튼 감지 → 해석적 IK → 누르기
Phase 2 (NUMBER):  YOLO-seg + EasyOCR → 숫자 버튼 감지 → 해석적 IK → 누르기

상태 전이:
  IDLE → UPDOWN_READY → UPDOWN_PRESS → WAIT → NUMBER_READY → NUMBER_PRESS → DONE

실행 순서 (MoveIt 불필요):
  ros2 launch open_manipulator_x_bringup hardware.launch.py
  ros2 launch realsense2_camera rs_launch.py
  ros2 run tf2_ros static_transform_publisher --x 0.12 --y 0.01 --z 0.062 \
      --roll 0 --pitch 0 --yaw 0 --frame-id link5 --child-frame-id camera_link
  python3 nodes/real_robot/real_robot_unified.py

/target_floor 수신 시 자동으로 UP 또는 DOWN 버튼을 누른 뒤,
ELEVATOR_WAIT_SEC 초 대기하고 목표 층 숫자 버튼을 누릅니다.
"""

import math
import threading
import time

import cv2
import numpy as np
import rclpy
import rclpy.time
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Int32, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_ros
import tf2_geometry_msgs

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

try:
    import easyocr
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

# ─── 모델 경로 ────────────────────────────────────────────────────────────────

UPDOWN_MODEL_PATH = '/home/sejong/yolo_dataset_real_v2/runs/train_real_v2/weights/best.pt'
NUM_MODEL_PATH    = '/home/sejong/yolo_dataset_num/runs/segment/train/weights/best.pt'

# ─── OpenMANIPULATOR-X 링크 파라미터 ─────────────────────────────────────────

L1    = 0.0595
L2    = math.sqrt(0.024**2 + 0.128**2)
ALPHA = math.atan2(0.128, 0.024)
L3    = 0.124
L4    = 0.126

JOINT_LIMITS = [
    (-math.pi, math.pi),
    (-1.5,     1.5),
    (-1.5,     1.4),
    (-1.7,     1.97),
]

JOINT_NAMES  = ['joint1', 'joint2', 'joint3', 'joint4']
HOME_JOINTS  = [-3.141, -0.9948, 0.6981, 0.2967]
MOVE_SPEED   = 0.5    # rad/s
MIN_DURATION = 2.0    # 초

# ─── 인식 파라미터 ────────────────────────────────────────────────────────────

UPDOWN_CONF_MIN   = 0.7    # UP/DOWN 인식 신뢰도 기준
NUM_CONF_MIN      = 0.5    # 숫자 박스 신뢰도 기준
NUM_PRESS_CONF    = 0.7    # 숫자 버튼 누르기 신뢰도 기준
OCR_INTERVAL      = 5      # 매 N프레임마다 OCR 실행
BUTTON_OFFSET_X   = 0.075  # 버튼 표면 앞 정지 거리 (m)
ELEVATOR_WAIT_SEC = 5.0    # UP/DOWN 누른 후 엘리베이터 도착 대기 (초)

# ─── 상태 상수 ────────────────────────────────────────────────────────────────

IDLE          = 'IDLE'
UPDOWN_READY  = 'UPDOWN_READY'
UPDOWN_PRESS  = 'UPDOWN_PRESS'
WAIT          = 'WAIT'
NUMBER_READY  = 'NUMBER_READY'
NUMBER_PRESS  = 'NUMBER_PRESS'
DONE          = 'DONE'


# ─── 해석적 IK ────────────────────────────────────────────────────────────────

def solve_ik(X: float, Y: float, Z: float):
    """
    end_effector 수평 접근 해석적 IK.
    Returns [j1, j2, j3, j4] (rad) 또는 None (도달 불가).
    """
    j1 = math.atan2(Y, X)
    # atan2 returns +π for X<0,Y≈0; normalize to -π side so shortest path
    # from HOME (-π) stays within joint limits instead of wrapping out of range
    if j1 > math.pi / 2:
        j1 -= 2 * math.pi
    r  = math.sqrt(X**2 + Y**2)

    wr = r - L4
    wz = Z
    dr = wr
    dz = wz - L1
    D  = math.sqrt(dr**2 + dz**2)

    if D > (L2 + L3) * 0.999:
        return None
    if D < abs(L2 - L3) * 1.001:
        return None

    c_psi = (D**2 - L2**2 - L3**2) / (2.0 * L2 * L3)
    c_psi = max(-1.0, min(1.0, c_psi))

    for psi in (-math.acos(c_psi), math.acos(c_psi)):
        s_psi  = math.sin(psi)
        gamma  = math.atan2(L3 * s_psi, L2 + L3 * c_psi)
        alpha1 = math.atan2(dz, dr) - gamma
        j2     = ALPHA - alpha1
        j3     = -psi - ALPHA
        j4     = -(j2 + j3)

        angles = [j1, j2, j3, j4]
        if all(lo <= a <= hi for a, (lo, hi) in zip(angles, JOINT_LIMITS)):
            return angles

    return None


def _shortest_path(target: float, current: float) -> float:
    diff = (target - current + math.pi) % (2 * math.pi) - math.pi
    return current + diff


def make_trajectory(target_joints: list, current_joints: list, speed: float = MOVE_SPEED):
    target_joints = [_shortest_path(t, c) for t, c in zip(target_joints, current_joints)]
    target_joints = [
        max(lo, min(hi, t))
        for t, (lo, hi) in zip(target_joints, JOINT_LIMITS)
    ]
    max_disp = max(abs(t - c) for t, c in zip(target_joints, current_joints))
    duration = max(max_disp / speed, MIN_DURATION)

    traj = JointTrajectory()
    traj.joint_names = JOINT_NAMES

    pt = JointTrajectoryPoint()
    pt.positions = target_joints
    pt.velocities = [0.0] * 4
    secs  = int(duration)
    nsecs = int((duration - secs) * 1e9)
    pt.time_from_start = Duration(sec=secs, nanosec=nsecs)
    traj.points.append(pt)

    return traj, duration


# ─── 통합 노드 ────────────────────────────────────────────────────────────────

class UnifiedButtonNode(Node):
    def __init__(self):
        super().__init__('real_robot_unified')

        self.lock   = threading.Lock()
        self.bridge = CvBridge()

        # 상태 머신
        self.state = IDLE

        # 층수
        self.target_floor   = None   # 목표 층 (Int32로 수신)
        self.current_floor  = -1     # 현재 층 (UP/DOWN 누른 후 갱신)
        self.target_button  = None   # 'up_button' or 'down_button'

        # 관절 / 이동
        self.current_joints = None
        self.moving         = False

        # 카메라
        self.depth_image = None
        self.fx, self.fy = 615.0, 615.0
        self.cx, self.cy = 320.0, 240.0

        # OCR 캐시
        self.ocr_cache   = {}
        self.frame_count = 0

        # TF
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # arm_controller 액션
        self._arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')

        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        # 공통 구독
        self.create_subscription(Int32,        '/target_floor',  self._cb_target_floor,  10)
        self.create_subscription(JointState,   '/joint_states',  self._cb_joint_state,   10)
        self.create_subscription(PointStamped, '/target_point',  self._cb_target_point,  10)

        # 카메라 공통 구독 (항상 받되, state에 따라 처리)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info',
                                 self._cb_camera_info, 10)
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw',
                                 self._cb_depth, 10)

        # YOLO 모델 로드
        self.updown_model = None
        self.num_model    = None
        self.ocr          = None

        if _YOLO_AVAILABLE:
            try:
                self.updown_model = YOLO(UPDOWN_MODEL_PATH)
                self.get_logger().info('UP/DOWN YOLO 모델 로드 완료')
            except Exception as e:
                self.get_logger().warn(f'UP/DOWN YOLO 로드 실패: {e}')

            try:
                self.num_model = YOLO(NUM_MODEL_PATH)
                self.get_logger().info('숫자 YOLO 모델 로드 완료')
            except Exception as e:
                self.get_logger().warn(f'숫자 YOLO 로드 실패: {e}')

        if _OCR_AVAILABLE and self.num_model is not None:
            self.get_logger().info('EasyOCR 초기화 중...')
            self.ocr = easyocr.Reader(['en'], gpu=False)
            self.get_logger().info('EasyOCR 초기화 완료')

        # 이미지는 모델 준비 상태에서만 구독
        if self.updown_model is not None or self.num_model is not None:
            self.create_subscription(Image, '/camera/camera/color/image_raw',
                                     self._cb_image, 10)

        self.get_logger().info('통합 노드 시작. arm_controller 서버 대기 중...')
        self._home_timer = self.create_timer(2.0, self._move_to_home_once)

    # ─── 공통 콜백 ───────────────────────────────────────────────────────────

    def _cb_joint_state(self, msg: JointState):
        with self.lock:
            self.current_joints = msg

    def _cb_target_floor(self, msg: Int32):
        floor = msg.data
        if self.state not in (IDLE, DONE):
            self.get_logger().warn(f'작업 중 ({self.state}). /target_floor 무시.')
            return
        if floor == self.current_floor:
            self.get_logger().warn(f'이미 {floor}층입니다. 무시.')
            return
        self.target_floor  = floor
        self.target_button = 'up_button' if floor > self.current_floor else 'down_button'
        self.ocr_cache.clear()
        self.frame_count = 0
        self.get_logger().info(
            f'목표 층: {floor}F | {self.target_button} 누르기 대기 중')
        self.state = UPDOWN_READY

    def _cb_camera_info(self, msg: CameraInfo):
        self.fx = msg.k[0]; self.fy = msg.k[4]
        self.cx = msg.k[2]; self.cy = msg.k[5]

    def _cb_depth(self, msg: Image):
        with self.lock:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            self.depth_image = raw.astype(np.float32) / 1000.0   # mm → m

    def _cb_target_point(self, msg: PointStamped):
        """수동 테스트: world 좌표 직접 수신."""
        if self.moving:
            self.get_logger().warn('이동 중. /target_point 무시.')
            return
        X, Y, Z = msg.point.x, msg.point.y, msg.point.z
        self.get_logger().info(f'/target_point 수신: ({X:.3f}, {Y:.3f}, {Z:.3f})')
        threading.Thread(target=self._press_button, args=(X, Y, Z, '수동'), daemon=True).start()

    # ─── 이미지 처리 (state에 따라 분기) ────────────────────────────────────

    def _cb_image(self, msg: Image):
        if self.moving:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        with self.lock:
            depth = self.depth_image.copy() if self.depth_image is not None else None

        state = self.state   # snapshot (GIL 덕에 안전)

        if state == UPDOWN_READY and self.updown_model is not None:
            self._process_updown(frame, depth)
        elif state == NUMBER_READY and self.num_model is not None:
            self._process_number(frame, depth)
        else:
            # 인식하지 않는 phase에도 화면은 표시
            label = f'State: {state} | Target: {self.target_floor}F'
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.imshow('Unified', frame)
            cv2.waitKey(1)

    # ─── Phase 1: UP/DOWN 인식 ───────────────────────────────────────────────

    def _process_updown(self, frame, depth):
        results = self.updown_model(frame, conf=0.5, verbose=False)
        colors  = {'up_button': (0, 255, 0), 'down_button': (0, 0, 255)}

        for result in results:
            for box in result.boxes:
                cls  = result.names[int(box.cls)]
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx_box = (x1 + x2) // 2
                cy_box = (y1 + y2) // 2
                color  = colors.get(cls, (255, 255, 0))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{cls} {conf:.2f}',
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if cls != self.target_button or conf < UPDOWN_CONF_MIN or depth is None:
                    continue

                region = depth[cy_box-2:cy_box+3, cx_box-2:cx_box+3]
                valid  = region[(region > 0) & ~np.isnan(region)]
                if len(valid) == 0:
                    continue
                d = float(np.median(valid))

                X_cam = (cx_box - self.cx) / self.fx * d
                Y_cam = (cy_box - self.cy) / self.fy * d

                pt_cam = PointStamped()
                pt_cam.header.frame_id = 'camera_color_optical_frame'
                pt_cam.header.stamp    = rclpy.time.Time().to_msg()
                pt_cam.point.x = X_cam
                pt_cam.point.y = Y_cam
                pt_cam.point.z = d

                try:
                    pt_w = self.tf_buffer.transform(pt_cam, 'world')
                    X, Y, Z = pt_w.point.x, pt_w.point.y, pt_w.point.z
                    cv2.putText(frame, f'({X:.2f},{Y:.2f},{Z:.2f})',
                                (x1, y1 - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                except Exception as e:
                    self.get_logger().warn(f'TF 변환 실패: {e}')
                    continue

                self.state = UPDOWN_PRESS
                self.get_logger().info(f'{cls} 감지! IK 시작')
                threading.Thread(
                    target=self._press_button,
                    args=(X - BUTTON_OFFSET_X * math.copysign(1.0, X), Y, Z - 0.025, cls),
                    daemon=True,
                ).start()

        cv2.putText(frame, f'UPDOWN | Target: {self.target_button}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow('Unified', frame)
        cv2.waitKey(1)

    # ─── Phase 2: 숫자 인식 ─────────────────────────────────────────────────

    def _process_number(self, frame, depth):
        self.frame_count += 1
        run_ocr = (self.frame_count % OCR_INTERVAL == 0)

        results = self.num_model(frame, conf=NUM_CONF_MIN, verbose=False)

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                pad  = 5
                cx1  = max(0, x1 - pad);  cy1 = max(0, y1 - pad)
                cx2  = min(frame.shape[1], x2 + pad)
                cy2  = min(frame.shape[0], y2 + pad)
                crop = frame[cy1:cy2, cx1:cx2]

                key = f'{x1//20}_{y1//20}_{x2//20}_{y2//20}'
                if run_ocr or key not in self.ocr_cache:
                    self.ocr_cache[key] = self._read_number(crop)
                number = self.ocr_cache.get(key)

                matched = (number == self.target_floor)
                color   = (0, 255, 0) if matched else (180, 180, 180)
                label   = str(number) if number is not None else '?'

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label} {conf:.2f}',
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if matched and conf > NUM_PRESS_CONF and depth is not None:
                    self.state = NUMBER_PRESS
                    self._trigger_number_press(depth, x1, y1, x2, y2)

        cv2.putText(frame, f'NUMBER | Target: {self.target_floor}F',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow('Unified', frame)
        cv2.waitKey(1)

    def _read_number(self, crop):
        """버튼 crop에서 숫자 인식. 실패 시 None."""
        if self.ocr is None:
            return None
        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return None

        scale   = max(64 / max(h, w), 1.0)
        resized = cv2.resize(crop, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
        gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        ocr_results = self.ocr.readtext(enhanced, allowlist='0123456789Bb', detail=1)
        best_conf, best_num = 0.0, None
        for (_, text, conf) in ocr_results:
            text = text.strip().upper()
            if text.isdigit() and conf > best_conf:
                best_conf = conf;  best_num = int(text)
            elif text.startswith('B') and text[1:].isdigit() and conf > best_conf:
                best_conf = conf;  best_num = -int(text[1:])   # B1 → -1
        return best_num

    def _trigger_number_press(self, depth, x1, y1, x2, y2):
        cx_box = (x1 + x2) // 2
        cy_box = (y1 + y2) // 2

        region = depth[cy_box-2:cy_box+3, cx_box-2:cx_box+3]
        valid  = region[(region > 0.1) & ~np.isnan(region)]
        if len(valid) == 0:
            self.get_logger().warn('깊이값 없음. 재시도...')
            self.state = NUMBER_READY
            return
        d = float(np.median(valid))

        X_cam = (cx_box - self.cx) / self.fx * d
        Y_cam = (cy_box - self.cy) / self.fy * d

        pt_cam = PointStamped()
        pt_cam.header.frame_id = 'camera_color_optical_frame'
        pt_cam.header.stamp    = rclpy.time.Time().to_msg()
        pt_cam.point.x = X_cam
        pt_cam.point.y = Y_cam
        pt_cam.point.z = d

        try:
            pt_w = self.tf_buffer.transform(pt_cam, 'world')
            X, Y, Z = pt_w.point.x, pt_w.point.y, pt_w.point.z
        except Exception as e:
            self.get_logger().warn(f'TF 변환 실패: {e}')
            self.state = NUMBER_READY
            return

        self.get_logger().info(
            f'{self.target_floor}층 버튼 감지! 위치: ({X:.3f}, {Y:.3f}, {Z:.3f})')
        threading.Thread(
            target=self._press_button,
            args=(X - BUTTON_OFFSET_X * math.copysign(1.0, X), Y, Z - 0.025,
                  f'{self.target_floor}층'),
            daemon=True,
        ).start()

    # ─── 공통 IK + 이동 ──────────────────────────────────────────────────────

    def _press_button(self, X: float, Y: float, Z: float, label: str = ''):
        joints = solve_ik(X, Y, Z)
        if joints is None:
            self.get_logger().error(
                f'IK 해 없음 [{label}]: 목표({X:.3f},{Y:.3f},{Z:.3f})가 도달 범위 밖')
            self.status_pub.publish(String(data='FAILED'))
            # 이전 phase로 복귀하여 재시도
            if self.state == UPDOWN_PRESS:
                self.state = UPDOWN_READY
            elif self.state == NUMBER_PRESS:
                self.state = NUMBER_READY
            return

        self.get_logger().info(
            f'IK 성공 [{label}]: '
            f'j1={math.degrees(joints[0]):.1f}° '
            f'j2={math.degrees(joints[1]):.1f}° '
            f'j3={math.degrees(joints[2]):.1f}° '
            f'j4={math.degrees(joints[3]):.1f}°')

        ok = self._send_trajectory(joints)
        if ok:
            self.get_logger().info(f'✅ [{label}] 버튼 누르기 성공!')
            self.status_pub.publish(String(data='BUTTON_PRESSED'))

            if self.state == UPDOWN_PRESS:
                self.current_floor = self.target_floor
                self.state = WAIT
                self.get_logger().info('UP/DOWN 완료. home으로 복귀 중...')
                threading.Thread(target=self._return_home_then_wait, daemon=True).start()
            elif self.state == NUMBER_PRESS:
                self.state = DONE
                self.get_logger().info('✅ 전체 시퀀스 완료! 3초 후 home 복귀')
                threading.Timer(3.0, self._move_to_home).start()
        else:
            self.get_logger().error(f'❌ [{label}] 버튼 이동 실패')
            self.status_pub.publish(String(data='FAILED'))
            if self.state == UPDOWN_PRESS:
                self.state = UPDOWN_READY
            elif self.state == NUMBER_PRESS:
                self.state = NUMBER_READY

    def _return_home_then_wait(self):
        ok = self._send_trajectory(HOME_JOINTS)
        if ok:
            self.get_logger().info(
                f'✅ home 복귀 완료. 엘리베이터 대기 {ELEVATOR_WAIT_SEC:.0f}초...')
        else:
            self.get_logger().error('❌ home 복귀 실패. 그래도 대기 후 숫자 인식 진행')
        time.sleep(ELEVATOR_WAIT_SEC)
        self._start_number_phase()

    def _start_number_phase(self):
        self.get_logger().info('숫자 버튼 인식 Phase 시작!')
        self.ocr_cache.clear()
        self.frame_count = 0
        self.state = NUMBER_READY

    # ─── 액션 전송 ───────────────────────────────────────────────────────────

    def _send_trajectory(self, target_joints: list, blocking: bool = True) -> bool:
        if not self._arm_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('arm_controller 액션 서버 없음!')
            return False

        with self.lock:
            js = self.current_joints
        current = [0.0] * 4
        if js is not None:
            for i, name in enumerate(JOINT_NAMES):
                if name in js.name:
                    current[i] = js.position[js.name.index(name)]

        traj, duration = make_trajectory(target_joints, current)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.moving = True
        self.status_pub.publish(String(data='MOVING'))

        future   = self._arm_client.send_goal_async(goal)
        deadline = time.time() + 10.0
        while not future.done():
            if time.time() > deadline:
                self.get_logger().error('액션 수락 타임아웃')
                self.moving = False
                return False
            time.sleep(0.05)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('액션 거부됨')
            self.moving = False
            return False

        if not blocking:
            self.moving = False
            return True

        result_future = goal_handle.get_result_async()
        deadline = time.time() + duration + 5.0
        while not result_future.done():
            if time.time() > deadline:
                self.get_logger().error('이동 실행 타임아웃')
                self.moving = False
                return False
            time.sleep(0.1)

        self.moving = False
        code = result_future.result().result.error_code
        return code == FollowJointTrajectory.Result.SUCCESSFUL

    # ─── Home ────────────────────────────────────────────────────────────────

    def _move_to_home_once(self):
        self._home_timer.cancel()
        threading.Thread(target=self._move_to_home, daemon=True).start()

    def _move_to_home(self):
        self.get_logger().info('home 포지션으로 이동 중...')
        ok = self._send_trajectory(HOME_JOINTS)
        if ok:
            self.get_logger().info('✅ home 도착! 5초 후 인식 준비...')
        else:
            self.get_logger().error('❌ home 이동 실패')

        if self.state == DONE:
            self.state = IDLE
            self.target_floor = None
            self.get_logger().info('✅ 작업 완료. /target_floor 대기 중...')
        elif self.state == IDLE:
            # 최초 기동: 5초 후 IDLE 유지 (target_floor 수신 대기)
            self.get_logger().info('초기 home 완료. /target_floor 대기 중...')


# ─── 엔트리포인트 ─────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = UnifiedButtonNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
