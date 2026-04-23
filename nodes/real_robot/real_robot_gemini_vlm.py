"""
Gemini VLM 전용 엘리베이터 버튼 인식 노드.

YOLO + EasyOCR 대신 Gemini Vision API 단일 호출로 버튼을 인식한다.

Phase 1 (UPDOWN):  Gemini → UP/DOWN 버튼 위치 반환 → 해석적 IK → 누르기
Phase 2 (NUMBER):  Gemini → 목표 층 숫자 버튼 위치 반환 → 해석적 IK → 누르기

실행 전 준비:
  pip install google-genai
  export GEMINI_API_KEY="your_key_here"   # aistudio.google.com/apikey

실행 순서:
  ros2 launch open_manipulator_x_bringup hardware.launch.py
  ros2 launch realsense2_camera rs_launch.py
  ros2 run tf2_ros static_transform_publisher --x 0.12 --y 0.01 --z 0.062 \
      --roll 0 --pitch 0 --yaw 0 --frame-id link5 --child-frame-id camera_link
  python3 nodes/real_robot/real_robot_gemini_vlm.py

층수 입력:
  ros2 topic pub --once /target_floor std_msgs/Int32 "{data: 3}"
"""

import math
import os
import threading
import time
import json

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

# ─── Gemini SDK ───────────────────────────────────────────────────────────────

try:
    from google import genai
    from google.genai import types as genai_types
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

# ─── 설정 ─────────────────────────────────────────────────────────────────────

GEMINI_MODEL    = 'gemini-2.5-flash'
GEMINI_INTERVAL = 2.0    # API 호출 최소 간격 (초)

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

BUTTON_OFFSET_X   = 0.05
ELEVATOR_WAIT_SEC = 5.0

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
    j1 = math.atan2(Y, X)
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


# ─── Gemini 클라이언트 ────────────────────────────────────────────────────────

class GeminiClient:
    def __init__(self, api_key: str, model: str):
        self.model   = model
        self._client = genai.Client(api_key=api_key)

    def ask(self, frame: np.ndarray, prompt: str) -> str:
        _, buf = cv2.imencode('.jpg', frame)
        resp = self._client.models.generate_content(
            model=self.model,
            contents=[
                genai_types.Part.from_bytes(data=buf.tobytes(), mime_type='image/jpeg'),
                prompt,
            ],
            config=genai_types.GenerateContentConfig(
                response_mime_type='application/json',
            ),
        )
        return resp.text


# ─── 프롬프트 ─────────────────────────────────────────────────────────────────

def _updown_prompt(target: str) -> str:
    return (
        f'Find the elevator UP and DOWN call buttons. '
        f'Return JSON only: [{{"point": [y, x], "label": "UP"}}, ...]. '
        f'point is [y, x] normalized 0-1000. Empty array [] if not found. '
        f'Target to press: {target}'
    )

def _number_prompt(target_floor: int) -> str:
    hint = f'B{abs(target_floor)}' if target_floor < 0 else str(target_floor)
    return (
        f'Find all elevator floor buttons in the panel. '
        f'Return JSON only: [{{"point": [y, x], "label": "<text>", "floor": <int>}}, ...]. '
        f'point is [y, x] normalized 0-1000. floor is integer (B1=-1, B2=-2). '
        f'Empty array [] if not found. Target floor: {hint}'
    )


# ─── Gemini 응답 파싱 ─────────────────────────────────────────────────────────

def _parse_json(text: str) -> list:
    cleaned = text.strip().lstrip('`').rstrip('`')
    if cleaned.startswith('json'):
        cleaned = cleaned[4:]
    data = json.loads(cleaned)
    return data if isinstance(data, list) else []

def parse_updown(text: str, target: str):
    """반환: {'label', 'cx_norm', 'cy_norm'} 또는 None. cx/cy_norm 은 0~1."""
    try:
        for item in _parse_json(text):
            lbl = str(item.get('label', '')).upper()
            pt  = item.get('point')
            if lbl != target.upper() or not pt or len(pt) != 2:
                continue
            return {
                'label':   lbl,
                'cy_norm': float(pt[0]) / 1000.0,
                'cx_norm': float(pt[1]) / 1000.0,
            }
    except Exception:
        pass
    return None

def parse_number(text: str, target_floor: int):
    """반환: {'label', 'floor', 'cx_norm', 'cy_norm'} 또는 None."""
    try:
        for item in _parse_json(text):
            floor = int(item.get('floor', 9999))
            pt    = item.get('point')
            if floor != target_floor or not pt or len(pt) != 2:
                continue
            return {
                'label':   str(item.get('label', floor)),
                'floor':   floor,
                'cy_norm': float(pt[0]) / 1000.0,
                'cx_norm': float(pt[1]) / 1000.0,
            }
    except Exception:
        pass
    return None


# ─── 메인 노드 ────────────────────────────────────────────────────────────────

class GeminiButtonNode(Node):
    def __init__(self):
        super().__init__('real_robot_gemini_vlm')

        # Gemini API 키 확인
        api_key = os.environ.get('GEMINI_API_KEY', '')
        if not api_key:
            self.get_logger().error(
                'GEMINI_API_KEY 환경변수가 없습니다. '
                'export GEMINI_API_KEY="your_key" 후 재실행하세요.')
            raise RuntimeError('GEMINI_API_KEY missing')

        if not _GEMINI_AVAILABLE:
            self.get_logger().error(
                'Gemini SDK 미설치. pip install google-genai 실행 후 재실행하세요.')
            raise RuntimeError('google-genai not installed')

        self.gemini  = GeminiClient(api_key, GEMINI_MODEL)
        self.get_logger().info(f'Gemini 클라이언트 초기화 완료 (model={GEMINI_MODEL})')

        self.lock   = threading.Lock()
        self.bridge = CvBridge()

        # 상태 머신
        self.state         = IDLE
        self.target_floor  = None
        self.current_floor = -1
        self.target_button = None   # 'UP' | 'DOWN'

        # 관절 / 이동
        self.current_joints = None
        self.moving         = False

        # 카메라
        self.depth_image = None
        self.fx, self.fy = 615.0, 615.0
        self.cx, self.cy = 320.0, 240.0

        # Gemini 호출 제어 — phase당 1회만 호출
        self._gemini_busy        = False
        self._phase_called       = False   # 현재 phase에서 이미 호출했으면 True
        self._last_detection     = None
        self._last_raw_text      = ''

        # TF
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # arm_controller 액션
        self._arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')

        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        # 구독
        self.create_subscription(Int32,      '/target_floor', self._cb_target_floor,  10)
        self.create_subscription(JointState, '/joint_states', self._cb_joint_state,   10)
        self.create_subscription(CameraInfo,
            '/camera/camera/color/camera_info', self._cb_camera_info, 10)
        self.create_subscription(Image,
            '/camera/camera/depth/image_rect_raw', self._cb_depth, 10)
        self.create_subscription(Image,
            '/camera/camera/color/image_raw', self._cb_image, 10)

        self.get_logger().info('GeminiButtonNode 시작. arm_controller 대기 중...')
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
        self.target_floor    = floor
        self.target_button   = 'UP' if floor > self.current_floor else 'DOWN'
        self._last_detection = None
        self._last_raw_text  = ''
        self._phase_called   = False
        self.get_logger().info(
            f'목표 층: {floor}F | {self.target_button} 버튼 누르기 대기')
        self.state = UPDOWN_READY

    def _cb_camera_info(self, msg: CameraInfo):
        self.fx = msg.k[0]; self.fy = msg.k[4]
        self.cx = msg.k[2]; self.cy = msg.k[5]

    def _cb_depth(self, msg: Image):
        with self.lock:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            self.depth_image = raw.astype(np.float32) / 1000.0

    # ─── 이미지 처리 ─────────────────────────────────────────────────────────

    def _cb_image(self, msg: Image):
        if self.moving:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        state = self.state

        if state in (UPDOWN_READY, NUMBER_READY):
            self._maybe_call_gemini(frame, state)

        self._draw_and_show(frame, state)

    def _maybe_call_gemini(self, frame: np.ndarray, state: str):
        """phase당 1회만 Gemini 호출."""
        if self._gemini_busy or self._phase_called:
            return

        self._gemini_busy  = True
        self._phase_called = True

        frame_copy = frame.copy()
        threading.Thread(
            target=self._call_gemini_thread,
            args=(frame_copy, state),
            daemon=True,
        ).start()

    def _call_gemini_thread(self, frame: np.ndarray, state: str):
        """백그라운드: Gemini API 호출 → 결과 파싱 → 버튼 누르기 트리거."""
        try:
            h, w = frame.shape[:2]

            if state == UPDOWN_READY:
                prompt = _updown_prompt(self.target_button)
                text   = self.gemini.ask(frame, prompt)
                self._last_raw_text = text
                self.get_logger().info(f'[Gemini UPDOWN] 응답: {text[:200]}')

                det = parse_updown(text, self.target_button)
                self._last_detection = det

                if det and self.state == UPDOWN_READY:
                    self._trigger_updown(frame, det, w, h)

            elif state == NUMBER_READY:
                prompt = _number_prompt(self.target_floor)
                text   = self.gemini.ask(frame, prompt)
                self._last_raw_text = text
                self.get_logger().info(f'[Gemini NUMBER] 응답: {text[:200]}')

                det = parse_number(text, self.target_floor)
                self._last_detection = det

                if det and self.state == NUMBER_READY:
                    self._trigger_number(frame, det, w, h)

        except Exception as e:
            self.get_logger().error(f'Gemini 호출 오류: {e}')
        finally:
            self._gemini_busy = False

    # ─── 감지 → IK 트리거 ────────────────────────────────────────────────────

    def _trigger_updown(self, frame: np.ndarray, det: dict, w: int, h: int):
        """UP/DOWN 버튼 감지 결과 → 3D 좌표 변환 → 누르기."""
        cx_px = int(det['cx_norm'] * w)
        cy_px = int(det['cy_norm'] * h)

        with self.lock:
            depth = self.depth_image.copy() if self.depth_image is not None else None

        if depth is None:
            self.get_logger().warn('깊이 이미지 없음. 재시도...')
            return

        region = depth[max(0, cy_px-2):cy_px+3, max(0, cx_px-2):cx_px+3]
        valid  = region[(region > 0.1) & ~np.isnan(region)]
        if len(valid) == 0:
            self.get_logger().warn('깊이값 없음 (버튼 중심). 재시도...')
            return
        d = float(np.median(valid))

        X_cam = (cx_px - self.cx) / self.fx * d
        Y_cam = (cy_px - self.cy) / self.fy * d

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
            return

        self.get_logger().info(
            f'{det["label"]} 감지! world=({X:.3f},{Y:.3f},{Z:.3f}) depth={d:.3f}m')

        self.state = UPDOWN_PRESS
        threading.Thread(
            target=self._press_button,
            args=(X - BUTTON_OFFSET_X * math.copysign(1.0, X), Y, Z - 0.025,
                  det['label']),
            daemon=True,
        ).start()

    def _trigger_number(self, frame: np.ndarray, det: dict, w: int, h: int):
        """숫자 버튼 감지 결과 → 3D 좌표 변환 → 누르기."""
        cx_px = int(det['cx_norm'] * w)
        cy_px = int(det['cy_norm'] * h)

        with self.lock:
            depth = self.depth_image.copy() if self.depth_image is not None else None

        if depth is None:
            self.get_logger().warn('깊이 이미지 없음. 재시도...')
            return

        region = depth[max(0, cy_px-2):cy_px+3, max(0, cx_px-2):cx_px+3]
        valid  = region[(region > 0.1) & ~np.isnan(region)]
        if len(valid) == 0:
            self.get_logger().warn('깊이값 없음 (숫자 버튼). 재시도...')
            return
        d = float(np.median(valid))

        X_cam = (cx_px - self.cx) / self.fx * d
        Y_cam = (cy_px - self.cy) / self.fy * d

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
            return

        self.get_logger().info(
            f'{det["label"]}층 버튼 감지! world=({X:.3f},{Y:.3f},{Z:.3f}) depth={d:.3f}m')

        self.state = NUMBER_PRESS
        threading.Thread(
            target=self._press_button,
            args=(X - BUTTON_OFFSET_X * math.copysign(1.0, X), Y, Z - 0.025,
                  f'{det["label"]}층'),
            daemon=True,
        ).start()

    # ─── 화면 표시 ───────────────────────────────────────────────────────────

    def _draw_and_show(self, frame: np.ndarray, state: str):
        h, w = frame.shape[:2]
        det = self._last_detection

        # 감지 포인트 표시
        if det and 'cx_norm' in det:
            cx = int(det['cx_norm'] * w)
            cy = int(det['cy_norm'] * h)
            cv2.circle(frame, (cx, cy), 18, (0, 255, 0), 3)
            cv2.circle(frame, (cx, cy), 4,  (0, 255, 0), -1)
            cv2.putText(frame, det.get('label', '?'),
                        (cx + 22, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 상태 HUD
        busy_mark = ' [Gemini...]' if self._gemini_busy else ''
        hud = f'State: {state}{busy_mark}  Target: {self.target_floor}F'
        cv2.putText(frame, hud, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        # Gemini 응답 원문 (디버그)
        if self._last_raw_text:
            snippet = self._last_raw_text[:120].replace('\n', ' ')
            cv2.putText(frame, snippet, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        cv2.imshow('Gemini VLM', frame)
        cv2.waitKey(1)

    # ─── 공통 IK + 이동 ──────────────────────────────────────────────────────

    def _press_button(self, X: float, Y: float, Z: float, label: str = ''):
        joints = solve_ik(X, Y, Z)
        if joints is None:
            self.get_logger().error(
                f'IK 해 없음 [{label}]: ({X:.3f},{Y:.3f},{Z:.3f}) 도달 불가')
            self.status_pub.publish(String(data='FAILED'))
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
                self.get_logger().info('UP/DOWN 완료. home 복귀 후 엘리베이터 대기...')
                threading.Thread(target=self._return_home_then_wait, daemon=True).start()
            elif self.state == NUMBER_PRESS:
                self.state = DONE
                self.get_logger().info('✅ 전체 시퀀스 완료! 3초 후 home 복귀')
                threading.Timer(3.0, self._move_to_home).start()
        else:
            self.get_logger().error(f'❌ [{label}] 이동 실패')
            self.status_pub.publish(String(data='FAILED'))
            if self.state == UPDOWN_PRESS:
                self.state = UPDOWN_READY
            elif self.state == NUMBER_PRESS:
                self.state = NUMBER_READY

    def _return_home_then_wait(self):
        ok = self._send_trajectory(HOME_JOINTS)
        if ok:
            self.get_logger().info(
                f'home 복귀 완료. {ELEVATOR_WAIT_SEC:.0f}초 대기...')
        else:
            self.get_logger().error('home 복귀 실패. 그래도 대기 진행')
        time.sleep(ELEVATOR_WAIT_SEC)
        self.get_logger().info('숫자 버튼 인식 Phase 시작!')
        self._last_detection = None
        self._last_raw_text  = ''
        self._phase_called   = False
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
            self.get_logger().info('✅ home 도착!')
        else:
            self.get_logger().error('❌ home 이동 실패')

        if self.state == DONE:
            self.state = IDLE
            self.target_floor = None
            self.get_logger().info('작업 완료. /target_floor 대기 중...')
        elif self.state == IDLE:
            self.get_logger().info('초기 home 완료. /target_floor 대기 중...')


# ─── 엔트리포인트 ─────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    try:
        node = GeminiButtonNode()
    except RuntimeError as e:
        print(f'노드 초기화 실패: {e}')
        rclpy.shutdown()
        return

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
