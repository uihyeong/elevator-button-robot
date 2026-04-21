"""
MoveIt 없이 동작하는 숫자 버튼 인식 + 해석적 IK 노드.

YOLO-seg 로 버튼 영역 감지 → EasyOCR 로 숫자 인식 → 해석적 IK → arm_controller 직접 전송.

실행 순서 (MoveIt 불필요):
  ros2 launch open_manipulator_x_bringup hardware.launch.py
  ros2 launch realsense2_camera rs_launch.py
  ros2 run tf2_ros static_transform_publisher --x 0.12 --y 0.01 --z 0.062 \
      --roll 0 --pitch 0 --yaw 0 --frame-id link5 --child-frame-id camera_link
  python3 nodes/real_robot/real_robot_num_ocr_ik.py

수동 테스트 (카메라 없을 때):
  ros2 topic pub /target_point geometry_msgs/PointStamped \
      '{header: {frame_id: "world"}, point: {x: 0.25, y: 0.0, z: 0.2}}'
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

NUM_MODEL_PATH = '/home/sejong/yolo_dataset_num/runs/segment/train/weights/best.pt'

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
MOVE_SPEED   = 0.5   # rad/s
MIN_DURATION = 2.0   # 초

# ─── OCR / 인식 파라미터 ──────────────────────────────────────────────────────

OCR_INTERVAL   = 5      # 매 N프레임마다 OCR 실행
OCR_CONF_MIN   = 0.5
PRESS_CONF_MIN = 0.7
BUTTON_OFFSET  = 0.075  # 버튼 표면 앞 정지 거리 (m)


# ─── 해석적 IK ────────────────────────────────────────────────────────────────

def solve_ik(X: float, Y: float, Z: float):
    """
    end_effector 수평 접근 해석적 IK.
    Returns [j1, j2, j3, j4] (rad) 또는 None (도달 불가).
    """
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


# ─── 메인 노드 ────────────────────────────────────────────────────────────────

class NumOCRIKNode(Node):
    def __init__(self):
        super().__init__('real_robot_num_ocr_ik')

        self.lock   = threading.Lock()
        self.bridge = CvBridge()

        # 상태 플래그
        self.current_joints  = None
        self.depth_image     = None
        self.moving          = False
        self.ready           = False
        self.task_done       = False
        self.button_pressed  = False
        self.target_floor    = None

        # 카메라 내부 파라미터 (camera_info로 자동 갱신)
        self.fx, self.fy = 615.0, 615.0
        self.cx, self.cy = 320.0, 240.0

        # OCR 캐시
        self.ocr_cache   = {}   # {box_key: number}
        self.frame_count = 0

        # TF
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # arm_controller 직접 액션 (MoveIt 불필요)
        self._arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')

        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        # 공통 구독
        self.create_subscription(Int32,      '/target_floor',  self.target_floor_callback, 10)
        self.create_subscription(JointState, '/joint_states',  self.joint_state_callback,  10)
        self.create_subscription(PointStamped, '/target_point', self.target_point_callback, 10)

        # YOLO + EasyOCR + 카메라
        self.model = None
        self.ocr   = None

        if _YOLO_AVAILABLE:
            try:
                self.get_logger().info('YOLO 세그멘테이션 모델 로드 중...')
                self.model = YOLO(NUM_MODEL_PATH)
                self.get_logger().info('YOLO 로드 완료')
            except Exception as e:
                self.get_logger().warn(f'YOLO 로드 실패 ({e}), /target_point 수동 모드로 전환')

        if _OCR_AVAILABLE and self.model is not None:
            self.get_logger().info('EasyOCR 초기화 중... (첫 실행 시 모델 다운로드로 오래 걸릴 수 있음)')
            self.ocr = easyocr.Reader(['en'], gpu=False)
            self.get_logger().info('EasyOCR 초기화 완료')

        if self.model is not None:
            self.create_subscription(
                Image,      '/camera/camera/color/image_raw',      self.image_callback,        10)
            self.create_subscription(
                Image,      '/camera/camera/depth/image_rect_raw', self.depth_callback,        10)
            self.create_subscription(
                CameraInfo, '/camera/camera/color/camera_info',    self.camera_info_callback,  10)
            self.get_logger().info('카메라 모드: YOLO-seg + EasyOCR 활성화')
        else:
            self.get_logger().info('수동 모드: /target_point 로 world 좌표 직접 수신')

        self.get_logger().info('NumOCR IK 노드 시작! /target_floor 대기 중...')
        self._home_timer = self.create_timer(2.0, self._move_to_home_once)

    # ─── 공통 콜백 ───────────────────────────────────────────────────────────

    def joint_state_callback(self, msg: JointState):
        with self.lock:
            self.current_joints = msg

    def target_floor_callback(self, msg: Int32):
        floor = msg.data
        if floor == self.target_floor and (self.button_pressed or self.task_done):
            return
        self.target_floor   = floor
        self.button_pressed = False
        self.task_done      = False
        self.ocr_cache.clear()
        self.get_logger().info(f'목표 층수: {floor}층')

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0];  self.fy = msg.k[4]
        self.cx = msg.k[2];  self.cy = msg.k[5]

    def depth_callback(self, msg: Image):
        with self.lock:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            self.depth_image = raw.astype(np.float32) / 1000.0  # mm → m

    def target_point_callback(self, msg: PointStamped):
        """수동 테스트: world 좌표 직접 수신"""
        if self.moving:
            self.get_logger().warn('이동 중. /target_point 무시.')
            return
        X, Y, Z = msg.point.x, msg.point.y, msg.point.z
        self.get_logger().info(f'/target_point 수신: ({X:.3f}, {Y:.3f}, {Z:.3f})')
        threading.Thread(target=self._press_button, args=(X, Y, Z), daemon=True).start()

    # ─── OCR ─────────────────────────────────────────────────────────────────

    def _box_key(self, x1, y1, x2, y2, grid=20):
        return f'{x1//grid}_{y1//grid}_{x2//grid}_{y2//grid}'

    def _read_number(self, crop):
        """버튼 crop 이미지에서 숫자 인식. 실패 시 None 반환."""
        if self.ocr is None:
            return None
        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return None

        scale   = max(64 / max(h, w), 1.0)
        resized = cv2.resize(crop, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)

        gray  = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        results   = self.ocr.readtext(enhanced, allowlist='0123456789Bb', detail=1)
        best_conf = 0.0
        best_num  = None

        for (_, text, conf) in results:
            text = text.strip().upper()
            if text.isdigit() and conf > best_conf:
                best_conf = conf
                best_num  = int(text)
            elif text.startswith('B') and text[1:].isdigit() and conf > best_conf:
                best_conf = conf
                best_num  = -int(text[1:])   # B1 → -1, B2 → -2

        return best_num

    # ─── 이미지 처리 ─────────────────────────────────────────────────────────

    def image_callback(self, msg: Image):
        if self.target_floor is None or self.model is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.frame_count += 1
        run_ocr = (self.frame_count % OCR_INTERVAL == 0)

        with self.lock:
            depth = self.depth_image.copy() if self.depth_image is not None else None

        results = self.model(frame, conf=OCR_CONF_MIN, verbose=False)

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                pad  = 5
                cx1  = max(0, x1 - pad);  cy1 = max(0, y1 - pad)
                cx2  = min(frame.shape[1], x2 + pad);  cy2 = min(frame.shape[0], y2 + pad)
                crop = frame[cy1:cy2, cx1:cx2]

                key = self._box_key(x1, y1, x2, y2)
                if run_ocr or key not in self.ocr_cache:
                    number = self._read_number(crop)
                    self.ocr_cache[key] = number
                else:
                    number = self.ocr_cache.get(key)

                label   = str(number) if number is not None else '?'
                matched = (number == self.target_floor)
                color   = (0, 255, 0) if matched else (180, 180, 180)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label} {conf:.2f}',
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if (matched and conf > PRESS_CONF_MIN
                        and not self.button_pressed and self.ready and not self.task_done):
                    self._trigger_press(depth, x1, y1, x2, y2)

        cv2.putText(frame, f'Target: {self.target_floor}F  ready={self.ready}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow('Num OCR IK', frame)
        cv2.waitKey(1)

    def _trigger_press(self, depth, x1, y1, x2, y2):
        """목표 버튼 좌표 추출 → IK 스레드 시작"""
        if depth is None:
            return

        cx_box = (x1 + x2) // 2
        cy_box = (y1 + y2) // 2

        region = depth[cy_box-2:cy_box+3, cx_box-2:cx_box+3]
        valid  = region[(region > 0.1) & ~np.isnan(region)]
        if len(valid) == 0:
            self.get_logger().warn('깊이값 없음, 스킵')
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
            return

        self.get_logger().info(
            f'{self.target_floor}층 버튼 감지! 위치: ({X:.3f}, {Y:.3f}, {Z:.3f})')
        self.button_pressed = True
        threading.Thread(
            target=self._press_button,
            args=(X - BUTTON_OFFSET, Y, Z),
            daemon=True,
        ).start()

    # ─── IK + 이동 ───────────────────────────────────────────────────────────

    def _press_button(self, X: float, Y: float, Z: float):
        joints = solve_ik(X, Y, Z)
        if joints is None:
            self.get_logger().error(
                f'IK 해 없음: 목표({X:.3f},{Y:.3f},{Z:.3f})가 도달 범위 밖')
            self.button_pressed = False
            self.status_pub.publish(String(data='FAILED'))
            return

        self.get_logger().info(
            f'IK 성공: '
            f'j1={math.degrees(joints[0]):.1f}° '
            f'j2={math.degrees(joints[1]):.1f}° '
            f'j3={math.degrees(joints[2]):.1f}° '
            f'j4={math.degrees(joints[3]):.1f}°')

        ok = self._send_trajectory(joints)
        if ok:
            self.get_logger().info(f'✅ {self.target_floor}층 버튼 누르기 성공! 3초 후 복귀...')
            self.status_pub.publish(String(data='BUTTON_PRESSED'))
            self.task_done = True
            threading.Timer(3.0, self._move_to_home).start()
        else:
            self.get_logger().error('❌ 버튼 이동 실패')
            self.status_pub.publish(String(data='FAILED'))
            self.button_pressed = False

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
            self.get_logger().info('✅ home 도착! 5초 후 인식 시작...')
        else:
            self.get_logger().error('❌ home 이동 실패')
        self.button_pressed = False
        if not self.task_done:
            threading.Timer(5.0, self._set_ready).start()
        else:
            self.get_logger().info('✅ 작업 완료. /target_floor 대기 중...')
            self.task_done = False

    def _set_ready(self):
        self.ready = True
        self.get_logger().info('✅ 버튼 인식 시작!')


# ─── 엔트리포인트 ─────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = NumOCRIKNode()
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
