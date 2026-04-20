"""
MoveIt 없이 동작하는 Isaac Sim 시뮬레이션 제어 노드.

해석적 IK로 관절 각도 계산 → /joint_target (JointState) 발행 → PID 컨트롤러 → Isaac Sim.

실행 순서 (MoveIt, PID 모두 불필요):
  # 1. Isaac Sim 실행 후 Play ▶️
  # 2. Static TF 발행
  ros2 launch open_manipulator_x_description isaac_sim_tf.launch.py
  # 3. 브릿지 노드 (/joint_command 토픽 연결용)
  ros2 run isaac_moveit_bridge bridge
  # 4. 이 노드
  python3 nodes/simulation/isaac_sim_direct_ik.py

수동 테스트:
  ros2 topic pub /target_point geometry_msgs/PointStamped \
      '{header: {frame_id: "open_manipulator_x"}, point: {x: 0.25, y: 0.0, z: 0.2}}'
"""

import math
import threading
import time

import cv2
import numpy as np
import rclpy
import rclpy.time
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Int32, String
from cv_bridge import CvBridge
from ultralytics import YOLO
import tf2_ros
import tf2_geometry_msgs

# ─── OpenMANIPULATOR-X 링크 파라미터 ─────────────────────────────────────────
# URDF joint 오프셋:
#   joint2→joint3: xyz=(0.024, 0, 0.128) → ALPHA = atan2(0.128, 0.024) ≈ 79.4°
#   joint3→joint4: xyz=(0.124, 0, 0.0)  → 순수 x방향
L1    = 0.0595
L2    = math.sqrt(0.024**2 + 0.128**2)
ALPHA = math.atan2(0.128, 0.024)   # ≈ 1.385 rad (79.4°)
L3    = 0.124
L4    = 0.126

JOINT_LIMITS = [
    (-math.pi,  math.pi),
    (-1.5,      1.5),
    (-1.5,      1.4),
    (-1.7,      1.97),
]

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4']
HOME_JOINTS  = [0.0, -0.9948, 0.6981, 0.2967]

INTERP_HZ    = 50     # 보간 발행 주파수 (PID와 동일)
MOVE_SPEED   = 0.4    # rad/s — 보간 이동 속도
MIN_DURATION = 2.0    # 최소 이동 시간 (초)
READY_DELAY  = 3.0    # home 도착 후 인식 시작까지 대기 (초)
RETURN_DELAY = 3.0    # 버튼 누른 후 복귀까지 대기 (초)


# ─── 해석적 IK ────────────────────────────────────────────────────────────────

def solve_ik(X: float, Y: float, Z: float):
    """
    OpenMANIPULATOR-X 수평 접근 해석적 IK.
    Returns [j1, j2, j3, j4] (rad) 또는 None.
    """
    j1 = math.atan2(Y, X)
    r  = math.sqrt(X**2 + Y**2)

    wr = r - L4   # 수평 EEF: wrist는 목표에서 L4만큼 뒤
    wz = Z
    dr = wr
    dz = wz - L1
    D  = math.sqrt(dr**2 + dz**2)

    if D > (L2 + L3) * 0.999:
        return None
    if D < abs(L2 - L3) * 1.001:
        return None

    c_psi = max(-1.0, min(1.0, (D**2 - L2**2 - L3**2) / (2.0 * L2 * L3)))

    for psi in (-math.acos(c_psi), math.acos(c_psi)):
        s_psi  = math.sin(psi)
        gamma  = math.atan2(L3 * s_psi, L2 + L3 * c_psi)
        alpha1 = math.atan2(dz, dr) - gamma   # link2 수평각 = ALPHA - j2
        j2     = ALPHA - alpha1
        j3     = -psi - ALPHA
        j4     = -(j2 + j3)

        angles = [j1, j2, j3, j4]
        if all(lo <= a <= hi for a, (lo, hi) in zip(angles, JOINT_LIMITS)):
            return angles

    return None


# ─── 메인 노드 ────────────────────────────────────────────────────────────────

class IsaacSimDirectIK(Node):
    def __init__(self):
        super().__init__('isaac_sim_direct_ik')

        self.lock = threading.Lock()
        self.bridge = CvBridge()
        self.model = YOLO('/home/sejong/yolo_dataset_real_v2/runs/train_real_v2/weights/best.pt')

        self.depth_image = None
        self.moving = False
        self.ready = False
        self.task_done = False
        self.button_pressed = False
        self.target_button = None
        self.current_floor = -1
        self.target_floor = None

        # 카메라 파라미터 기본값
        self.fx, self.fy = 615.0, 615.0
        self.cx, self.cy = 320.0, 240.0

        # TF
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.current_joints = None

        # /joint_command 직접 발행 (bridge와 동일한 방식, PID 불필요)
        self.joint_pub = self.create_publisher(JointState, '/joint_command', 10)

        # 상태 발행
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        # 수동 테스트용
        self.create_subscription(
            PointStamped, '/target_point', self.target_point_callback, 10)

        # 층수 인터페이스
        self.create_subscription(Int32, '/target_floor', self.target_floor_callback, 10)
        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)

        # 시뮬레이션 카메라 토픽 (/camera/color/... — 실제 로봇과 다름)
        self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)
        self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)

        self.get_logger().info('Isaac Sim Direct IK 노드 시작!')
        self._home_timer = self.create_timer(2.0, self._move_to_home_once)

    # ─── 콜백 ────────────────────────────────────────────────────────────────

    def _joint_state_cb(self, msg: JointState):
        with self.lock:
            self.current_joints = msg

    def target_floor_callback(self, msg: Int32):
        floor = msg.data
        if floor == self.current_floor:
            self.get_logger().warn(f'현재 층({floor})과 동일. 무시.')
            return
        self.target_floor = floor
        self.target_button = 'up_button' if floor > self.current_floor else 'down_button'
        self.button_pressed = False
        self.task_done = False
        self.get_logger().info(f'목표 층: {floor}F | {self.target_button} 누르기 대기 중')

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]; self.fy = msg.k[4]
        self.cx = msg.k[2]; self.cy = msg.k[5]

    def depth_callback(self, msg: Image):
        with self.lock:
            # 시뮬레이션: 32FC1 (이미 미터 단위, 변환 불필요)
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        with self.lock:
            depth = self.depth_image.copy() if self.depth_image is not None else None

        results = self.model(frame, conf=0.5, verbose=False)
        colors = {'up_button': (0, 255, 0), 'down_button': (0, 0, 255)}

        for result in results:
            for box in result.boxes:
                cls  = result.names[int(box.cls)]
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx_box = (x1 + x2) // 2
                cy_box = (y1 + y2) // 2
                color  = colors.get(cls, (255, 255, 0))

                if depth is None:
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
                    # 시뮬레이션 베이스 프레임: open_manipulator_x
                    pt_w = self.tf_buffer.transform(pt_cam, 'open_manipulator_x')
                    X, Y, Z = pt_w.point.x, pt_w.point.y, pt_w.point.z

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{cls} {conf:.2f}', (x1, y1 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(frame, f'({X:.2f},{Y:.2f},{Z:.2f})',
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    if (cls == self.target_button and conf > 0.7
                            and not self.button_pressed
                            and self.ready and not self.task_done):
                        self.button_pressed = True
                        self.get_logger().info(
                            f'{cls} 감지! 3D 위치(open_manipulator_x): '
                            f'X={X:.3f} Y={Y:.3f} Z={Z:.3f}')
                        threading.Thread(
                            target=self._press_button,
                            args=(X, Y, Z), daemon=True).start()

                except Exception as e:
                    self.get_logger().warn(f'TF 변환 실패: {e}')

        cv2.imshow('Isaac Sim Direct IK', frame)
        cv2.waitKey(1)

    def target_point_callback(self, msg: PointStamped):
        if self.moving:
            self.get_logger().warn('이동 중. /target_point 무시.')
            return
        X, Y, Z = msg.point.x, msg.point.y, msg.point.z
        self.get_logger().info(f'/target_point 수신: ({X:.3f}, {Y:.3f}, {Z:.3f})')
        threading.Thread(target=self._press_button, args=(X, Y, Z), daemon=True).start()

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
            f'IK 성공: j1={math.degrees(joints[0]):.1f}° '
            f'j2={math.degrees(joints[1]):.1f}° '
            f'j3={math.degrees(joints[2]):.1f}° '
            f'j4={math.degrees(joints[3]):.1f}°')

        self.status_pub.publish(String(data='MOVING'))
        self._interp_move(joints)

        self.get_logger().info('✅ 버튼 누르기 완료! 3초 후 복귀...')
        self.status_pub.publish(String(data='BUTTON_PRESSED'))
        self.task_done = True
        if self.target_floor is not None:
            self.current_floor = self.target_floor
        threading.Timer(RETURN_DELAY, self._move_to_home).start()

    def _interp_move(self, target_joints: list):
        """
        현재 관절 위치 → 목표 관절 위치를 INTERP_HZ 로 보간하며 발행.
        PID가 급격한 목표 변화 없이 부드럽게 추종하도록 함.
        """
        with self.lock:
            js = self.current_joints

        # 현재 위치 읽기 (없으면 home으로 가정)
        start = list(HOME_JOINTS)
        if js is not None:
            for i, name in enumerate(JOINT_NAMES):
                if name in js.name:
                    start[i] = js.position[js.name.index(name)]

        max_disp = max(abs(t - s) for t, s in zip(target_joints, start))
        duration = max(max_disp / MOVE_SPEED, MIN_DURATION)
        steps    = int(duration * INTERP_HZ)
        dt       = duration / steps

        for k in range(1, steps + 1):
            alpha = k / steps   # 0→1 선형 보간
            waypoint = [s + alpha * (t - s) for s, t in zip(start, target_joints)]
            self._publish_target(waypoint)
            time.sleep(dt)

    def _publish_target(self, joints: list):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = JOINT_NAMES
        msg.position = joints
        self.joint_pub.publish(msg)

    # ─── Home ────────────────────────────────────────────────────────────────

    def _move_to_home_once(self):
        self._home_timer.cancel()
        threading.Thread(target=self._move_to_home, daemon=True).start()

    def _move_to_home(self):
        self.get_logger().info('home 포지션으로 이동 중...')
        self.moving = True
        self._interp_move(HOME_JOINTS)
        self.moving = False
        self.button_pressed = False
        self.get_logger().info('✅ home 도착! 5초 후 인식 시작...')
        if not self.task_done:
            threading.Timer(READY_DELAY, self._set_ready).start()
        else:
            self.get_logger().info('✅ 작업 완료. 대기 중...')

    def _set_ready(self):
        self.ready = True
        self.get_logger().info('✅ 버튼 인식 시작!')


# ─── 엔트리포인트 ────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = IsaacSimDirectIK()
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
