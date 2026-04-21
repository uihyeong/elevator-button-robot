"""
MoveIt 없이 동작하는 실제 로봇 제어 노드.

해석적 IK로 관절 각도 계산 → /arm_controller/follow_joint_trajectory 로 직접 전송.

실행 순서 (MoveIt 불필요):
  ros2 launch open_manipulator_x_bringup hardware.launch.py
  ros2 run tf2_ros static_transform_publisher --x 0.12 --y 0.01 --z 0.062 \
      --roll 0 --pitch 0 --yaw 0 --frame-id link5 --child-frame-id camera_link
  python3 nodes/real_robot/real_robot_direct_ik.py

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

# YOLO는 선택적 의존성 (카메라 없을 때도 동작)
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# ─── OpenMANIPULATOR-X 링크 파라미터 ─────────────────────────────────────────
# URDF 기준 joint 오프셋:
#   joint2→joint3: xyz=(0.024, 0, 0.128) → link2가 z방향으로 0.128m, x방향으로 0.024m
#   joint3→joint4: xyz=(0.124, 0, 0.0)  → link3가 x방향으로만 0.124m
#   ALPHA = atan2(z_offset, x_offset) = atan2(0.128, 0.024) ≈ 79.4°
L1    = 0.0595                             # joint2 높이 (base→joint2, z축)
L2    = math.sqrt(0.024**2 + 0.128**2)    # joint2→joint3 유효 길이 (≈0.1302m)
ALPHA = math.atan2(0.128, 0.024)          # link2의 수평 기준 각도 (≈1.385 rad = 79.4°)
L3    = 0.124                              # joint3→joint4 (순수 x방향, 오프셋 없음)
L4    = 0.126                              # joint4→end_effector

# 관절 한계 [lower, upper] (rad) — URDF 기준
JOINT_LIMITS = [
    (-math.pi,  math.pi),  # joint1
    (-1.5,      1.5),      # joint2
    (-1.5,      1.4),      # joint3
    (-1.7,      1.97),     # joint4
]

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4']

HOME_JOINTS  = [-3.141, -0.9948, 0.6981, 0.2967]   # home 포지션 (라디안)
MOVE_SPEED   = 0.5   # rad/s (최대 이동 속도, 안전용 제한)
MIN_DURATION = 2.0   # 최소 이동 시간 (초)


# ─── 해석적 IK ────────────────────────────────────────────────────────────────

def solve_ik(X: float, Y: float, Z: float):
    """
    OpenMANIPULATOR-X 수평 접근 해석적 IK.
    end_effector가 수평으로 버튼을 향해 접근하는 해를 반환.

    Args:
        X, Y, Z: 목표 좌표 (world 프레임, 미터)

    Returns:
        [j1, j2, j3, j4] (rad) 또는 None (도달 불가)
    """
    # joint1: 수평 방향 회전
    j1 = math.atan2(Y, X)

    # r-z 평면으로 투영
    r = math.sqrt(X**2 + Y**2)
    z = Z

    # 수평 접근 시 wrist(joint4) 위치: end_effector에서 L4만큼 뒤로
    wr = r - L4
    wz = z

    # joint2에서 wrist까지 벡터
    dr = wr
    dz = wz - L1
    D  = math.sqrt(dr**2 + dz**2)

    # 도달 가능 범위 체크
    if D > (L2 + L3) * 0.999:
        return None   # 너무 멀다
    if D < abs(L2 - L3) * 1.001:
        return None   # 너무 가깝다

    # ─── 2-link IK 수식 유도 ────────────────────────────────────────────────
    # FK에서 link2(joint2→joint3) 방향 = R_y(j2) * (0.024, 0, 0.128)
    #   r성분 = L2*cos(j2 - ALPHA),  z성분 = L2*sin(ALPHA - j2)
    #   → link2 수평각 φ2 = ALPHA - j2  → j2 = ALPHA - φ2
    # link3(joint3→joint4) 방향 = R_y(j2+j3) * (0.124, 0, 0)
    #   r성분 = L3*cos(j2+j3),      z성분 = -L3*sin(j2+j3)
    #   → link3 수평각 φ3 = -(j2+j3)
    #
    # 표준 2R IK: alpha1 = φ2 (link2 수평각)
    # psi = θ_3 = φ3 - φ2 (elbow 상대각) → φ3 = φ2 + psi
    # j2+j3 = -φ3 = -(φ2 + psi) → j3 = -φ2 - psi - j2 = -ALPHA - psi
    # ────────────────────────────────────────────────────────────────────────

    c_psi = (D**2 - L2**2 - L3**2) / (2.0 * L2 * L3)
    c_psi = max(-1.0, min(1.0, c_psi))

    # 두 가지 해(psi ± ) 모두 시도, 관절 한계를 통과하는 첫 번째 반환
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

    return None   # 두 해 모두 관절 한계 위반


def _shortest_path(target: float, current: float) -> float:
    """최단 경로 각도 정규화: current 기준으로 ±π 이내로 target을 조정."""
    diff = (target - current + math.pi) % (2 * math.pi) - math.pi
    return current + diff


def make_trajectory(target_joints: list, current_joints: list, speed: float = MOVE_SPEED):
    """
    현재 → 목표 관절 각도를 잇는 단순 1-웨이포인트 JointTrajectory 생성.
    이동 시간은 최대 관절 변위 / speed 로 계산.
    """
    target_joints = [_shortest_path(t, c) for t, c in zip(target_joints, current_joints)]
    max_disp = max(abs(t - c) for t, c in zip(target_joints, current_joints))
    duration = max(max_disp / speed, MIN_DURATION)

    traj = JointTrajectory()
    traj.joint_names = JOINT_NAMES

    pt = JointTrajectoryPoint()
    pt.positions = target_joints
    pt.velocities = [0.0] * 4
    secs = int(duration)
    nsecs = int((duration - secs) * 1e9)
    pt.time_from_start = Duration(sec=secs, nanosec=nsecs)
    traj.points.append(pt)

    return traj, duration


# ─── 메인 노드 ────────────────────────────────────────────────────────────────

class DirectIKNode(Node):
    def __init__(self):
        super().__init__('real_robot_direct_ik')

        self.lock = threading.Lock()
        self.bridge = CvBridge()

        self.current_joints = None      # 최신 관절 상태
        self.moving = False             # 이동 중 플래그
        self.ready = False              # home 완료 후 인식 시작
        self.task_done = False
        self.button_pressed = False
        self.target_button = None
        self.current_floor = -1
        self.target_floor = None

        # 카메라 파라미터
        self.fx, self.fy = 615.0, 615.0
        self.cx, self.cy = 320.0, 240.0
        self.depth_image = None

        # TF
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # arm_controller 직접 액션 (MoveIt 불필요)
        self._arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')

        # 상태 발행
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        # 수동 테스트용: world 프레임 좌표 직접 수신
        self.create_subscription(
            PointStamped, '/target_point', self.target_point_callback, 10)

        # 층수 인터페이스 (자율주행 로봇과 연동)
        self.create_subscription(Int32, '/target_floor', self.target_floor_callback, 10)

        # 관절 상태
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        # YOLO + 카메라 (카메라 없으면 스킵)
        if _YOLO_AVAILABLE:
            try:
                self.model = YOLO(
                    '/home/sejong/yolo_dataset_real_v2/runs/train_real_v2/weights/best.pt')
                self.create_subscription(
                    Image, '/camera/camera/color/image_raw', self.image_callback, 10)
                self.create_subscription(
                    Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
                self.create_subscription(
                    CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
                self.get_logger().info('YOLO 모델 로드 완료 (카메라 모드)')
            except Exception as e:
                self.model = None
                self.get_logger().warn(f'YOLO 로드 실패 ({e}), 수동 모드로 전환')
        else:
            self.model = None
            self.get_logger().info('ultralytics 미설치 — 수동 /target_point 모드')

        self.get_logger().info('DirectIK 노드 시작. arm_controller 서버 대기 중...')
        self._home_timer = self.create_timer(2.0, self._move_to_home_once)

    # ─── 콜백 ────────────────────────────────────────────────────────────────

    def joint_state_callback(self, msg: JointState):
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
        self.get_logger().info(
            f'목표 층: {floor}F | {self.target_button} 누르기 대기 중')

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]; self.fy = msg.k[4]
        self.cx = msg.k[2]; self.cy = msg.k[5]

    def depth_callback(self, msg: Image):
        with self.lock:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            self.depth_image = raw.astype(np.float32) / 1000.0  # mm → m

    def image_callback(self, msg: Image):
        if self.model is None:
            return
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
                    pt_w = self.tf_buffer.transform(pt_cam, 'world')
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
                        self.get_logger().info(f'{cls} 감지! IK 계산 시작')
                        threading.Thread(
                            target=self._press_button,
                            args=(X + 0.05, Y - 0.02, Z - 0.03), daemon=True).start()

                except Exception as e:
                    self.get_logger().warn(f'TF 변환 실패: {e}')

        cv2.imshow('DirectIK', frame)
        cv2.waitKey(1)

    def target_point_callback(self, msg: PointStamped):
        """
        수동 테스트: /target_point 토픽으로 world 좌표 직접 수신.
        이동 중이거나 task_done이면 무시.
        """
        if self.moving:
            self.get_logger().warn('이동 중. /target_point 무시.')
            return
        X = msg.point.x
        Y = msg.point.y
        Z = msg.point.z
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

        ok = self._send_trajectory(joints)
        if ok:
            self.get_logger().info('✅ 버튼 누르기 성공! 3초 후 복귀...')
            self.status_pub.publish(String(data='BUTTON_PRESSED'))
            self.task_done = True
            if self.target_floor is not None:
                self.current_floor = self.target_floor
            threading.Timer(3.0, self._move_to_home).start()
        else:
            self.get_logger().error('❌ 버튼 이동 실패')
            self.status_pub.publish(String(data='FAILED'))
            self.button_pressed = False

    def _send_trajectory(self, target_joints: list, blocking: bool = True) -> bool:
        """
        /arm_controller/follow_joint_trajectory 액션으로 관절 이동.
        blocking=True: 이동 완료까지 대기.
        Returns True on success.
        """
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

        future = self._arm_client.send_goal_async(goal)

        # 액션 수락 대기
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

        # 실행 결과 대기
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
            self.get_logger().info('✅ 작업 완료. 대기 중...')

    def _set_ready(self):
        self.ready = True
        self.get_logger().info('✅ 버튼 인식 시작!')


# ─── 엔트리포인트 ────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = DirectIKNode()
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
