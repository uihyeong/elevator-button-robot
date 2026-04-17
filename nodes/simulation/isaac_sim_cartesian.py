import math
import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String, Int32
from cv_bridge import CvBridge
from ultralytics import YOLO
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
import cv2
import numpy as np
import threading
import time
import tf2_ros
import tf2_geometry_msgs

# 시뮬레이션 vs 실제 로봇 차이점
# - 베이스 프레임: open_manipulator_x (sim) / world (real)
# - 카메라 토픽:  /camera/color/...  (sim) / /camera/camera/color/... (real)
# - 뎁스 인코딩: 32FC1 미터 단위    (sim) / 16UC1 mm→/1000 변환       (real)
# - 버튼 오프셋:  없음               (sim) / X - 0.075m                (real)

BASE_FRAME = 'open_manipulator_x'

# ─── 해석적 IK (j4 = -(j2+j3) 수평 유지) ────────────────────────────────
L1    = 0.0595
L2    = math.sqrt(0.128**2 + 0.024**2)
ALPHA = math.atan2(0.024, 0.128)
L3    = 0.124
L4    = 0.126

JOINT_LIMITS = [
    (-3.14,  3.14),
    (-1.5,   1.5),
    (-1.5,   1.4),
    (-1.7,   1.97),
]


def analytical_ik(X, Y, Z):
    """end_effector 수평 유지 (j4 = -(j2+j3)) 해석적 IK. 실패 시 None"""
    j1 = math.atan2(Y, X)
    r  = math.sqrt(X**2 + Y**2) - L4
    h  = Z - L1
    D  = math.sqrt(r**2 + h**2)
    if D > L2 + L3:
        return None
    cos_rel = (D**2 - L2**2 - L3**2) / (2 * L2 * L3)
    cos_rel = max(-1.0, min(1.0, cos_rel))
    rel = math.acos(cos_rel)    # elbow-down: j2가 앞으로 이동 (자연스러운 동작)
    j3  = -rel - ALPHA
    phi = math.atan2(h, r)
    psi = math.atan2(L3 * math.sin(rel), L2 + L3 * math.cos(rel))
    j2  = ALPHA - (phi - psi)
    j4  = -(j2 + j3)
    joints = [j1, j2, j3, j4]
    for j, (lo, hi) in zip(joints, JOINT_LIMITS):
        if not (lo <= j <= hi):
            return None
    return joints


class IsaacSimCartesian(Node):
    def __init__(self):
        super().__init__('isaac_sim_cartesian')
        self.bridge = CvBridge()
        self.model = YOLO('/home/sejong/yolo_dataset_real_v2/runs/train_real_v2/weights/best.pt')
        self.lock = threading.Lock()

        self.depth_image = None
        self.current_joint_state = None
        self.button_pressed = False
        self.target_button = None
        self.ready = False
        self.task_done = False
        self.current_floor = -1
        self.target_floor = None

        self.HOME_JOINTS = [
            ('joint1',  0.0),
            ('joint2', -0.9948),
            ('joint3',  0.6981),
            ('joint4',  0.2967),
        ]

        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._action_client = ActionClient(self, MoveGroup, '/move_action')

        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        self.sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.floor_sub = self.create_subscription(
            Int32, '/target_floor', self.target_floor_callback, 10)

        self.get_logger().info('Isaac Sim Cartesian (2-step IK) 시작!')
        self._home_timer = self.create_timer(2.0, self._move_to_home_once)

    # ─── 콜백 ────────────────────────────────────────────────────────────

    def target_floor_callback(self, msg):
        floor = msg.data
        if floor == self.current_floor:
            self.get_logger().warn(f'목표 층수({floor})가 현재 층수와 같음. 무시.')
            return
        if floor == self.target_floor and (self.button_pressed or self.task_done):
            return
        self.target_floor = floor
        self.target_button = 'up_button' if floor > self.current_floor else 'down_button'
        self.button_pressed = False
        self.task_done = False
        self.get_logger().info(
            f'목표 층수: {floor}층 | 현재: {self.current_floor}층 → {self.target_button} 누르기')

    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def joint_state_callback(self, msg):
        with self.lock:
            self.current_joint_state = msg

    def depth_callback(self, msg):
        with self.lock:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        with self.lock:
            depth = self.depth_image.copy() if self.depth_image is not None else None

        results = self.model(frame, conf=0.5, verbose=False)
        colors = {'up_button': (0, 255, 0), 'down_button': (0, 0, 255)}

        for result in results:
            for box in result.boxes:
                cls = result.names[int(box.cls)]
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = colors.get(cls, (255, 255, 0))
                cx_box = (x1 + x2) // 2
                cy_box = (y1 + y2) // 2

                if depth is None:
                    continue
                region = depth[cy_box-2:cy_box+3, cx_box-2:cx_box+3]
                valid = region[(region > 0) & ~np.isnan(region)]
                if len(valid) == 0:
                    continue
                d = float(np.median(valid))

                X_cam = (cx_box - self.cx) / self.fx * d
                Y_cam = (cy_box - self.cy) / self.fy * d

                point_cam = PointStamped()
                point_cam.header.frame_id = 'camera_color_optical_frame'
                point_cam.header.stamp = rclpy.time.Time().to_msg()
                point_cam.point.x = X_cam
                point_cam.point.y = Y_cam
                point_cam.point.z = d

                try:
                    point_base = self.tf_buffer.transform(point_cam, BASE_FRAME)
                    X = point_base.point.x
                    Y = point_base.point.y
                    Z = point_base.point.z

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (cx_box, cy_box), 4, color, -1)
                    cv2.putText(frame, f'{cls} {conf:.2f}', (x1, y1 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(frame, f'({X:.2f},{Y:.2f},{Z:.2f})',
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    if (cls == self.target_button and conf > 0.7
                            and not self.button_pressed and self.ready and not self.task_done):
                        self.get_logger().info(f'{cls} 감지! 2-step IK 시작')
                        self.button_pressed = True
                        threading.Thread(
                            target=self.move_direct,
                            args=(X - 0.05, Y, Z), daemon=True).start()  # 버튼 5cm 앞에서 멈춤

                except Exception as e:
                    self.get_logger().warn(f'TF 변환 실패: {e}')

        cv2.imshow('Isaac Sim Cartesian', frame)
        cv2.waitKey(1)

    # ─── 2-step IK 이동 ───────────────────────────────────────────────────


    def move_direct(self, target_x, target_y, target_z):
        """해석적 IK로 직접 이동 (j4=-(j2+j3) 수평 유지)"""
        self.get_logger().info(f'목표: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})')

        joints = self.compute_ik(target_x, target_y, target_z)
        if joints is None:
            self.button_pressed = False
            return

        success = self.move_and_wait(joints)
        if success:
            self.get_logger().info('✅ 버튼 누르기 성공! 3초 후 복귀...')
            self.status_pub.publish(String(data='BUTTON_PRESSED'))
            self.task_done = True
            if self.target_floor is not None:
                self.current_floor = self.target_floor
            threading.Timer(3.0, self.return_to_init).start()
        else:
            self.get_logger().error('이동 실패')
            self.button_pressed = False
            self.status_pub.publish(String(data='FAILED'))

    def compute_ik(self, X, Y, Z):
        """해석적 IK → joint 값 반환, 실패 시 None"""
        joints = analytical_ik(X, Y, Z)
        if joints is None:
            self.get_logger().error(f'해석적 IK 실패: ({X:.3f}, {Y:.3f}, {Z:.3f}) 도달 불가')
            return None
        names = ['joint1', 'joint2', 'joint3', 'joint4']
        self.get_logger().info(
            f'IK: j1={joints[0]:.3f} j2={joints[1]:.3f} j3={joints[2]:.3f} j4={joints[3]:.3f}')
        return list(zip(names, joints))

    def move_and_wait(self, joint_values):
        """MoveGroup으로 이동 후 결과 반환 (블로킹)"""
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveIt2 서버 없음!')
            return False

        self.status_pub.publish(String(data='MOVING'))

        goal = MoveGroup.Goal()
        goal.request.group_name = 'arm'
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 10.0
        goal.request.max_velocity_scaling_factor = 0.1
        goal.request.max_acceleration_scaling_factor = 0.1
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True

        constraints = Constraints()
        for name, value in joint_values:
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = value
            jc.tolerance_above = 0.05
            jc.tolerance_below = 0.05
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal.request.goal_constraints.append(constraints)

        # 블로킹으로 결과 대기
        result_container = [None]
        done_event = threading.Event()

        def result_cb(future):
            try:
                result_container[0] = future.result().result
            except Exception:
                pass
            done_event.set()

        def response_cb(future):
            gh = future.result()
            if not gh.accepted:
                done_event.set()
                return
            gh.get_result_async().add_done_callback(result_cb)

        self._action_client.send_goal_async(goal).add_done_callback(response_cb)
        done_event.wait(timeout=20.0)

        result = result_container[0]
        return result is not None and result.error_code.val == 1

    # ─── Home 이동 ────────────────────────────────────────────────────────

    def _move_to_home_once(self):
        self._home_timer.cancel()
        threading.Thread(target=self._home_blocking, daemon=True).start()

    def _home_blocking(self):
        self.get_logger().info('home 포지션으로 이동 중...')
        success = self.move_and_wait(self.HOME_JOINTS)
        if success:
            self.get_logger().info('✅ home 도착! 5초 후 인식 시작...')
            self.status_pub.publish(String(data='SUCCESS'))
        else:
            self.get_logger().error('❌ home 이동 실패')
        self.button_pressed = False
        if not self.task_done:
            threading.Timer(5.0, self._set_ready).start()
        else:
            self.get_logger().info('✅ 작업 완료! 대기 중...')

    def return_to_init(self):
        threading.Thread(target=self._home_blocking, daemon=True).start()

    def _set_ready(self):
        self.ready = True
        self.get_logger().info('✅ 인식 시작!')


def main():
    rclpy.init()
    node = IsaacSimCartesian()
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
