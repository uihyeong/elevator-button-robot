import math
import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
import cv2
import numpy as np
import threading
import tf2_ros
import tf2_geometry_msgs


# OpenMANIPULATOR-X 링크 길이 (URDF 기준, 단위: m)
L1 = 0.0595                              # joint1→joint2 (z)
L2 = math.sqrt(0.128**2 + 0.024**2)     # joint2→joint3
ALPHA = math.atan2(0.024, 0.128)        # link2 오프셋 보정각
L3 = 0.124                               # joint3→joint4
L4 = 0.126                               # joint4→end_effector

JOINT_LIMITS = [
    (-3.14,  3.14),   # joint1
    (-1.5,   1.5),    # joint2
    (-1.5,   1.4),    # joint3
    (-1.7,   1.97),   # joint4
]


def analytical_ik(X, Y, Z):
    """
    OpenMANIPULATOR-X 해석적 IK
    수평 접근(end_effector가 X축 방향으로 버튼을 향함) 가정
    반환: [j1, j2, j3, j4] (라디안) 또는 None

    FK 규칙 (부호 주의):
      link2 global angle = ALPHA - j2
      link3 global angle = -(j2 + j3)
    """
    j1 = math.atan2(Y, X)

    r = math.sqrt(X**2 + Y**2) - L4   # end_effector 길이 제거
    h = Z - L1                          # joint2 높이 기준

    D = math.sqrt(r**2 + h**2)
    if D > L2 + L3:
        return None  # 도달 불가

    cos_rel = (D**2 - L2**2 - L3**2) / (2 * L2 * L3)
    cos_rel = max(-1.0, min(1.0, cos_rel))
    rel = -math.acos(cos_rel)   # elbow-up (음수)

    j3 = -rel - ALPHA           # = acos(cos_rel) - ALPHA

    phi = math.atan2(h, r)
    psi = math.atan2(L3 * math.sin(rel), L2 + L3 * math.cos(rel))
    j2  = ALPHA - (phi - psi)   # = ALPHA - phi + psi

    j4 = -(j2 + j3)             # end_effector 수평 유지

    joints = [j1, j2, j3, j4]
    for j, (lo, hi) in zip(joints, JOINT_LIMITS):
        if not (lo <= j <= hi):
            return None

    return joints


class AnalyticalIKNode(Node):
    def __init__(self):
        super().__init__('analytical_ik_moveit')
        self.bridge = CvBridge()
        self.model = YOLO('/home/sejong/yolo_dataset_real_v2/runs/train_real_v2/weights/best.pt')
        self.lock = threading.Lock()

        self.depth_image = None
        self.button_pressed = False
        self.target_button = 'up_button'
        self.ready = False
        self.task_done = False

        self.HOME_JOINTS = [
            ('joint1',  0.0),
            ('joint2', -0.9948),  # -57deg
            ('joint3',  0.6981),  # 40deg
            ('joint4',  0.2967),  # 17deg
        ]

        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._action_client = ActionClient(self, MoveGroup, '/move_action')
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)

        self.get_logger().info('해석적 IK 노드 시작!')
        self.get_logger().info(f'목표 버튼: {self.target_button}')
        self._home_timer = self.create_timer(2.0, self._move_to_home_once)

    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

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
                Z_cam = d

                point_cam = PointStamped()
                point_cam.header.frame_id = 'camera_color_optical_frame'
                point_cam.header.stamp = rclpy.time.Time().to_msg()
                point_cam.point.x = X_cam
                point_cam.point.y = Y_cam
                point_cam.point.z = Z_cam

                try:
                    point_base = self.tf_buffer.transform(point_cam, 'open_manipulator_x')
                    X = point_base.point.x
                    Y = point_base.point.y
                    Z = point_base.point.z

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (cx_box, cy_box), 4, color, -1)
                    cv2.putText(frame, f'{cls} {conf:.2f}', (x1, y1 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(frame, f'({X:.2f},{Y:.2f},{Z:.2f})',
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    if cls == self.target_button and conf > 0.7 and not self.button_pressed and self.ready and not self.task_done:
                        self.get_logger().info(f'{cls} 감지! 해석적 IK 시작')
                        self.button_pressed = True
                        threading.Thread(
                            target=self.compute_and_move,
                            args=(X, Y, Z), daemon=True).start()

                except Exception as e:
                    self.get_logger().warn(f'TF 변환 실패: {e}')

        cv2.imshow('Analytical IK', frame)
        cv2.waitKey(1)

    def compute_and_move(self, X, Y, Z):
        self.get_logger().info(f'IK 계산 중... 목표: ({X:.3f}, {Y:.3f}, {Z:.3f})')

        joints = analytical_ik(X, Y, Z)
        if joints is None:
            self.get_logger().error('IK 실패: 도달 불가 또는 관절 범위 초과')
            self.button_pressed = False
            self.status_pub.publish(String(data='FAILED'))
            return

        j1, j2, j3, j4 = joints
        self.get_logger().info(f'IK 성공! j1={j1:.3f} j2={j2:.3f} j3={j3:.3f} j4={j4:.3f}')
        joint_values = [('joint1', j1), ('joint2', j2), ('joint3', j3), ('joint4', j4)]
        self.move_to_joint_target(joint_values, self.button_response_callback)

    def move_to_joint_target(self, joint_values, callback):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('MoveIt2 서버 연결 안됨!')
            self.button_pressed = False
            return

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
        future = self._action_client.send_goal_async(goal)
        future.add_done_callback(callback)

    def button_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('버튼 이동 거부됨')
            self.button_pressed = False
            self.status_pub.publish(String(data='FAILED'))
            return
        self.get_logger().info('버튼 이동 수락됨!')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.button_result_callback)

    def button_result_callback(self, future):
        result = future.result().result
        code = result.error_code.val
        if code == 1:
            self.get_logger().info('✅ 버튼 누르기 성공! 3초 후 복귀...')
            self.status_pub.publish(String(data='BUTTON_PRESSED'))
            self.task_done = True
            threading.Timer(3.0, self.move_to_home).start()
        else:
            self.get_logger().error(f'❌ 버튼 누르기 실패: error_code={code}')
            self.status_pub.publish(String(data='FAILED'))
            self.button_pressed = False

    def _move_to_home_once(self):
        self._home_timer.cancel()
        self.move_to_home()

    def move_to_home(self):
        self.get_logger().info('home 포지션으로 이동 중...')
        self.move_to_joint_target(self.HOME_JOINTS, self.home_response_callback)

    def home_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('home 이동 거부됨')
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.home_result_callback)

    def home_result_callback(self, future):
        result = future.result().result
        code = result.error_code.val
        if code == 1:
            self.get_logger().info('✅ home 도착! 5초 후 인식 시작...')
            self.status_pub.publish(String(data='SUCCESS'))
        else:
            self.get_logger().error(f'❌ home 이동 실패: error_code={code}')
        self.button_pressed = False
        if not self.task_done:
            threading.Timer(5.0, self._set_ready).start()
        else:
            self.get_logger().info('✅ 작업 완료! 대기 중...')

    def _set_ready(self):
        self.ready = True
        self.get_logger().info('✅ 인식 시작!')


def main():
    rclpy.init()
    node = AnalyticalIKNode()
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
