import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, PositionIKRequest, RobotState
from moveit_msgs.srv import GetPositionIK
import cv2
import numpy as np
import threading
import tf2_ros
import tf2_geometry_msgs


class IsaacSimYoloMoveIt(Node):
    def __init__(self):
        super().__init__('isaac_sim_yolo_moveit')
        self.bridge = CvBridge()
        self.model = YOLO('/home/sejong/yolo_dataset_real_v2/runs/train_real_v2/weights/best.pt')
        self.lock = threading.Lock()

        self.depth_image = None
        self.current_joint_state = None
        self.button_pressed = False
        self.target_button = 'up_button'
        self.ready = False  # home 도착 후 5초 뒤에 인식 시작
        self.task_done = False

        self.HOME_JOINTS = [
            ('joint1',  0.0),
            ('joint2', -0.9948),  # -57deg
            ('joint3',  0.6981),  # 40deg
            ('joint4',  0.2967),  # 17deg
        ]

        # 카메라 파라미터 기본값
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # MoveIt2 Action Client
        self._action_client = ActionClient(self, MoveGroup, '/move_action')

        # compute_ik 서비스 클라이언트
        self._ik_client = self.create_client(GetPositionIK, '/compute_ik')

        # 상태 발행
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        self.sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        self.get_logger().info('Isaac Sim YOLO MoveIt2 시작!')
        self.get_logger().info(f'목표 버튼: {self.target_button}')

        # 시작 시 home 포지션으로 이동
        self._home_timer = self.create_timer(2.0, self._move_to_home_once)

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
        colors = {
            'up_button':   (0, 255, 0),
            'down_button': (0, 0, 255),
        }

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
                    point_base = self.tf_buffer.transform(
                        point_cam, 'open_manipulator_x')
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
                        self.get_logger().info(f'{cls} 감지! IK 계산 시작')
                        self.button_pressed = True
                        threading.Thread(
                            target=self.compute_ik_and_move,
                            args=(X, Y, Z), daemon=True).start()

                except Exception as e:
                    self.get_logger().warn(f'TF 변환 실패: {e}')

        cv2.imshow('Isaac Sim YOLO MoveIt2', frame)
        cv2.waitKey(1)

    def compute_ik_and_move(self, X, Y, Z):
        # compute_ik 서비스 대기
        if not self._ik_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('compute_ik 서비스 없음!')
            self.button_pressed = False
            return

        # IK 요청 생성
        request = GetPositionIK.Request()
        request.ik_request.group_name = 'arm'
        request.ik_request.ik_link_name = 'end_effector_link'
        request.ik_request.timeout.sec = 5

        # 목표 pose 설정
        target_pose = PoseStamped()
        target_pose.header.frame_id = 'open_manipulator_x'
        target_pose.header.stamp = rclpy.time.Time().to_msg()
        target_pose.pose.position.x = X
        target_pose.pose.position.y = Y
        target_pose.pose.position.z = Z
        target_pose.pose.orientation.w = 1.0
        request.ik_request.pose_stamped = target_pose

        # 현재 joint state를 초기값으로
        with self.lock:
            if self.current_joint_state is not None:
                request.ik_request.robot_state.joint_state = self.current_joint_state

        self.get_logger().info(f'IK 계산 중... 목표: ({X:.3f}, {Y:.3f}, {Z:.3f})')
        future = self._ik_client.call_async(request)

        # 스레드에서 future 완료 대기 (최대 5초)
        import time
        timeout = 5.0
        start = time.time()
        while not future.done():
            if time.time() - start > timeout:
                self.get_logger().error('IK 서비스 타임아웃!')
                self.button_pressed = False
                return
            time.sleep(0.05)

        self.ik_result_callback(future)

    def ik_result_callback(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f'IK 서비스 오류: {e}')
            self.button_pressed = False
            return
        error_code = response.error_code.val

        if error_code != 1:
            self.get_logger().error(f'IK 계산 실패: error_code={error_code}')
            self.button_pressed = False
            self.status_pub.publish(String(data='FAILED'))
            return

        # IK 결과에서 joint 값 추출
        joint_state = response.solution.joint_state
        joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        joint_values = []

        for name in joint_names:
            if name in joint_state.name:
                idx = joint_state.name.index(name)
                joint_values.append((name, joint_state.position[idx]))

        self.get_logger().info(f'IK 성공! joint 값: {joint_values}')
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
            threading.Timer(3.0, self.return_to_init).start()
        else:
            self.get_logger().error(f'❌ 버튼 누르기 실패: error_code={code}')
            self.status_pub.publish(String(data='FAILED'))
            self.button_pressed = False

    def _move_to_home_once(self):
        self._home_timer.cancel()
        self.move_to_home()

    def move_to_home(self):
        self.get_logger().info('home 포지션으로 이동 중...')
        self.move_to_joint_target(self.HOME_JOINTS, self.init_response_callback)

    def return_to_init(self):
        self.move_to_home()

    def init_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('복귀 거부됨')
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.init_result_callback)

    def init_result_callback(self, future):
        result = future.result().result
        code = result.error_code.val
        if code == 1:
            self.get_logger().info('✅ home 도착! 5초 후 인식 시작...')
            self.status_pub.publish(String(data='SUCCESS'))
        else:
            self.get_logger().error(f'❌ 복귀 실패: error_code={code}')
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
    node = IsaacSimYoloMoveIt()
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
