import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PointStamped, Pose
from std_msgs.msg import String, Int32
from cv_bridge import CvBridge
from ultralytics import YOLO
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint
from moveit_msgs.srv import GetCartesianPath
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
            ('joint2', -0.9948),  # -57deg
            ('joint3',  0.6981),  # +40deg
            ('joint4',  0.2967),  # +17deg
        ]

        # 카메라 파라미터 기본값
        self.fx = 615.0
        self.fy = 615.0
        self.cx = 320.0
        self.cy = 240.0

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # MoveGroup 액션 (home 이동용)
        self._move_client = ActionClient(self, MoveGroup, '/move_action')

        # Cartesian path 서비스
        self._cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')

        # Cartesian path 실행 액션
        self._execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')

        # 상태 발행
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        # 시뮬레이션 카메라 토픽
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

        self.get_logger().info('Isaac Sim Cartesian Path 노드 시작!')
        self.get_logger().info(f'현재 층수: {self.current_floor}층 | /target_floor 대기 중...')

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
            # 시뮬레이션: 32FC1 (이미 미터 단위)
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
                        self.get_logger().info(f'{cls} 감지! Cartesian path 시작')
                        self.button_pressed = True
                        threading.Thread(
                            target=self.move_cartesian,
                            args=(X, Y, Z), daemon=True).start()  # 시뮬레이션: 오프셋 없음

                except Exception as e:
                    self.get_logger().warn(f'TF 변환 실패: {e}')

        cv2.imshow('Isaac Sim Cartesian', frame)
        cv2.waitKey(1)

    # ─── Cartesian Path ───────────────────────────────────────────────────

    def get_current_eef_pose(self):
        """현재 end_effector_link 위치를 BASE_FRAME으로 반환"""
        try:
            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME, 'end_effector_link', rclpy.time.Time())
            t = transform.transform.translation
            r = transform.transform.rotation
            return t.x, t.y, t.z, r
        except Exception as e:
            self.get_logger().warn(f'EEF TF 조회 실패: {e}')
            return None

    def move_cartesian(self, target_x, target_y, target_z):
        """Z축 이동 후 X축 이동 (2단계 Cartesian path)"""
        if not self._cartesian_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('compute_cartesian_path 서비스 없음!')
            self.button_pressed = False
            return

        result = self.get_current_eef_pose()
        if result is None:
            self.button_pressed = False
            return
        cur_x, cur_y, cur_z, cur_rot = result
        self.get_logger().info(
            f'현재 EEF: ({cur_x:.3f}, {cur_y:.3f}, {cur_z:.3f}) '
            f'→ 목표: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})')

        # 웨이포인트 1: z축 먼저
        waypoint_z = Pose()
        waypoint_z.position.x = cur_x
        waypoint_z.position.y = cur_y
        waypoint_z.position.z = target_z
        waypoint_z.orientation.w = 1.0

        # 웨이포인트 2: x축 이동 (버튼 접근)
        waypoint_x = Pose()
        waypoint_x.position.x = target_x
        waypoint_x.position.y = target_y
        waypoint_x.position.z = target_z
        waypoint_x.orientation.w = 1.0

        req = GetCartesianPath.Request()
        req.header.frame_id = BASE_FRAME
        req.header.stamp = rclpy.time.Time().to_msg()
        req.group_name = 'arm'
        req.link_name = 'end_effector_link'
        req.waypoints = [waypoint_z, waypoint_x]
        req.max_step = 0.01        # 1cm 간격 보간
        req.jump_threshold = 0.0   # jump 체크 비활성화
        req.avoid_collisions = True

        with self.lock:
            if self.current_joint_state is not None:
                req.start_state.joint_state = self.current_joint_state

        self.get_logger().info('Cartesian path 계산 중...')
        future = self._cartesian_client.call_async(req)

        start = time.time()
        while not future.done():
            if time.time() - start > 10.0:
                self.get_logger().error('Cartesian path 서비스 타임아웃!')
                self.button_pressed = False
                return
            time.sleep(0.05)

        response = future.result()
        fraction = response.fraction

        if fraction < 0.9:
            self.get_logger().error(f'Cartesian path 불완전: {fraction:.1%} 만 계획됨')
            self.button_pressed = False
            self.status_pub.publish(String(data='FAILED'))
            return

        self.get_logger().info(f'Cartesian path 성공 ({fraction:.1%}) → 실행')
        self.execute_trajectory(response.solution)

    def execute_trajectory(self, trajectory):
        if not self._execute_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('execute_trajectory 서버 없음!')
            self.button_pressed = False
            return

        self.status_pub.publish(String(data='MOVING'))

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        future = self._execute_client.send_goal_async(goal)
        future.add_done_callback(self.execute_response_callback)

    def execute_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('궤적 실행 거부됨')
            self.button_pressed = False
            self.status_pub.publish(String(data='FAILED'))
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.execute_result_callback)

    def execute_result_callback(self, future):
        result = future.result().result
        code = result.error_code.val
        if code == 1:
            self.get_logger().info('✅ 버튼 누르기 성공! 3초 후 복귀...')
            self.status_pub.publish(String(data='BUTTON_PRESSED'))
            self.task_done = True
            if self.target_floor is not None:
                self.current_floor = self.target_floor
            threading.Timer(3.0, self.return_to_init).start()
        else:
            self.get_logger().error(f'❌ 궤적 실행 실패: error_code={code}')
            self.status_pub.publish(String(data='FAILED'))
            self.button_pressed = False

    # ─── Home 이동 ────────────────────────────────────────────────────────

    def _move_to_home_once(self):
        self._home_timer.cancel()
        self.move_to_home()

    def move_to_home(self):
        self.get_logger().info('home 포지션으로 이동 중...')
        if not self._move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('MoveIt2 서버 연결 안됨!')
            return

        goal = MoveGroup.Goal()
        goal.request.group_name = 'arm'
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 10.0
        goal.request.max_velocity_scaling_factor = 0.1
        goal.request.max_acceleration_scaling_factor = 0.1
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True

        constraints = Constraints()
        for name, value in self.HOME_JOINTS:
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = value
            jc.tolerance_above = 0.05
            jc.tolerance_below = 0.05
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        goal.request.goal_constraints.append(constraints)
        future = self._move_client.send_goal_async(goal)
        future.add_done_callback(self.home_response_callback)

    def return_to_init(self):
        self.move_to_home()

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
