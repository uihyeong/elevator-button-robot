import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import String, Int32
from cv_bridge import CvBridge
from ultralytics import YOLO
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from moveit_msgs.srv import GetPositionIK
import cv2
import numpy as np
import threading
import time
import tf2_ros
import easyocr

NUM_MODEL_PATH = '/home/sejong/yolo_dataset_num/runs/segment/train/weights/best.pt'
OCR_INTERVAL   = 5   # 매 N프레임마다 OCR 실행 (이미지콜백 블로킹 방지)
OCR_CONF_MIN   = 0.5
PRESS_CONF_MIN = 0.7
BUTTON_OFFSET  = 0.075  # 버튼 표면 7.5cm 앞에서 멈춤


class RealRobotNumOCR(Node):
    def __init__(self):
        super().__init__('real_robot_num_ocr')
        self.bridge = CvBridge()

        self.get_logger().info('YOLO 세그멘테이션 모델 로드 중...')
        self.model = YOLO(NUM_MODEL_PATH)

        self.get_logger().info('EasyOCR 초기화 중... (첫 실행 시 모델 다운로드로 오래 걸릴 수 있음)')
        self.ocr = easyocr.Reader(['en'], gpu=False)
        self.get_logger().info('EasyOCR 초기화 완료!')

        self.lock = threading.Lock()
        self.depth_image       = None
        self.current_joint_state = None
        self.button_pressed    = False
        self.ready             = False
        self.task_done         = False
        self.target_floor      = None

        # OCR 결과 캐시: {box_key: number}  (box_key = "x1_y1_x2_y2" 그리드 근사)
        self.ocr_cache  = {}
        self.frame_count = 0

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

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._action_client = ActionClient(self, MoveGroup, '/move_action')
        self._ik_client     = self.create_client(GetPositionIK, '/compute_ik')
        self.status_pub     = self.create_publisher(String, '/robot_status', 10)

        self.sub       = self.create_subscription(
            Image,      '/camera/camera/color/image_raw',        self.image_callback, 10)
        self.depth_sub = self.create_subscription(
            Image,      '/camera/camera/depth/image_rect_raw',   self.depth_callback, 10)
        self.info_sub  = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info',      self.camera_info_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states',                         self.joint_state_callback, 10)
        self.floor_sub = self.create_subscription(
            Int32,      '/target_floor',                         self.target_floor_callback, 10)

        self.get_logger().info('숫자 버튼 OCR 노드 시작! /target_floor 대기 중...')
        self._home_timer = self.create_timer(2.0, self._move_to_home_once)

    # ── 토픽 콜백 ─────────────────────────────────────────────────────────────

    def target_floor_callback(self, msg):
        floor = msg.data
        if floor == self.target_floor and (self.button_pressed or self.task_done):
            return
        self.target_floor  = floor
        self.button_pressed = False
        self.task_done      = False
        self.ocr_cache.clear()
        self.get_logger().info(f'목표 층수: {floor}층')

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
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            self.depth_image = raw.astype(np.float32) / 1000.0  # mm → m

    # ── OCR ───────────────────────────────────────────────────────────────────

    def _box_key(self, x1, y1, x2, y2, grid=20):
        """bounding box를 grid 단위로 양자화해서 캐시 키 생성"""
        return f'{x1//grid}_{y1//grid}_{x2//grid}_{y2//grid}'

    def _read_number(self, crop):
        """버튼 crop에서 숫자 인식. 실패 시 None 반환."""
        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return None

        # 최소 64px 보장 (작은 버튼도 OCR 가능하게)
        scale = max(64 / max(h, w), 1.0)
        resized = cv2.resize(crop, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)

        # 대비 강화 (CLAHE)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        results = self.ocr.readtext(
            enhanced,
            allowlist='0123456789Bb',
            detail=1,
            paragraph=False,
        )

        best_conf = 0.0
        best_num  = None
        for (_, text, conf) in results:
            text = text.strip().upper()
            if text.isdigit() and conf > best_conf:
                best_conf = conf
                best_num  = int(text)
            elif text.startswith('B') and text[1:].isdigit() and conf > best_conf:
                # B1, B2 → 지하 -1, -2
                best_conf = conf
                best_num  = -int(text[1:])

        return best_num

    # ── 이미지 처리 메인 루프 ─────────────────────────────────────────────────

    def image_callback(self, msg):
        if self.target_floor is None:
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

                # 패딩
                pad = 5
                cx1 = max(0, x1 - pad)
                cy1 = max(0, y1 - pad)
                cx2 = min(frame.shape[1], x2 + pad)
                cy2 = min(frame.shape[0], y2 + pad)
                crop = frame[cy1:cy2, cx1:cx2]

                key = self._box_key(x1, y1, x2, y2)

                # OCR: N프레임마다 실행하고 결과 캐시
                if run_ocr or key not in self.ocr_cache:
                    number = self._read_number(crop)
                    self.ocr_cache[key] = number
                else:
                    number = self.ocr_cache.get(key)

                label = str(number) if number is not None else '?'
                matched = (number == self.target_floor)
                color = (0, 255, 0) if matched else (180, 180, 180)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label} {conf:.2f}',
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if matched and conf > PRESS_CONF_MIN \
                        and not self.button_pressed and self.ready and not self.task_done:
                    self._trigger_press(frame, depth, x1, y1, x2, y2)

        cv2.putText(frame, f'Target: {self.target_floor}F  ready={self.ready}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow('Num Button OCR', frame)
        cv2.waitKey(1)

    def _trigger_press(self, frame, depth, x1, y1, x2, y2):
        """목표 버튼 발견 → 깊이 추출 → TF → IK 스레드 시작"""
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
        Z_cam = d

        point_cam = PointStamped()
        point_cam.header.frame_id = 'camera_color_optical_frame'
        point_cam.header.stamp    = rclpy.time.Time().to_msg()
        point_cam.point.x = X_cam
        point_cam.point.y = Y_cam
        point_cam.point.z = Z_cam

        try:
            point_base = self.tf_buffer.transform(point_cam, 'world')
            X = point_base.point.x
            Y = point_base.point.y
            Z = point_base.point.z
        except Exception as e:
            self.get_logger().warn(f'TF 변환 실패: {e}')
            return

        self.get_logger().info(
            f'{self.target_floor}층 버튼 감지! 위치: ({X:.3f}, {Y:.3f}, {Z:.3f}) → IK 시작')
        self.button_pressed = True
        threading.Thread(
            target=self.compute_ik_and_move,
            args=(X - BUTTON_OFFSET, Y, Z),
            daemon=True,
        ).start()

    # ── IK / MoveIt2 (기존 노드와 동일) ──────────────────────────────────────

    def compute_ik_and_move(self, X, Y, Z):
        if not self._ik_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('compute_ik 서비스 없음!')
            self.button_pressed = False
            return

        request = GetPositionIK.Request()
        request.ik_request.group_name   = 'arm'
        request.ik_request.ik_link_name = 'end_effector_link'
        request.ik_request.timeout.sec  = 5

        target_pose = PoseStamped()
        target_pose.header.frame_id      = 'world'
        target_pose.header.stamp         = rclpy.time.Time().to_msg()
        target_pose.pose.position.x      = X
        target_pose.pose.position.y      = Y
        target_pose.pose.position.z      = Z
        target_pose.pose.orientation.w   = 1.0
        request.ik_request.pose_stamped  = target_pose

        with self.lock:
            if self.current_joint_state is not None:
                request.ik_request.robot_state.joint_state = self.current_joint_state

        future = self._ik_client.call_async(request)
        start  = time.time()
        while not future.done():
            if time.time() - start > 5.0:
                self.get_logger().error('IK 서비스 타임아웃!')
                self.button_pressed = False
                return
            time.sleep(0.05)

        self._on_ik_result(future)

    def _on_ik_result(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f'IK 서비스 오류: {e}')
            self.button_pressed = False
            return

        if response.error_code.val != 1:
            self.get_logger().error(f'IK 실패: error_code={response.error_code.val}')
            self.button_pressed = False
            self.status_pub.publish(String(data='FAILED'))
            return

        joint_state  = response.solution.joint_state
        joint_names  = ['joint1', 'joint2', 'joint3', 'joint4']
        joint_values = [
            (name, joint_state.position[joint_state.name.index(name)])
            for name in joint_names if name in joint_state.name
        ]

        self.get_logger().info(f'IK 성공! joints: {joint_values}')
        self._move_to_joints(joint_values, self._on_press_response)

    def _move_to_joints(self, joint_values, callback):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('MoveIt2 서버 없음!')
            self.button_pressed = False
            return

        self.status_pub.publish(String(data='MOVING'))

        goal = MoveGroup.Goal()
        goal.request.group_name                    = 'arm'
        goal.request.num_planning_attempts         = 10
        goal.request.allowed_planning_time         = 10.0
        goal.request.max_velocity_scaling_factor     = 0.1
        goal.request.max_acceleration_scaling_factor = 0.1
        goal.planning_options.plan_only = False
        goal.planning_options.replan    = True

        constraints = Constraints()
        for name, value in joint_values:
            jc = JointConstraint()
            jc.joint_name      = name
            jc.position        = value
            jc.tolerance_above = 0.05
            jc.tolerance_below = 0.05
            jc.weight          = 1.0
            constraints.joint_constraints.append(jc)

        goal.request.goal_constraints.append(constraints)
        future = self._action_client.send_goal_async(goal)
        future.add_done_callback(callback)

    def _on_press_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('이동 거부됨')
            self.button_pressed = False
            self.status_pub.publish(String(data='FAILED'))
            return
        goal_handle.get_result_async().add_done_callback(self._on_press_result)

    def _on_press_result(self, future):
        code = future.result().result.error_code.val
        if code == 1:
            self.get_logger().info(f'✅ {self.target_floor}층 버튼 누르기 성공! 3초 후 복귀...')
            self.status_pub.publish(String(data='BUTTON_PRESSED'))
            self.task_done = True
            threading.Timer(3.0, self._return_home).start()
        else:
            self.get_logger().error(f'❌ 버튼 누르기 실패: error_code={code}')
            self.status_pub.publish(String(data='FAILED'))
            self.button_pressed = False

    # ── Home 이동 ─────────────────────────────────────────────────────────────

    def _move_to_home_once(self):
        self._home_timer.cancel()
        self._move_to_home()

    def _move_to_home(self):
        self.get_logger().info('home 포지션으로 이동 중...')
        self._move_to_joints(self.HOME_JOINTS, self._on_home_response)

    def _return_home(self):
        self._move_to_home()

    def _on_home_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('home 이동 거부됨')
            return
        goal_handle.get_result_async().add_done_callback(self._on_home_result)

    def _on_home_result(self, future):
        code = future.result().result.error_code.val
        if code == 1:
            self.get_logger().info('✅ home 도착! 5초 후 인식 시작...')
            self.status_pub.publish(String(data='SUCCESS'))
        else:
            self.get_logger().error(f'❌ home 이동 실패: error_code={code}')
        self.button_pressed = False
        if not self.task_done:
            threading.Timer(5.0, self._set_ready).start()
        else:
            self.get_logger().info('✅ 작업 완료! /target_floor 대기 중...')
            self.task_done = False

    def _set_ready(self):
        self.ready = True
        self.get_logger().info('✅ 버튼 인식 시작!')


def main():
    rclpy.init()
    node = RealRobotNumOCR()
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
