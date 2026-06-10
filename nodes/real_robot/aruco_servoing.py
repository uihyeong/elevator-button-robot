import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Int32, String
import cv2
import numpy as np


class ArucoServoing(Node):
    def __init__(self):
        super().__init__('aruco_servoing')
        self.bridge = CvBridge()
        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 self.image_callback, 10)

        self.cmd_pub              = self.create_publisher(Twist,  '/cmd_vel',         10)
        self.elevator_ready_pub   = self.create_publisher(Bool,   '/elevator_ready',  10)
        self.aligned_ready_pub    = self.create_publisher(Bool,   '/aligned_ready',   10)
        self.target_floor_pub     = self.create_publisher(Int32,  '/target_floor',    10)

        self.create_subscription(Bool,   '/start_pickup',  self._cb_start_pickup,  10)
        self.create_subscription(Bool,   '/pickup_done',   self._cb_pickup_done,   10)
        self.create_subscription(String, '/robot_status',  self._cb_robot_status,  10)
        self.create_subscription(Int32,  '/mission_floor', self._cb_mission_floor, 10)

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        self.camera_matrix = np.array([
            [1367.6949, 0.0,       972.8739],
            [0.0,       1366.4255, 561.2661],
            [0.0,       0.0,       1.0     ]
        ])
        self.dist_coeffs = np.zeros((5, 1))
        self.marker_size = 0.1

        # PD 제어 파라미터
        self.kp_z = 0.3
        self.kp_x = 0.15
        self.kd_x = 0.05
        self.prev_error_x = 0.0
        self.tol_z = 0.03
        self.tol_x = 0.02
        self.max_linear  = 0.15
        self.max_angular = 0.1

        # 블라인드 이동 속도
        self.rotate_angular_speed = 0.2
        self.blind_linear_speed   = 0.1
        self.rotate_duration = (np.pi / 2) / self.rotate_angular_speed

        # ── 실행 시퀀스 ──────────────────────────────────────────────
        # type: 'servo'       → 마커 서보잉 (marker_id, target_z, target_x)
        # type: 'rotate'      → 제자리 회전 (direction: -1=우회전, +1=좌회전)
        # type: 'forward'     → 블라인드 전진 (distance m)
        # type: 'backward'    → 블라인드 후진 (distance m)
        # type: 'wait_status' → /robot_status 특정 값 수신까지 대기 (value)
        # type: 'pub_ready'   → Bool(True) 발행 후 즉시 다음 스텝 (topic)
        # type: 'pub_floor'   → 저장된 층수를 /target_floor로 발행 (arm_elevator 트리거)
        self.sequence = [
            {'type': 'servo',       'marker_id': 0, 'target_z': 0.22, 'target_x': 0.0},
            {'type': 'pub_floor'},
            {'type': 'wait_status', 'value': 'ELEVATOR_ARRIVED'},
            {'type': 'rotate',      'direction': -1},    # 90도 우회전
            {'type': 'forward',     'distance': 0.65},   # 60cm 전진
            {'type': 'rotate',      'direction': +1},    # 90도 좌회전
            {'type': 'forward',     'distance': 1.84},   # 1.8m 전진
            {'type': 'rotate',      'direction': -1},    # 90도 우회전
            {'type': 'servo',       'marker_id': 1, 'target_z': 0.26, 'target_x': 0.0},
            {'type': 'pub_ready',   'topic': '/elevator_ready'},
            {'type': 'wait_status', 'value': 'NUMBER_PRESSED'},
            {'type': 'backward',    'distance': 0.35},
            {'type': 'rotate',      'direction': -1},    # 90도 우회전
            {'type': 'servo',       'marker_id': 2, 'target_z': 0.30, 'target_x': 0.0},
            {'type': 'pub_ready',   'topic': '/aligned_ready'},
        ]

        self.step_index        = 0
        self.state             = 'idle'
        self.current_step      = None
        self.step_start_time   = None
        self._latest_status    = ''
        self._waiting_status   = None
        self._mission_floor    = None
        self._pickup_done_recv = False

        self.get_logger().info('aruco_servoing 대기 중. /start_pickup + /mission_floor 수신 후 시작.')

    # ── 토픽 콜백 ────────────────────────────────────────────────────

    def _cb_mission_floor(self, msg: Int32):
        self._mission_floor = msg.data
        self.get_logger().info(f'/mission_floor 수신: {self._mission_floor}층 저장')
        self._try_start()

    def _cb_start_pickup(self, msg: Bool):
        if not msg.data:
            return
        self.get_logger().info('/start_pickup 수신 → arm_delivery 픽업 시작 대기 중')

    def _cb_pickup_done(self, msg: Bool):
        if not msg.data:
            return
        if self.state != 'idle':
            self.get_logger().warn(f'/pickup_done 무시 (state={self.state})')
            return
        self.get_logger().info('/pickup_done 수신 → 픽업 완료')
        self._pickup_done_recv = True
        self._try_start()

    def _try_start(self):
        if self.state != 'idle':
            return
        if self._mission_floor is None:
            self.get_logger().info('층수 대기 중...')
            return
        if not self._pickup_done_recv:
            self.get_logger().info(f'층수={self._mission_floor} 저장됨. /pickup_done 대기 중...')
            return
        self.get_logger().info(f'층수={self._mission_floor}, 픽업 완료 → 시퀀스 시작')
        self._advance_step()

    def _cb_robot_status(self, msg: String):
        self._latest_status = msg.data
        if self.state == 'wait_status' and msg.data == self._waiting_status:
            self.get_logger().info(f'✅ 상태 확인: {msg.data} → 다음 스텝')
            self._waiting_status = None
            self._advance_step()

    # ── 다음 스텝으로 전환 ────────────────────────────────────────────

    def _advance_step(self):
        if self.step_index >= len(self.sequence):
            self.state = 'done'
            self.get_logger().info('✅ 모든 시퀀스 완료!')
            return

        self.current_step = self.sequence[self.step_index]
        stype = self.current_step['type']
        self.get_logger().info(
            f'▶ Step {self.step_index + 1}/{len(self.sequence)}: {stype} | {self.current_step}')
        self.step_index += 1

        if stype == 'servo':
            self.state = 'servoing'
            self.prev_error_x = 0.0

        elif stype == 'rotate':
            self.state = 'rotating'
            self.step_start_time = self.get_clock().now()

        elif stype == 'forward':
            self.state = 'forward'
            self.step_start_time = self.get_clock().now()
            self.current_step['duration'] = (
                self.current_step['distance'] / self.blind_linear_speed
            )

        elif stype == 'backward':
            self.state = 'backward'
            self.step_start_time = self.get_clock().now()
            self.current_step['duration'] = (
                self.current_step['distance'] / self.blind_linear_speed
            )

        elif stype == 'wait_status':
            self._waiting_status = self.current_step['value']
            self.state = 'wait_status'
            self.get_logger().info(f'⏳ /robot_status="{self._waiting_status}" 대기 중...')

        elif stype == 'pub_floor':
            if self._mission_floor is None:
                self.get_logger().error('층수 미설정! /mission_floor를 먼저 발행하세요.')
                self.state = 'done'
                return
            self.target_floor_pub.publish(Int32(data=self._mission_floor))
            self.get_logger().info(
                f'/target_floor {self._mission_floor} 발행 → arm_elevator UPDOWN_READY 시작')
            self._advance_step()

        elif stype == 'pub_ready':
            topic = self.current_step['topic']
            if topic == '/elevator_ready':
                self.elevator_ready_pub.publish(Bool(data=True))
                self.get_logger().info('/elevator_ready 발행 → arm_elevator 숫자 Phase 시작')
            elif topic == '/aligned_ready':
                self.aligned_ready_pub.publish(Bool(data=True))
                self.get_logger().info('/aligned_ready 발행 → arm_delivery 배달 시작')
            self._advance_step()

    # ── 이미지 콜백 (메인 루프) ───────────────────────────────────────

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        corners, ids, _ = self.detector.detectMarkers(frame)
        twist = Twist()

        if self.state == 'done':
            cv2.putText(frame, "ALL DONE",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        elif self.state == 'idle':
            cv2.putText(frame, "IDLE — /pickup_done 대기 중",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)

        elif self.state == 'wait_status':
            cv2.putText(frame,
                        f"WAIT: '{self._waiting_status}' (현재: '{self._latest_status}')",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

        elif self.state == 'rotating':
            elapsed = (self.get_clock().now() - self.step_start_time).nanoseconds / 1e9
            if elapsed < self.rotate_duration:
                twist.angular.z = self.rotate_angular_speed * self.current_step['direction']
                label = 'RIGHT' if self.current_step['direction'] < 0 else 'LEFT'
                cv2.putText(frame, f"ROTATING {label} 90deg...",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            else:
                self.get_logger().info('✅ 회전 완료')
                self._advance_step()

        elif self.state == 'forward':
            elapsed = (self.get_clock().now() - self.step_start_time).nanoseconds / 1e9
            if elapsed < self.current_step['duration']:
                twist.linear.x = self.blind_linear_speed
                cv2.putText(frame, f"FORWARD {self.current_step['distance']}m...",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 2)
            else:
                self.get_logger().info(f"✅ 전진 완료: {self.current_step['distance']}m")
                self._advance_step()

        elif self.state == 'backward':
            elapsed = (self.get_clock().now() - self.step_start_time).nanoseconds / 1e9
            if elapsed < self.current_step['duration']:
                twist.linear.x = -self.blind_linear_speed
                cv2.putText(frame, f"BACKWARD {self.current_step['distance']}m...",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 165), 2)
            else:
                self.get_logger().info(f"✅ 후진 완료: {self.current_step['distance']}m")
                self._advance_step()

        elif self.state == 'servoing':
            marker_id = self.current_step['marker_id']
            target_z  = self.current_step['target_z']
            target_x  = self.current_step['target_x']

            found = False
            if ids is not None:
                for i, corner in enumerate(corners):
                    if ids[i][0] != marker_id:
                        continue

                    found = True
                    half = 0.015 if marker_id in (0, 1) else 0.05
                    obj_points = np.array([
                        [-half,  half, 0],
                        [ half,  half, 0],
                        [ half, -half, 0],
                        [-half, -half, 0]
                    ], dtype=np.float32)

                    ret, rvec, tvec = cv2.solvePnP(
                        obj_points, corner[0],
                        self.camera_matrix, self.dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )

                    x = tvec[0][0]
                    z = tvec[2][0]
                    error_z = z - target_z
                    error_x = x - target_x

                    self.get_logger().info(
                        f'ID:{marker_id} | Z:{z:.3f}m | X:{x:.3f}m | '
                        f'err_Z:{error_z:.3f} | err_X:{error_x:.3f}')

                    if abs(error_z) < self.tol_z and abs(error_x) < self.tol_x:
                        self.get_logger().info(
                            f'✅ ID={marker_id} 정렬 완료! ({target_z*100:.0f}cm)')
                        self.prev_error_x = 0.0
                        self._advance_step()
                    else:
                        vel_z = self.kp_z * error_z
                        vel_z = max(-self.max_linear, min(self.max_linear, vel_z))

                        d_error_x = error_x - self.prev_error_x
                        vel_x = -(self.kp_x * error_x + self.kd_x * d_error_x)
                        vel_x = max(-self.max_angular, min(self.max_angular, vel_x))
                        self.prev_error_x = error_x

                        twist.linear.x  = vel_z
                        twist.angular.z = vel_x

                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs,
                                      rvec, tvec, 0.05)
                    cv2.putText(frame, f"Target ID:{marker_id} Z:{z:.2f}m X:{x:.2f}m",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(frame, f"err_Z:{error_z:.2f} err_X:{error_x:.2f}",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            if not found:
                self.get_logger().info(f'마커 ID={marker_id} 탐색 중...')
                twist.linear.x  = 0.0
                twist.angular.z = 0.0
                self.prev_error_x = 0.0

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # ── 공통 HUD ──
        step_label  = f"Step {min(self.step_index, len(self.sequence))}/{len(self.sequence)}"
        floor_label = f"Floor: {self._mission_floor}" if self._mission_floor else "Floor: -"
        cv2.putText(frame, f"STATE: {self.state.upper()} | {step_label} | {floor_label}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        self.cmd_pub.publish(twist)
        cv2.imshow("ArUco Servoing", frame)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = ArucoServoing()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()