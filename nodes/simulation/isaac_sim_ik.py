"""
Isaac Sim - OpenMANIPULATOR-X 4DOF 해석적 IK 노드

FK 규칙 (검증 완료):
  phi2 = ALPHA - j2          # link2 global angle
  phi3 = -(j2 + j3)          # link3 global angle
  phi4 = -(j2 + j3 + j4)     # end effector pitch

IK:
  j1       = atan2(Y, X)
  r_wrist  = sqrt(X²+Y²) - L4*cos(phi_ee)   # joint4 목표 수평거리
  h_wrist  = Z - L1 - L4*sin(phi_ee)         # joint4 목표 높이
  → 2-link IK (L2, L3) → j2, j3
  j4       = -(j2+j3) - phi_ee
"""

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

# ── 링크 파라미터 ──────────────────────────────────────────────────────────
L1    = 0.0595
L2    = math.sqrt(0.128**2 + 0.024**2)
ALPHA = math.atan2(0.024, 0.128)
L3    = 0.124
L4    = 0.126

JOINT_LIMITS = [
    (-3.14,  3.14),   # joint1
    (-1.5,   1.5),    # joint2
    (-1.5,   1.4),    # joint3
    (-1.7,   1.97),   # joint4
]

HOME = [0.0, -0.9948, 0.6981, 0.2967]

# ── IK ────────────────────────────────────────────────────────────────────
def solve_ik(X, Y, Z, phi_ee=0.0):
    """
    4DOF 해석적 IK.
    phi_ee: end effector pitch (rad). 0 = 수평 접근.
    두 해 중 홈 포지션과 가장 가까운 것을 반환. 실패 시 None.
    """
    j1 = math.atan2(Y, X)

    # joint4 목표 위치 (end effector 길이 제거)
    r_w = math.sqrt(X**2 + Y**2) - L4 * math.cos(phi_ee)
    h_w = Z - L1 - L4 * math.sin(phi_ee)
    D   = math.sqrt(r_w**2 + h_w**2)

    if D > L2 + L3:
        return None  # 도달 불가

    cos_beta = (D**2 - L2**2 - L3**2) / (2 * L2 * L3)
    cos_beta = max(-1.0, min(1.0, cos_beta))
    phi      = math.atan2(h_w, r_w)

    solutions = []
    for sign in [1, -1]:  # elbow-down, elbow-up
        beta  = sign * math.acos(cos_beta)   # link2→link3 상대각
        j3    = -beta - ALPHA
        psi   = math.atan2(L3 * math.sin(beta), L2 + L3 * math.cos(beta))
        j2    = ALPHA - (phi - psi)
        j4    = -(j2 + j3) - phi_ee

        joints = [j1, j2, j3, j4]
        if all(lo <= j <= hi for j, (lo, hi) in zip(joints, JOINT_LIMITS)):
            solutions.append(joints)

    if not solutions:
        return None

    # 홈과 가장 가까운 해 선택 (joint2~4 기준)
    def dist_from_home(sol):
        return sum((sol[i] - HOME[i])**2 for i in range(1, 4))

    return min(solutions, key=dist_from_home)


# ── ROS2 노드 ──────────────────────────────────────────────────────────────
class IsaacSimIK(Node):
    def __init__(self):
        super().__init__('isaac_sim_ik')
        self.bridge = CvBridge()
        self.model  = YOLO('/home/sejong/yolo_dataset_real_v2/runs/train_real_v2/weights/best.pt')
        self.lock   = threading.Lock()

        self.depth_image         = None
        self.current_joint_state = None
        self.button_pressed      = False
        self.target_button       = None
        self.ready               = False
        self.task_done           = False
        self.current_floor       = -1
        self.target_floor        = None

        self.fx, self.fy = 615.0, 615.0
        self.cx, self.cy = 320.0, 240.0

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._move_client = ActionClient(self, MoveGroup, '/move_action')
        self.status_pub   = self.create_publisher(String, '/robot_status', 10)

        self.create_subscription(Image,    '/camera/color/image_raw',        self.image_cb,   10)
        self.create_subscription(Image,    '/camera/depth/image_rect_raw',   self.depth_cb,   10)
        self.create_subscription(CameraInfo,'/camera/color/camera_info',     self.info_cb,    10)
        self.create_subscription(JointState,'/joint_states',                 self.joint_cb,   10)
        self.create_subscription(Int32,    '/target_floor',                  self.floor_cb,   10)

        self.get_logger().info('Isaac Sim IK 노드 시작!')
        self._startup_timer = self.create_timer(2.0, self._home_once)

    # ── 콜백 ──────────────────────────────────────────────────────────────

    def floor_cb(self, msg):
        floor = msg.data
        if floor == self.current_floor:
            return
        if floor == self.target_floor and (self.button_pressed or self.task_done):
            return
        self.target_floor  = floor
        self.target_button = 'up_button' if floor > self.current_floor else 'down_button'
        self.button_pressed = False
        self.task_done      = False
        self.get_logger().info(f'목표 층수: {floor}층 → {self.target_button}')

    def info_cb(self, msg):
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]

    def joint_cb(self, msg):
        with self.lock:
            self.current_joint_state = msg

    def depth_cb(self, msg):
        with self.lock:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, '32FC1')

    def image_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        with self.lock:
            depth = self.depth_image.copy() if self.depth_image is not None else None

        results = self.model(frame, conf=0.5, verbose=False)
        colors  = {'up_button': (0, 255, 0), 'down_button': (0, 0, 255)}

        for r in results:
            for box in r.boxes:
                cls  = r.names[int(box.cls)]
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx_b = (x1 + x2) // 2
                cy_b = (y1 + y2) // 2

                if depth is None:
                    continue
                region = depth[cy_b-2:cy_b+3, cx_b-2:cx_b+3]
                valid  = region[(region > 0) & ~np.isnan(region)]
                if len(valid) == 0:
                    continue
                d = float(np.median(valid))

                pt = PointStamped()
                pt.header.frame_id = 'camera_color_optical_frame'
                pt.header.stamp    = rclpy.time.Time().to_msg()
                pt.point.x = (cx_b - self.cx) / self.fx * d
                pt.point.y = (cy_b - self.cy) / self.fy * d
                pt.point.z = d

                try:
                    pb = self.tf_buffer.transform(pt, 'open_manipulator_x')
                    X, Y, Z = pb.point.x, pb.point.y, pb.point.z

                    cv2.rectangle(frame, (x1,y1), (x2,y2), colors.get(cls,(255,255,0)), 2)
                    cv2.putText(frame, f'{cls} {conf:.2f}', (x1, y1-25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors.get(cls,(255,255,0)), 2)
                    cv2.putText(frame, f'({X:.2f},{Y:.2f},{Z:.2f})', (x1, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.get(cls,(255,255,0)), 1)

                    if (cls == self.target_button and conf > 0.7
                            and not self.button_pressed and self.ready and not self.task_done):
                        self.button_pressed = True
                        self.get_logger().info(f'{cls} 감지! IK 계산 시작 → ({X:.3f},{Y:.3f},{Z:.3f})')
                        threading.Thread(target=self.press_button,
                                         args=(X, Y, Z), daemon=True).start()
                except Exception as e:
                    self.get_logger().warn(f'TF 실패: {e}')

        cv2.imshow('Isaac Sim IK', frame)
        cv2.waitKey(1)

    # ── 버튼 누르기 ────────────────────────────────────────────────────────

    def press_button(self, X, Y, Z):
        joints = solve_ik(X, Y, Z, phi_ee=0.0)
        if joints is None:
            self.get_logger().error(f'IK 실패: ({X:.3f},{Y:.3f},{Z:.3f}) 도달 불가')
            self.button_pressed = False
            self.status_pub.publish(String(data='FAILED'))
            return

        j1, j2, j3, j4 = joints
        self.get_logger().info(
            f'IK 성공 → j1={math.degrees(j1):.1f}° j2={math.degrees(j2):.1f}° '
            f'j3={math.degrees(j3):.1f}° j4={math.degrees(j4):.1f}°')

        ok = self.move_joints([('joint1',j1),('joint2',j2),('joint3',j3),('joint4',j4)])
        if ok:
            self.get_logger().info('✅ 버튼 누르기 성공! 3초 후 복귀...')
            self.status_pub.publish(String(data='BUTTON_PRESSED'))
            self.task_done = True
            if self.target_floor is not None:
                self.current_floor = self.target_floor
            threading.Timer(3.0, self.go_home).start()
        else:
            self.get_logger().error('❌ 이동 실패')
            self.button_pressed = False
            self.status_pub.publish(String(data='FAILED'))

    # ── MoveGroup 이동 (블로킹) ────────────────────────────────────────────

    def move_joints(self, joint_values):
        if not self._move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveIt2 서버 없음!')
            return False

        self.status_pub.publish(String(data='MOVING'))

        goal = MoveGroup.Goal()
        goal.request.group_name                    = 'arm'
        goal.request.num_planning_attempts         = 10
        goal.request.allowed_planning_time         = 10.0
        goal.request.max_velocity_scaling_factor   = 0.1
        goal.request.max_acceleration_scaling_factor = 0.1
        goal.planning_options.plan_only            = False
        goal.planning_options.replan               = True

        c = Constraints()
        for name, val in joint_values:
            jc = JointConstraint()
            jc.joint_name    = name
            jc.position      = val
            jc.tolerance_above = 0.05
            jc.tolerance_below = 0.05
            jc.weight        = 1.0
            c.joint_constraints.append(jc)
        goal.request.goal_constraints.append(c)

        done   = threading.Event()
        result = [None]

        def on_result(f):
            try: result[0] = f.result().result
            except: pass
            done.set()

        def on_goal(f):
            gh = f.result()
            if not gh.accepted:
                done.set()
                return
            gh.get_result_async().add_done_callback(on_result)

        self._move_client.send_goal_async(goal).add_done_callback(on_goal)
        done.wait(timeout=20.0)
        return result[0] is not None and result[0].error_code.val == 1

    # ── Home ───────────────────────────────────────────────────────────────

    def _home_once(self):
        self._startup_timer.cancel()
        threading.Thread(target=self._go_home_thread, daemon=True).start()

    def go_home(self):
        threading.Thread(target=self._go_home_thread, daemon=True).start()

    def _go_home_thread(self):
        names = ['joint1','joint2','joint3','joint4']
        ok = self.move_joints(list(zip(names, HOME)))
        if ok:
            self.get_logger().info('✅ home 도착! 5초 후 인식 시작...')
        else:
            self.get_logger().error('❌ home 이동 실패')
        self.button_pressed = False
        if not self.task_done:
            threading.Timer(5.0, self._set_ready).start()
        else:
            self.get_logger().info('작업 완료.')

    def _set_ready(self):
        self.ready = True
        self.get_logger().info('✅ 인식 시작!')


def main():
    rclpy.init()
    node = IsaacSimIK()
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
