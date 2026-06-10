"""
프레시백 회수 노드 (arm_recover.py).

회수 시퀀스 (8스텝):
  1. 엘리베이터 홈 (배달 마지막 자세, 그리퍼 닫힌 상태)
  2. 오른쪽 확인 (TABLE_LOOK)
  3. YOLO 박스 인식 대기
  4. 고리에 끼우기 (HOOK_JOINTS — 조인트값 직접 입력)
  5. joint4 올리기 (고개 들어올려 백 걸기)
  6. 홈 복귀, joint4 올린 상태 유지
  7. joint4만 내리기
  8. 엘리베이터 홈 복귀 → 주행 재개

상태 전이:
  IDLE → /start_recover → RECOVER → /recover_done → IDLE

실행:
  ros2 launch open_manipulator_x_bringup hardware.launch.py
  ros2 launch realsense2_camera rs_launch.py
  python3 nodes/real_robot/arm_recover.py
  ros2 topic pub --once /start_recover std_msgs/Bool "{data: true}"
"""

import math
import threading
import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


JOINT_LIMITS = [
    (-math.pi, math.pi),
    (-1.5,     1.5),
    (-1.5,     1.4),
    (-1.7,     1.97),
]

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4']

# ─── 관절 상수 ────────────────────────────────────────────────────────────────

HOME_JOINTS          = [3.141,  -1.3963,  1.2217,  0.5236]
TABLE_LOOK_JOINTS    = [1.571,  -1.3963,  1.2217,  0.5236]
# joint1 을 +3.140(=+pi 쪽)으로 둔다. -3.140 과 물리적으로 동일한 자세지만,
# HOME 계열(+3.141)과 같은 부호 쪽에 둬서 마지막 스텝의 +pi/-pi wrap(한 바퀴 회전)을 제거.
ELEVATOR_HOME_JOINTS = [3.1400, -1.9190,  1.2701,  0.7240]

HOOK_JOINTS      = [1.571, 0.468, -0.331, -0.206]

# joint4만 올린 상태 (step 5): HOOK 위치에서 joint4만 변경
HOOK_LIFT_JOINTS  = [HOOK_JOINTS[0], HOOK_JOINTS[1], HOOK_JOINTS[2], -0.600]
HOOK_HOVER_JOINTS = [1.571, 0.405, -0.204, -0.600]

# joint4 올린 채로 홈 이동 (step 7): HOME[0:3] + 올린 joint4
HOME_HOOK_JOINTS  = [HOME_JOINTS[0], HOME_JOINTS[1], HOME_JOINTS[2], -0.600]

# joint4만 내리기 (step 8): HOME[0:3] + 내린 joint4
HOME_DOWN_JOINTS  = [HOME_JOINTS[0], HOME_JOINTS[1], HOME_JOINTS[2],  0.300]

MOVE_SPEED   = 0.4
MIN_DURATION = 2.0
STEP_DELAY   = 1.5
# 단일 관절 1회 이동 안전 상한(rad). 이보다 크면 위험 동작으로 보고 차단(한 바퀴 회전 방지).
# 180°(π) 정상 이동은 허용하고 360°(2π) 회전만 막도록 4.5 로 통일(3 노드 공통).
MAX_JOINT_STEP = 4.5

# ─── 시퀀스 정의 ─────────────────────────────────────────────────────────────

RECOVER_STEPS = [
    ('엘리베이터 홈 (시작 자세)',         ELEVATOR_HOME_JOINTS),
    ('오른쪽 확인',                       TABLE_LOOK_JOINTS),
    ('고리에 끼우기',                     HOOK_JOINTS),
    ('joint4 올리기 (백 들어올리기)',      HOOK_LIFT_JOINTS),
    ('살짝 호버',                         HOOK_HOVER_JOINTS),
    ('홈 복귀 (joint4 올린 상태 유지)',   HOME_HOOK_JOINTS),
    ('joint4 내리기',                     HOME_DOWN_JOINTS),
    ('엘리베이터 홈 복귀',                ELEVATOR_HOME_JOINTS),
]

# ─── 상태 상수 ────────────────────────────────────────────────────────────────

IDLE    = 'IDLE'
RECOVER = 'RECOVER'
DONE    = 'DONE'

def _shortest_path(target, current):
    diff = (target - current + math.pi) % (2 * math.pi) - math.pi
    return current + diff


def make_trajectory(target_joints, current_joints):
    target_joints = [_shortest_path(t, c) for t, c in zip(target_joints, current_joints)]
    # 관절 한계로 클램프 (_shortest_path 는 한계를 무시하므로 필수)
    target_joints = [max(lo, min(hi, t))
                     for t, (lo, hi) in zip(target_joints, JOINT_LIMITS)]
    max_disp = max(abs(t - c) for t, c in zip(target_joints, current_joints))
    # 과대 이동(한 바퀴 회전 등)이면 None 반환 → 호출부에서 중단
    if max_disp > MAX_JOINT_STEP:
        return None, max_disp
    duration = max(max_disp / MOVE_SPEED, MIN_DURATION)
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


# ─── 노드 ────────────────────────────────────────────────────────────────────

class ArmRecoverNode(Node):

    def __init__(self):
        super().__init__('arm_recover')
        self.lock           = threading.Lock()
        self.current_joints = None
        self.state          = IDLE

        self._arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')

        self.status_pub  = self.create_publisher(String, '/robot_status',  10)
        self.recover_pub = self.create_publisher(Bool,   '/recover_done',  10)

        self.create_subscription(JointState, '/joint_states',    self._cb_joints,         10)
        self.create_subscription(Bool,       '/start_recover',   self._cb_start_recover,  10)

        self._current_step_en = 'Waiting'

        self.get_logger().info('arm_recover 노드 시작. /start_recover 대기 중...')
        self._home_timer = self.create_timer(2.0, self._init_home)

    # ─── 콜백 ────────────────────────────────────────────────────────────────

    def _cb_joints(self, msg):
        with self.lock:
            self.current_joints = msg

    def _cb_start_recover(self, msg: Bool):
        if not msg.data:
            return
        if self.state != IDLE:
            self.get_logger().warn(f'작업 중 ({self.state}). /start_recover 무시.')
            return
        self.get_logger().info('/start_recover 수신 → 회수 시작')
        self.state = RECOVER
        threading.Thread(target=self._run_recover_flow, daemon=True).start()

    # ─── 회수 흐름 ───────────────────────────────────────────────────────────

    def _run_recover_flow(self):
        self.get_logger().info('회수 시퀀스 시작')
        ok = self._run_sequence(RECOVER_STEPS, '회수')
        if not ok:
            self.get_logger().error('회수 실패')
            self.status_pub.publish(String(data='FAILED'))
            self.state = IDLE
            return

        self.get_logger().info('✅ 회수 완료')
        self.status_pub.publish(String(data='RECOVER_DONE'))
        self.recover_pub.publish(Bool(data=True))
        self.state = IDLE
        self.get_logger().info('✅ /start_recover 대기 중...')

    # ─── 시퀀스 실행 ─────────────────────────────────────────────────────────

    def _run_sequence(self, steps, name) -> bool:
        self.get_logger().info(f'{name} 시퀀스 시작 ({len(steps)}스텝)')
        for i, (label, joints) in enumerate(steps):
            self.get_logger().info(f'[{i+1}/{len(steps)}] {label}')
            self._current_step_en = f'[{i+1}/{len(steps)}] {label}'
            time.sleep(STEP_DELAY)

            if joints is not None:
                if not self.move_to_joints(joints, label):
                    self.get_logger().error(f'{label} 실패')
                    return False

        self._current_step_en = f'{name} Done'
        self.get_logger().info(f'{name} 시퀀스 완료')
        return True

    # ─── 팔 이동 ─────────────────────────────────────────────────────────────

    def move_to_joints(self, joints, label='') -> bool:
        if not self._arm_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('arm_controller 서버 없음!')
            return False

        with self.lock:
            js = self.current_joints
        # 안전장치: /joint_states 가 없으면 현재값을 0 으로 가정하지 않는다.
        # (0 으로 가정하면 +pi/-pi 경계에서 한 바퀴 회전하는 위험 동작이 발생)
        if js is None:
            self.get_logger().error('/joint_states 미수신 → 안전을 위해 이동 중단')
            return False
        current = [None] * 4
        for i, name in enumerate(JOINT_NAMES):
            if name in js.name:
                current[i] = js.position[js.name.index(name)]
        if any(c is None for c in current):
            self.get_logger().error(f'joint_states 관절 누락 {current} → 이동 중단')
            return False

        traj, duration = make_trajectory(joints, current)
        if traj is None:
            self.get_logger().error(
                f'{label}: 단일 관절 이동량 {duration:.2f}rad 과대(>{MAX_JOINT_STEP}) '
                f'→ 위험 동작 차단, 이동 중단 (current={[round(c,3) for c in current]}, '
                f'target={[round(t,3) for t in joints]})')
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        self.status_pub.publish(String(data='MOVING'))

        future = self._arm_client.send_goal_async(goal)
        deadline = time.time() + 10.0
        while not future.done():
            if time.time() > deadline:
                self.get_logger().error('액션 수락 타임아웃')
                return False
            time.sleep(0.05)

        gh = future.result()
        if not gh.accepted:
            self.get_logger().error('액션 거부됨')
            return False

        rf = gh.get_result_async()
        deadline = time.time() + duration + 5.0
        while not rf.done():
            if time.time() > deadline:
                self.get_logger().error('실행 타임아웃')
                return False
            time.sleep(0.1)

        ok = (rf.result().result.error_code == FollowJointTrajectory.Result.SUCCESSFUL)
        if not ok:
            self.get_logger().error(f'{label} error_code={rf.result().result.error_code}')
        return ok

    # ─── 초기 홈 ─────────────────────────────────────────────────────────────

    def _init_home(self):
        self._home_timer.cancel()
        threading.Thread(target=self.move_to_joints,
                         args=(ELEVATOR_HOME_JOINTS, 'init_home'), daemon=True).start()


# ─── 엔트리포인트 ─────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = ArmRecoverNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
