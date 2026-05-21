"""
배달 모션 데모 테스트.

실측 전에 팔 + 그리퍼 동작 흐름을 확인하기 위한 스크립트.

실행:
  ros2 launch open_manipulator_x_bringup hardware.launch.py
  python3 nodes/real_robot/test_delivery_motion.py

조작:
  Enter → 다음 단계
  q     → 즉시 종료 (홈 복귀 후)
  r     → 처음부터 다시
"""

import math
import threading
import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ─── 링크 파라미터 ────────────────────────────────────────────────────────────

L1    = 0.0595
L2    = math.sqrt(0.024**2 + 0.128**2)
ALPHA = math.atan2(0.128, 0.024)
L3    = 0.124
L4    = 0.126

JOINT_LIMITS = [
    (-math.pi, math.pi),
    (-1.5,     1.5),
    (-1.5,     1.4),
    (-1.7,     1.97),
]

JOINT_NAMES   = ['joint1', 'joint2', 'joint3', 'joint4']
GRIPPER_NAMES = ['gripper']
HOME_JOINTS   = [-3.141, -0.9948, 0.6981, 0.2967]
GRIPPER_OPEN  = [0.01]    # rad
GRIPPER_CLOSE = [-0.005]  # rad (에코백 손잡이 두께에 맞게 조정)
MOVE_SPEED    = 0.4       # 데모용 느리게
MIN_DURATION  = 2.0

# ─── 데모 웨이포인트 ──────────────────────────────────────────────────────────
# 좌표 부호 기준 (joint1 = -π 홈 기준):
#   -X = 팔이 향하는 방향 (문/엘리베이터 방향)
#   +Y = 오른쪽 (문 바라볼 때 레버 있는 쪽)

BASKET_HOVER  = (-0.20,  0.00, 0.12)  # 바구니 위       (TODO: 실측)
BASKET_GRIP   = (-0.20,  0.00, 0.05)  # 에코백 손잡이   (TODO: 실측)

HANDLE_SIDE   = (-0.20,  0.20, 0.23)  # 레버 끝 오른쪽 바깥
HANDLE_INSERT = (-0.20,  0.07, 0.23)  # 슬라이딩 후 (레버 안쪽)
HANDLE_HANG   = (-0.20,  0.07, 0.15)  # 하강 (루프 안착, 더 내려옴)

# ─── IK ──────────────────────────────────────────────────────────────────────

def solve_ik(X, Y, Z):
    j1 = math.atan2(Y, X)
    r  = math.sqrt(X**2 + Y**2)
    wr = r - L4
    dr = wr
    dz = Z - L1
    D  = math.sqrt(dr**2 + dz**2)
    if D > (L2 + L3) * 0.999 or D < abs(L2 - L3) * 1.001:
        return None
    c_psi = max(-1.0, min(1.0, (D**2 - L2**2 - L3**2) / (2.0 * L2 * L3)))
    for psi in (-math.acos(c_psi), math.acos(c_psi)):
        s_psi  = math.sin(psi)
        gamma  = math.atan2(L3 * s_psi, L2 + L3 * c_psi)
        alpha1 = math.atan2(dz, dr) - gamma
        j2     = ALPHA - alpha1
        j3     = -psi - ALPHA
        j4     = -(j2 + j3)
        angles = [j1, j2, j3, j4]
        if all(lo <= a <= hi for a, (lo, hi) in zip(angles, JOINT_LIMITS)):
            return angles
    return None

def _shortest_path(target, current):
    diff = (target - current + math.pi) % (2 * math.pi) - math.pi
    return current + diff

def make_trajectory(target_joints, current_joints):
    target_joints = [_shortest_path(t, c) for t, c in zip(target_joints, current_joints)]
    max_disp = max(abs(t - c) for t, c in zip(target_joints, current_joints))
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

# ─── 슬라이딩 전용 조인트 (joint1만 회전, joint2/3/4 고정) ──────────────────
# HANDLE_SIDE → HANDLE_INSERT 이동 시 joint2 고정 → 몸쪽 진입 방지
_j_side = solve_ik(*HANDLE_SIDE)
HANDLE_SLIDE_JOINTS = None
if _j_side:
    _j1_insert = math.atan2(HANDLE_INSERT[1], HANDLE_INSERT[0])
    HANDLE_SLIDE_JOINTS = [_j1_insert, _j_side[1], _j_side[2], _j_side[3]]

# ─── 시퀀스 정의 ─────────────────────────────────────────────────────────────
# 스텝 형식: (label, joints, xyz, gripper)
#   joints  : 관절값 리스트 or None
#   xyz     : (X, Y, Z) 튜플 or None
#   gripper : GRIPPER_OPEN / GRIPPER_CLOSE / None (생략)

DELIVER_STEPS = [
    ('홈',                        HOME_JOINTS,          None,          None),
    ('그리퍼 열기',                None,                 None,          GRIPPER_OPEN),
    ('바구니 위',                  None,                 BASKET_HOVER,  None),
    ('바구니 하강',                None,                 BASKET_GRIP,   None),
    ('그리퍼 닫기 (에코백 잡기)', None,                 None,          GRIPPER_CLOSE),
    ('바구니 위 들어올리기',       None,                 BASKET_HOVER,  None),
    ('레버 끝 오른쪽 접근',        None,                 HANDLE_SIDE,   None),
    ('왼쪽 슬라이딩 (j1만 회전)', HANDLE_SLIDE_JOINTS,  None,          None),
    ('하강 (루프 안착)',           None,                 HANDLE_HANG,   None),
    ('그리퍼 열기 (에코백 놓기)', None,                 None,          GRIPPER_OPEN),
    ('홈 복귀',                    HOME_JOINTS,          None,          None),
]

RETRIEVE_STEPS = [
    ('홈',                        HOME_JOINTS,  None,          None),
    ('그리퍼 열기',                None,         None,          GRIPPER_OPEN),
    ('루프 집기 위치',             None,         HANDLE_INSERT, None),
    ('그리퍼 닫기 (루프 잡기)',    None,         None,          GRIPPER_CLOSE),
    ('오른쪽 슬라이딩 (이탈)',     None,         HANDLE_SIDE,   None),
    ('바구니 위',                  None,         BASKET_HOVER,  None),
    ('바구니 하강',                None,         BASKET_GRIP,   None),
    ('그리퍼 열기 (에코백 놓기)',  None,         None,          GRIPPER_OPEN),
    ('바구니 위 후퇴',             None,         BASKET_HOVER,  None),
    ('홈 복귀',                    HOME_JOINTS,  None,          None),
]

# ─── 테스트 노드 ──────────────────────────────────────────────────────────────

class DeliveryTestNode(Node):

    def __init__(self):
        super().__init__('test_delivery_motion')
        self.lock           = threading.Lock()
        self.current_joints = None

        self._arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')

        self._gripper_client = ActionClient(
            self, GripperCommand,
            '/gripper_controller/gripper_cmd')

        self.create_subscription(JointState, '/joint_states', self._cb_joints, 10)
        self.get_logger().info('테스트 노드 시작. 2초 후 홈 이동...')

    def _cb_joints(self, msg):
        with self.lock:
            self.current_joints = msg

    # ─── 팔 이동 ─────────────────────────────────────────────────────────────

    def move_to_joints(self, joints, label=''):
        if not self._arm_client.wait_for_server(timeout_sec=5.0):
            print('  ❌ arm_controller 서버 없음!')
            return False

        with self.lock:
            js = self.current_joints
        current = [0.0] * 4
        if js:
            for i, name in enumerate(JOINT_NAMES):
                if name in js.name:
                    current[i] = js.position[js.name.index(name)]

        traj, duration = make_trajectory(joints, current)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        future = self._arm_client.send_goal_async(goal)
        deadline = time.time() + 10.0
        while not future.done():
            if time.time() > deadline:
                print('  ❌ 수락 타임아웃')
                return False
            time.sleep(0.05)

        gh = future.result()
        if not gh.accepted:
            print('  ❌ 액션 거부됨')
            return False

        rf = gh.get_result_async()
        deadline = time.time() + duration + 5.0
        while not rf.done():
            if time.time() > deadline:
                print('  ❌ 실행 타임아웃')
                return False
            time.sleep(0.1)

        ok = (rf.result().result.error_code == FollowJointTrajectory.Result.SUCCESSFUL)
        print('  ✅ 완료' if ok else f'  ❌ 실패 (error_code={rf.result().result.error_code})')
        return ok

    def move_to_xyz(self, X, Y, Z, label=''):
        joints = solve_ik(X, Y, Z)
        if joints is None:
            print(f'  ❌ IK 해 없음: ({X:.3f}, {Y:.3f}, {Z:.3f})')
            return False
        print(f'  IK: j={[f"{j:.3f}" for j in joints]}')
        return self.move_to_joints(joints, label)

    # ─── 그리퍼 ──────────────────────────────────────────────────────────────

    def send_gripper(self, position: list) -> bool:
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            print('  ❌ gripper_controller 서버 없음!')
            return False

        goal = GripperCommand.Goal()
        goal.command.position   = position[0]  # rad
        goal.command.max_effort = 0.0           # 제한 없음

        future = self._gripper_client.send_goal_async(goal)
        deadline = time.time() + 10.0
        while not future.done():
            if time.time() > deadline:
                print('  ❌ 그리퍼 수락 타임아웃')
                return False
            time.sleep(0.05)

        gh = future.result()
        if not gh.accepted:
            print('  ❌ 그리퍼 액션 거부됨')
            return False

        rf = gh.get_result_async()
        deadline = time.time() + 5.0
        while not rf.done():
            if time.time() > deadline:
                print('  ❌ 그리퍼 실행 타임아웃')
                return False
            time.sleep(0.05)

        print('  ✅ 완료')
        return True

    # ─── 시퀀스 실행 ─────────────────────────────────────────────────────────

    def run_sequence(self, steps, name):
        print(f'\n{"="*50}')
        print(f' {name} 시퀀스 시작')
        print(f'{"="*50}')

        for i, (label, joints, xyz, gripper) in enumerate(steps):
            print(f'\n[{i+1}/{len(steps)}] {label}')

            key = input('  Enter: 실행 / q: 종료 / r: 처음부터 > ').strip().lower()
            if key == 'q':
                print('  종료 → 홈 복귀')
                self.move_to_joints(HOME_JOINTS, 'home')
                return 'quit'
            if key == 'r':
                return 'restart'

            # 그리퍼 동작
            if gripper is not None:
                label_g = '열기' if gripper == GRIPPER_OPEN else '닫기'
                print(f'  그리퍼 {label_g} ({gripper[0]} rad)')
                self.send_gripper(gripper)
                time.sleep(0.3)

            # 팔 이동
            if joints is not None:
                self.move_to_joints(joints, label)
            elif xyz is not None:
                if solve_ik(*xyz) is None:
                    print(f'  ⚠️  IK 불가 {xyz} → 스킵')
                else:
                    self.move_to_xyz(*xyz, label=label)

        print(f'\n✅ {name} 시퀀스 완료!\n')
        return 'done'


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = DeliveryTestNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    time.sleep(2.0)
    print('\n홈 포지션으로 이동...')
    node.move_to_joints(HOME_JOINTS, 'home')

    # IK 사전 확인
    print('\n─── 웨이포인트 IK 확인 ───')
    all_ok = True
    for name, xyz in [
        ('BASKET_HOVER',  BASKET_HOVER),
        ('BASKET_GRIP',   BASKET_GRIP),
        ('HANDLE_SIDE',   HANDLE_SIDE),
        ('HANDLE_INSERT', HANDLE_INSERT),
        ('HANDLE_HANG',   HANDLE_HANG),
    ]:
        j = solve_ik(*xyz)
        status = '✅' if j else '❌ IK 불가'
        print(f'  {name:14s} {str(xyz):30s} {status}')
        if not j:
            all_ok = False

    print('\n모든 웨이포인트 IK 가능 ✅' if all_ok else '\n⚠️  IK 불가 웨이포인트 있음. 해당 스텝은 건너뜁니다.')

    while True:
        print('\n─── 메뉴 ───')
        print('  1. 배달 (DELIVER) 시퀀스')
        print('  2. 회수 (RETRIEVE) 시퀀스')
        print('  h. 홈 복귀')
        print('  q. 종료')
        choice = input('선택 > ').strip().lower()

        if choice == '1':
            result = node.run_sequence(DELIVER_STEPS, '배달')
            if result == 'quit':
                break
        elif choice == '2':
            result = node.run_sequence(RETRIEVE_STEPS, '회수')
            if result == 'quit':
                break
        elif choice == 'h':
            node.move_to_joints(HOME_JOINTS, 'home')
        elif choice == 'q':
            node.move_to_joints(HOME_JOINTS, 'home')
            break

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
