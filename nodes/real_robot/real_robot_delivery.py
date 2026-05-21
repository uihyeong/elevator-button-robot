"""
배달/회수 노드.

새벽 무인 배달 시나리오 (03~05시):
  [배달] IDLE → PICKUP (바구니에서 에코백 집기) → HANG (문고리에 걸기) → DONE
  [회수] IDLE → RETRIEVE (문고리에서 에코백 집기) → PLACE (바구니에 넣기) → DONE

현재 구현: 카메라 없이 고정 웨이포인트 기반 동작
추후 개선: Gemini VLM으로 문고리 자동 인식 추가 예정

실행 순서:
  ros2 launch open_manipulator_x_bringup hardware.launch.py
  python3 nodes/real_robot/real_robot_delivery.py

명령 전송:
  ros2 topic pub --once /delivery_command std_msgs/String "{data: 'DELIVER'}"
  ros2 topic pub --once /delivery_command std_msgs/String "{data: 'RETRIEVE'}"

⚠️  웨이포인트 (BASKET_*, HANDLE_*) 는 실측 후 반드시 교체할 것.
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
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ─── OpenMANIPULATOR-X 링크 파라미터 (unified.py 와 동일) ─────────────────────

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

GRIPPER_OPEN  = [0.01]    # rad (열기)
GRIPPER_CLOSE = [-0.005]  # rad (닫기) — 에코백 손잡이 두께에 맞게 현장 조정

MOVE_SPEED    = 0.5   # rad/s
MIN_DURATION  = 2.0   # 초

# ─── 웨이포인트 (arm base = world 프레임 기준, 단위: m) ──────────────────────
# ⚠️  TODO: 아래 좌표는 플레이스홀더. 실제 로봇 팔로 직접 실측 후 교체 필요.
#
# 측정 방법 (저녁 현장 테스트 시):
#   1. 팔을 원하는 위치로 수동으로 이동
#   2. ros2 topic echo /joint_states 로 관절값 확인
#   3. solve_ik(X, Y, Z) 로 IK 계산 가능 여부 확인
#
# 좌표 기준 (joint1 = -π 홈 기준):
#   -X = 팔이 향하는 방향 (문/엘리베이터 방향)
#   +X = 팔 등 뒤 방향 (Scout Mini 상판/바구니 방향)
#   +Y = 오른쪽 (문 바라볼 때 레버 있는 쪽)
#   -Y = 왼쪽
#   Z  = arm base 기준 높이 (지면 + 약 0.724m)
#      → 예) 지면에서 1.0m = Z ≈ 1.0 - 0.724 = 0.276m

# 바구니 (Scout Mini 상판, 팔 앞 아래쪽 -X 방향)
BASKET_HOVER_XYZ = (-0.15,  0.0,  0.12)  # TODO: 바구니 위 ~10cm 위치
BASKET_GRIP_XYZ  = (-0.15,  0.0,  0.03)  # TODO: 에코백 손잡이 실제 위치

# 레버형 문고리 — 걸기 전략
# ┌──────────────────┐
# │       문          │
# │              ════╪═══  ← 레버 (오른쪽으로 수평)
# └──────────────────┘
#
# [배달] 오른쪽에서 왼쪽으로 수평 이동하며 루프를 레버에 끼움
#   SIDE    : 레버 끝 오른쪽 바깥 (루프가 레버와 같은 높이)
#   INSERT  : 왼쪽 수평 이동 → 레버가 루프 안으로 들어옴
#   그리퍼 오픈 → 오른쪽으로 후퇴
#
# [회수] 레버 안쪽에서 집어 위로 들어올린 후 오른쪽으로 빼냄
#   GRIP    : 레버에 걸린 에코백 루프 위치 (안쪽, 레버 높이)
#   그리퍼 클로즈 → 위로 들어올리기 → 오른쪽으로 후퇴

# TODO: 실측 후 아래 6개 값만 채우면 됨
LEVER_X     = -0.30  # TODO: arm base → 문 X 거리 (음수, -X 방향)
LEVER_Y_TIP =  0.20  # TODO: 레버 끝 Y좌표 (오른쪽이면 양수)
LEVER_Y_MID =  0.10  # TODO: 레버 중간 Y좌표 (끼운 후 안착 위치)
LEVER_Z     = 0.18   # TODO: arm base 기준 레버 높이 (지면높이 - 0.724)
LOOP_HEIGHT = 0.10   # TODO: 에코백 루프 높이 (그리퍼 ~ 루프 바닥, 실측)

# 계산된 웨이포인트
HANDLE_SIDE_XYZ   = (LEVER_X, LEVER_Y_TIP - 0.05, LEVER_Z + LOOP_HEIGHT)
# 레버 끝 오른쪽 바깥 5cm, 루프가 레버 높이에 오도록 Z 보정

HANDLE_INSERT_XYZ = (LEVER_X, LEVER_Y_MID,         LEVER_Z + LOOP_HEIGHT)
# 왼쪽으로 수평 이동 → 레버가 루프 안으로 들어옴

HANDLE_HANG_XYZ   = (LEVER_X, LEVER_Y_MID,         LEVER_Z + LOOP_HEIGHT - 0.02)
# 2cm 하강으로 루프 레버에 안착

HANDLE_GRIP_XYZ   = (LEVER_X, LEVER_Y_MID,         LEVER_Z + LOOP_HEIGHT)
# 회수 시 집기 위치 (걸린 상태 그대로)

# ─── 상태 상수 ────────────────────────────────────────────────────────────────

IDLE     = 'IDLE'
PICKUP   = 'PICKUP'      # 바구니에서 에코백 집기
HANG     = 'HANG'        # 문고리에 에코백 걸기
RETRIEVE = 'RETRIEVE'    # 문고리에서 에코백 집기
PLACE    = 'PLACE'       # 바구니에 에코백 내려놓기
DONE     = 'DONE'


# ─── 해석적 IK (unified.py 와 동일) ──────────────────────────────────────────

def solve_ik(X: float, Y: float, Z: float):
    """
    end_effector 수평 접근 해석적 IK.
    반환: [j1, j2, j3, j4] (rad) 또는 None (도달 불가).
    """
    j1 = math.atan2(Y, X)
    r  = math.sqrt(X**2 + Y**2)

    wr = r - L4
    dr = wr
    dz = Z - L1
    D  = math.sqrt(dr**2 + dz**2)

    if D > (L2 + L3) * 0.999:
        return None
    if D < abs(L2 - L3) * 1.001:
        return None

    c_psi = (D**2 - L2**2 - L3**2) / (2.0 * L2 * L3)
    c_psi = max(-1.0, min(1.0, c_psi))

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


def _shortest_path(target: float, current: float) -> float:
    diff = (target - current + math.pi) % (2 * math.pi) - math.pi
    return current + diff


def make_trajectory(target_joints: list, current_joints: list,
                    joint_names: list = None, speed: float = MOVE_SPEED):
    if joint_names is None:
        joint_names = JOINT_NAMES
    target_joints = [_shortest_path(t, c) for t, c in zip(target_joints, current_joints)]
    max_disp = max(abs(t - c) for t, c in zip(target_joints, current_joints))
    duration = max(max_disp / speed, MIN_DURATION)

    traj            = JointTrajectory()
    traj.joint_names = joint_names
    pt              = JointTrajectoryPoint()
    pt.positions    = target_joints
    pt.velocities   = [0.0] * len(target_joints)
    secs  = int(duration)
    nsecs = int((duration - secs) * 1e9)
    pt.time_from_start = Duration(sec=secs, nanosec=nsecs)
    traj.points.append(pt)

    return traj, duration


# ─── 배달 노드 ────────────────────────────────────────────────────────────────

class DeliveryRobot(Node):

    def __init__(self):
        super().__init__('real_robot_delivery')

        self.lock           = threading.Lock()
        self.state          = IDLE
        self.moving         = False
        self.current_joints = None

        # arm 액션 클라이언트
        self._arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')

        # gripper 액션 클라이언트 (unified.py 에 없던 신규)
        self._gripper_client = ActionClient(
            self, GripperCommand,
            '/gripper_controller/gripper_cmd')

        # 발행
        self.status_pub = self.create_publisher(String, '/delivery_status', 10)

        # 구독
        self.create_subscription(String,     '/delivery_command', self._cb_command,     10)
        self.create_subscription(JointState, '/joint_states',     self._cb_joint_state, 10)

        self.get_logger().info('배달 노드 시작. 2초 후 홈 이동...')
        self._home_timer = self.create_timer(2.0, self._move_to_home_once)

    # ─── 콜백 ────────────────────────────────────────────────────────────────

    def _cb_joint_state(self, msg: JointState):
        with self.lock:
            self.current_joints = msg

    def _cb_command(self, msg: String):
        cmd = msg.data.strip().upper()

        if self.state not in (IDLE, DONE):
            self.get_logger().warn(f'작업 중 ({self.state}). 명령 무시: {cmd}')
            return

        if cmd == 'DELIVER':
            self.get_logger().info('=== 배달 시작 ===')
            self.state = PICKUP
            threading.Thread(target=self._run_deliver, daemon=True).start()

        elif cmd == 'RETRIEVE':
            self.get_logger().info('=== 회수 시작 ===')
            self.state = RETRIEVE
            threading.Thread(target=self._run_retrieve, daemon=True).start()

        else:
            self.get_logger().warn(
                f'알 수 없는 명령: "{cmd}"  →  DELIVER 또는 RETRIEVE 사용')

    # ─── 배달 시퀀스 ─────────────────────────────────────────────────────────

    def _run_deliver(self):
        """
        바구니에서 에코백 집어 → 레버 문고리에 걸기

        [걸기 전략] 오른쪽 → 왼쪽 수평 슬라이딩
          레버 끝(오른쪽) 바깥에서 접근
          → 왼쪽으로 수평 이동하며 레버가 루프 안으로 들어옴
          → 그리퍼 오픈 → 오른쪽으로 후퇴
        """

        # ① 그리퍼 열기 (집기 준비)
        self.get_logger().info('[배달 1/9] 그리퍼 열기')
        if not self._send_gripper(GRIPPER_OPEN):
            return self._fail('그리퍼 열기 실패')

        # ② 바구니 위로 이동
        self.get_logger().info('[배달 2/9] 바구니 위로 이동')
        if not self._move_xyz(*BASKET_HOVER_XYZ, label='basket_hover'):
            return self._fail('바구니 hover 이동 실패')

        # ③ 바구니로 하강 (에코백 손잡이 위치)
        self.get_logger().info('[배달 3/9] 바구니 하강')
        if not self._move_xyz(*BASKET_GRIP_XYZ, label='basket_grip'):
            return self._fail('바구니 하강 실패')

        # ④ 그리퍼 닫기 (에코백 잡기)
        self.get_logger().info('[배달 4/9] 그리퍼 닫기 (에코백 잡기)')
        if not self._send_gripper(GRIPPER_CLOSE):
            return self._fail('그리퍼 닫기 실패')
        time.sleep(0.5)  # 그립 안정화

        # ⑤ 에코백 들어올리기
        self.get_logger().info('[배달 5/9] 에코백 들어올리기')
        if not self._move_xyz(*BASKET_HOVER_XYZ, label='basket_lift'):
            return self._fail('들어올리기 실패')

        # ⑥ 레버 끝 오른쪽 바깥으로 이동 (루프가 레버와 같은 높이)
        self.state = HANG
        self.get_logger().info('[배달 6/9] 레버 끝 오른쪽 접근')
        if not self._move_xyz(*HANDLE_SIDE_XYZ, label='handle_side'):
            return self._fail('레버 오른쪽 접근 실패')

        # ⑦ 왼쪽으로 수평 이동 → 레버가 루프 안으로 들어옴
        self.get_logger().info('[배달 7/9] 왼쪽으로 슬라이딩 (레버에 끼우기)')
        if not self._move_xyz(*HANDLE_INSERT_XYZ, label='handle_insert'):
            return self._fail('레버 끼우기 실패')

        # ⑧ 살짝 하강 → 루프 레버에 안착
        self.get_logger().info('[배달 8/9] 하강하여 루프 안착')
        if not self._move_xyz(*HANDLE_HANG_XYZ, label='handle_hang'):
            return self._fail('루프 안착 실패')

        # ⑨ 그리퍼 열기 (에코백 놓기) → 오른쪽으로 후퇴
        self.get_logger().info('[배달 9/9] 그리퍼 열기 → 오른쪽 후퇴 → 홈')
        if not self._send_gripper(GRIPPER_OPEN):
            return self._fail('그리퍼 열기 실패')
        time.sleep(0.3)
        self._move_xyz(*HANDLE_SIDE_XYZ, label='handle_retreat')  # 실패해도 계속

        self.get_logger().info('✅ 배달 완료!')
        self.status_pub.publish(String(data='DELIVERED'))
        self.state = DONE
        self._move_to_home()

    # ─── 회수 시퀀스 ─────────────────────────────────────────────────────────

    def _run_retrieve(self):
        """
        레버 문고리에서 에코백 집어 → 바구니에 내려놓기

        [회수 전략] 걸기의 역순 — 잡은 후 오른쪽으로 수평 슬라이딩
          레버 중간(안쪽)에서 루프 잡기
          → 그리퍼 클로즈 → 오른쪽으로 수평 이동하며 루프가 레버 끝에서 이탈
          → 바구니로 이동
        """

        # ① 그리퍼 열기
        self.get_logger().info('[회수 1/8] 그리퍼 열기')
        if not self._send_gripper(GRIPPER_OPEN):
            return self._fail('그리퍼 열기 실패')

        # ② 레버 끝 오른쪽 바깥으로 먼저 이동 (충돌 없이 접근)
        self.get_logger().info('[회수 2/8] 레버 오른쪽 바깥으로 접근')
        if not self._move_xyz(*HANDLE_SIDE_XYZ, label='handle_side'):
            return self._fail('레버 오른쪽 접근 실패')

        # ③ 왼쪽으로 수평 이동 → 루프 잡을 위치로
        self.get_logger().info('[회수 3/8] 왼쪽 이동 (루프 잡기 위치)')
        if not self._move_xyz(*HANDLE_GRIP_XYZ, label='handle_grip'):
            return self._fail('루프 집기 위치 이동 실패')

        # ④ 그리퍼 닫기 (에코백 루프 잡기)
        self.get_logger().info('[회수 4/8] 그리퍼 닫기 (루프 잡기)')
        if not self._send_gripper(GRIPPER_CLOSE):
            return self._fail('그리퍼 닫기 실패')
        time.sleep(0.5)

        # ⑤ 오른쪽으로 수평 이동 → 루프가 레버 끝에서 이탈
        self.get_logger().info('[회수 5/8] 오른쪽 슬라이딩 (레버에서 이탈)')
        if not self._move_xyz(*HANDLE_SIDE_XYZ, label='handle_slide_out'):
            return self._fail('레버 이탈 실패')

        # ⑥ 바구니 위로 이동
        self.state = PLACE
        self.get_logger().info('[회수 6/8] 바구니 위로 이동')
        if not self._move_xyz(*BASKET_HOVER_XYZ, label='basket_hover'):
            return self._fail('바구니 hover 이동 실패')

        # ⑦ 바구니로 하강
        self.get_logger().info('[회수 7/8] 바구니에 내려놓기')
        if not self._move_xyz(*BASKET_GRIP_XYZ, label='basket_place'):
            return self._fail('바구니 하강 실패')

        # ⑧ 그리퍼 열기 → 후퇴
        self.get_logger().info('[회수 8/8] 그리퍼 열기 → 홈 복귀')
        if not self._send_gripper(GRIPPER_OPEN):
            return self._fail('그리퍼 열기 실패')
        time.sleep(0.3)
        self._move_xyz(*BASKET_HOVER_XYZ, label='basket_retreat')  # 실패해도 계속

        self.get_logger().info('✅ 회수 완료!')
        self.status_pub.publish(String(data='RETRIEVED'))
        self.state = DONE
        self._move_to_home()

    # ─── 이동 헬퍼 ───────────────────────────────────────────────────────────

    def _move_xyz(self, X: float, Y: float, Z: float, label: str = '') -> bool:
        """XYZ 좌표로 이동. IK 실패 또는 궤적 실패 시 False 반환."""
        joints = solve_ik(X, Y, Z)
        if joints is None:
            self.get_logger().error(
                f'IK 해 없음 [{label}]: ({X:.3f}, {Y:.3f}, {Z:.3f})')
            return False
        self.get_logger().info(
            f'  [{label}] ({X:.3f},{Y:.3f},{Z:.3f}) → '
            f'joints={[f"{j:.3f}" for j in joints]}')
        return self._send_trajectory(joints)

    def _fail(self, reason: str):
        """실패 처리: 로그 출력 + 상태 초기화 + 홈 복귀."""
        self.get_logger().error(f'❌ {reason}')
        self.status_pub.publish(String(data='FAILED'))
        self.state = IDLE
        threading.Thread(target=self._move_to_home, daemon=True).start()

    # ─── 궤적 전송 (unified.py 와 동일 구조) ──────────────────────────────────

    def _send_trajectory(self, target_joints: list, blocking: bool = True) -> bool:
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

        future   = self._arm_client.send_goal_async(goal)
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

    def _send_gripper(self, position: list) -> bool:
        """그리퍼 열기/닫기. position: [rad]"""
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('gripper_controller 액션 서버 없음!')
            return False

        goal = GripperCommand.Goal()
        goal.command.position   = position[0]  # rad
        goal.command.max_effort = 0.0

        future   = self._gripper_client.send_goal_async(goal)
        deadline = time.time() + 10.0
        while not future.done():
            if time.time() > deadline:
                self.get_logger().error('그리퍼 액션 수락 타임아웃')
                return False
            time.sleep(0.05)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('그리퍼 액션 거부됨')
            return False

        result_future = goal_handle.get_result_async()
        deadline = time.time() + 5.0
        while not result_future.done():
            if time.time() > deadline:
                self.get_logger().error('그리퍼 실행 타임아웃')
                return False
            time.sleep(0.05)

        return True

    # ─── Home ────────────────────────────────────────────────────────────────

    def _move_to_home_once(self):
        self._home_timer.cancel()
        threading.Thread(target=self._move_to_home, daemon=True).start()

    def _move_to_home(self):
        self.get_logger().info('홈 포지션으로 이동 중...')
        ok = self._send_trajectory(HOME_JOINTS)
        if ok:
            self.get_logger().info('✅ 홈 도착')
        else:
            self.get_logger().error('❌ 홈 이동 실패')

        if self.state == DONE:
            self.state = IDLE
            self.get_logger().info('대기 중... (DELIVER / RETRIEVE 명령 대기)')


# ─── 엔트리포인트 ─────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = DeliveryRobot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
