"""
SVM 기반 정지 중 접촉 감지 노드.

팔이 정지해 있을 때 joint velocity/effort를 최근 10샘플 윈도우로 모아
svm_collision_model.pkl 모델에 넣고, 충돌로 예측되면 joint3·4를 접어
움츠러든 뒤 홈으로 복귀합니다.

실행:
  python3 contact_detector_svm.py

모델 기본 경로:
  이 파일과 같은 폴더의 svm_collision_model.pkl
"""

import math
import threading
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ─── 파라미터 ─────────────────────────────────────────────────────────────────

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4']
MONITOR_JOINTS = ['joint2', 'joint3', 'joint4']

WINDOW_SIZE = 10
MODEL_PATH = Path(__file__).with_name('svm_collision_model.pkl')
VELOCITY_STILL = 0.01
COOLDOWN_SEC = 3.0
SVM_CONSECUTIVE_HITS = 2
SVM_PROB_THRESHOLD = 0.70
POST_ACTION_IGNORE_SEC = 4.0
LOCAL_BASELINE_SAMPLES = 3
HOME_POS_TOL = 0.15
BASELINE_EFFORT_STD_MAX = 5.0
MIN_EFFORT_DELTA = 6.0

HOME_JOINTS = [-3.141, -0.9948, 0.6981, 0.2967]
SHRINK_JOINTS = [-3.141, -0.9948, 1.3000, -1.5700]

POST_MOVING_IGNORE_SEC = 6.0  # MOVING 끝난 뒤 홈 복귀 완료까지 여유 시간
SHRINK_HOLD_SEC = 2.0
MOVE_SPEED = 0.5
MIN_DURATION = 2.0
SHRINK_SPEED = 2.0
SHRINK_MIN_DUR = 0.5


def make_trajectory(target_joints, current_joints,
                    speed: float = MOVE_SPEED, min_dur: float = MIN_DURATION):
    def shortest(t, c):
        diff = (t - c + math.pi) % (2 * math.pi) - math.pi
        return c + diff

    target_joints = [shortest(t, c) for t, c in zip(target_joints, current_joints)]
    max_disp = max(abs(t - c) for t, c in zip(target_joints, current_joints))
    duration = max(max_disp / speed, min_dur)

    traj = JointTrajectory()
    traj.joint_names = JOINT_NAMES

    pt = JointTrajectoryPoint()
    pt.positions = target_joints
    pt.velocities = [0.0] * 4
    secs = int(duration)
    nsecs = int((duration - secs) * 1e9)
    pt.time_from_start = Duration(sec=secs, nanosec=nsecs)
    traj.points.append(pt)

    return traj, duration


class ContactDetectorSvmNode(Node):
    def __init__(self):
        super().__init__('contact_detector_svm')

        self.lock = threading.Lock()
        self.current_joints = None
        self.robot_moving = False
        self._was_moving = False
        self._last_contact_t = 0.0
        self._ignore_until_t = 0.0
        self._svm_hits = 0

        # 최근 10개 feature의 diff를 정확히 만들기 위해 직전 샘플까지 11개 보관합니다.
        self._raw_history = deque(maxlen=WINDOW_SIZE + 1)

        self.model = joblib.load(MODEL_PATH)
        self.get_logger().info(f'SVM 모델 로드 완료: {MODEL_PATH}')

        self._arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')

        self.contact_pub = self.create_publisher(Bool, '/contact_detected', 10)
        self.status_pub = self.create_publisher(String, '/contact_status', 10)

        self.create_subscription(JointState, '/joint_states', self._cb_joint_state, 10)
        self.create_subscription(String, '/robot_status', self._cb_robot_status, 10)

        self.get_logger().info(
            f'SVM 접촉 감지 시작. window={WINDOW_SIZE}, consecutive_hits={SVM_CONSECUTIVE_HITS}')

    # ─── 콜백 ────────────────────────────────────────────────────────────────

    def _cb_robot_status(self, msg: String):
        is_moving = (msg.data == 'MOVING')
        # MOVING → 비이동 전환 시 홈 복귀 완료까지 감지 억제
        if self._was_moving and not is_moving:
            self._ignore_until_t = time.time() + POST_MOVING_IGNORE_SEC
            self._raw_history.clear()
            self._svm_hits = 0
        self.robot_moving = is_moving
        self._was_moving = is_moving

    def _cb_joint_state(self, msg: JointState):
        with self.lock:
            self.current_joints = msg

        if time.time() < self._ignore_until_t:
            self._raw_history.clear()
            self._svm_hits = 0
            return

        raw = self._make_raw_sample(msg)
        if raw is None:
            return

        self._raw_history.append(raw)

        if len(self._raw_history) < WINDOW_SIZE + 1:
            return

        self._check_contact(msg)

    # ─── Feature 생성 ───────────────────────────────────────────────────────

    def _make_raw_sample(self, msg: JointState):
        """모델 학습 때 사용한 [v1..v4, e1..e4] 형태로 현재 샘플을 만듭니다."""
        velocities = []
        efforts = []

        for name in JOINT_NAMES:
            if name not in msg.name:
                return None

            idx = msg.name.index(name)

            if idx >= len(msg.velocity) or idx >= len(msg.effort):
                return None

            velocities.append(float(msg.velocity[idx]))
            efforts.append(float(msg.effort[idx]))

        return np.array(velocities + efforts, dtype=np.float32)

    def _make_svm_feature(self):
        history = np.array(self._raw_history, dtype=np.float32)
        prev = history[0:1]
        window = history[1:]

        velocities = window[:, :4]
        efforts = window[:, 4:]
        baseline = efforts[:LOCAL_BASELINE_SAMPLES].mean(axis=0, keepdims=True)
        effort_delta = efforts - baseline

        raw = np.concatenate([velocities, effort_delta], axis=1)
        prev_raw = np.concatenate([prev[:, :4], prev[:, 4:] - baseline], axis=1)
        diff = np.abs(raw - np.concatenate([prev_raw, raw[:-1]], axis=0))
        feature_window = np.concatenate([raw, diff], axis=1)

        return feature_window.reshape(1, -1)

    # ─── 접촉 감지 ───────────────────────────────────────────────────────────

    def _is_still(self, msg: JointState) -> bool:
        for j in MONITOR_JOINTS:
            if j in msg.name:
                idx = msg.name.index(j)
                if idx >= len(msg.velocity):
                    return False
                if abs(msg.velocity[idx]) > VELOCITY_STILL:
                    return False
        return True

    def _is_near_home(self, msg: JointState) -> bool:
        for i, joint in enumerate(JOINT_NAMES):
            if joint not in msg.name:
                return False

            idx = msg.name.index(joint)
            if idx >= len(msg.position):
                return False

            current = msg.position[idx]
            target = HOME_JOINTS[i]
            diff = (current - target + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) > HOME_POS_TOL:
                return False

        return True

    def _passes_signal_gates(self) -> bool:
        history = np.array(self._raw_history, dtype=np.float32)
        window = history[1:]
        baseline_part = window[:LOCAL_BASELINE_SAMPLES]

        if np.max(np.abs(baseline_part[:, :4])) > VELOCITY_STILL:
            return False

        if np.max(np.std(baseline_part[:, 4:], axis=0)) > BASELINE_EFFORT_STD_MAX:
            return False

        baseline = baseline_part[:, 4:].mean(axis=0, keepdims=True)
        max_effort_delta = np.max(np.abs(window[:, 4:] - baseline))
        if max_effort_delta < MIN_EFFORT_DELTA:
            return False

        return True

    def _check_contact(self, msg: JointState):
        if self.robot_moving or not self._is_still(msg):
            self._raw_history.clear()
            self._svm_hits = 0
            return

        if time.time() - self._last_contact_t < COOLDOWN_SEC:
            return

        if not self._is_near_home(msg):
            self._raw_history.clear()
            self._svm_hits = 0
            return

        if not self._passes_signal_gates():
            self._svm_hits = 0
            return

        feature = self._make_svm_feature()
        prob = None
        if hasattr(self.model, 'predict_proba'):
            prob = float(self.model.predict_proba(feature)[0][1])
            pred = 1 if prob >= SVM_PROB_THRESHOLD else 0
        else:
            pred = int(self.model.predict(feature)[0])

        if pred == 1:
            self._svm_hits += 1
        else:
            self._svm_hits = 0

        if self._svm_hits >= SVM_CONSECUTIVE_HITS:
            probability_text = ''
            if prob is not None:
                probability_text = f' prob={prob:.3f}'

            self.get_logger().warn(
                f'SVM 접촉 감지! hits={self._svm_hits}{probability_text}')
            self._svm_hits = 0
            self._raw_history.clear()
            self._on_contact()

    def _on_contact(self):
        self._last_contact_t = time.time()
        self._ignore_until_t = time.time() + COOLDOWN_SEC
        self.contact_pub.publish(Bool(data=True))
        self.status_pub.publish(String(data='CONTACT_DETECTED'))
        threading.Thread(target=self._shrink_then_home, daemon=True).start()

    # ─── 움츠리기 → 홈 복귀 ─────────────────────────────────────────────────

    def _shrink_then_home(self):
        if not self._arm_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('arm_controller 없음!')
            return

        self.get_logger().info('접촉! joint3·4 접는 중...')
        ok = self._send_joints(SHRINK_JOINTS, speed=SHRINK_SPEED, min_dur=SHRINK_MIN_DUR)
        if not ok:
            self.get_logger().error('움츠리기 실패')

        time.sleep(SHRINK_HOLD_SEC)

        self.get_logger().info('홈으로 복귀 중...')
        self._send_joints(HOME_JOINTS)
        self._raw_history.clear()
        self._ignore_until_t = time.time() + POST_ACTION_IGNORE_SEC
        self.get_logger().info('홈 복귀 완료. 모니터링 재개.')
        self.status_pub.publish(String(data='CONTACT_RESOLVED'))

    def _send_joints(self, target_joints: list,
                     speed: float = MOVE_SPEED, min_dur: float = MIN_DURATION) -> bool:
        with self.lock:
            js = self.current_joints

        current = [0.0] * 4
        if js is not None:
            for i, name in enumerate(JOINT_NAMES):
                if name in js.name:
                    current[i] = js.position[js.name.index(name)]

        traj, duration = make_trajectory(target_joints, current, speed=speed, min_dur=min_dur)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        future = self._arm_client.send_goal_async(goal)
        deadline = time.time() + 10.0
        while not future.done():
            if time.time() > deadline:
                return False
            time.sleep(0.05)

        goal_handle = future.result()
        if not goal_handle.accepted:
            return False

        result_future = goal_handle.get_result_async()
        deadline = time.time() + duration + 5.0
        while not result_future.done():
            if time.time() > deadline:
                return False
            time.sleep(0.1)

        return result_future.result().result.error_code == FollowJointTrajectory.Result.SUCCESSFUL


def main():
    rclpy.init()
    node = ContactDetectorSvmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
