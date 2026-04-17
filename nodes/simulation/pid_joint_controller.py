"""
PID Joint Controller for Isaac Sim

구조:
  bridge.py → /joint_target (위치 목표)
  /joint_states  (Isaac Sim 피드백)
  → PID 계산 → /joint_command (Isaac Sim으로)

50Hz 루프에서 각 관절 독립적으로 PID 제어.
- dead band: 오차가 작으면 보정 안 함 (미세 흔들림 방지)
- derivative 저역통과 필터: 노이즈로 인한 Kd 증폭 억제
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4']

# PID 게인 (각 관절: Kp, Ki, Kd)
GAINS = {
    'joint1': (100.0, 2.0, 300.0),
    'joint2': (100.0, 2.0, 300.0),
    'joint3': (80.0,  1.5, 200.0),
    'joint4': (60.0,  1.0, 100.0),
}

MAX_VELOCITY = 1.0   # rad/s 클램프
DEAD_BAND    = 0.02  # rad — 이 안에 들어오면 보정 안 함
DERIV_ALPHA  = 0.1   # 저역통과 필터 계수 (0~1, 작을수록 더 부드럽게)


class PIDJointController(Node):
    def __init__(self):
        super().__init__('pid_joint_controller')

        self.target: dict[str, float] = {}
        self.actual: dict[str, float] = {}

        self.integral:        dict[str, float] = {n: 0.0 for n in JOINT_NAMES}
        self.prev_error:      dict[str, float] = {n: 0.0 for n in JOINT_NAMES}
        self.filtered_deriv:  dict[str, float] = {n: 0.0 for n in JOINT_NAMES}
        self.prev_time = None

        self.create_subscription(JointState, '/joint_target', self.target_callback, 10)
        self.create_subscription(JointState, '/joint_states', self.state_callback, 10)
        self.cmd_pub = self.create_publisher(JointState, '/joint_command', 10)

        self.create_timer(0.02, self.pid_loop)  # 50Hz

        self.get_logger().info('PID Joint Controller 시작! (50Hz)')
        self.get_logger().info(f'dead band: {DEAD_BAND} rad, deriv filter alpha: {DERIV_ALPHA}')

    def target_callback(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name not in JOINT_NAMES:
                continue
            if name not in self.target or abs(self.target[name] - pos) > 1e-6:
                self.integral[name] = 0.0       # 목표 변경 시 적분 windup 초기화
                self.filtered_deriv[name] = 0.0
            self.target[name] = pos

    def state_callback(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name in JOINT_NAMES:
                self.actual[name] = pos

    def pid_loop(self):
        if not self.target or not self.actual:
            return

        now = self.get_clock().now()
        if self.prev_time is None:
            self.prev_time = now
            return

        dt = (now - self.prev_time).nanoseconds * 1e-9
        if dt <= 0.0:
            return
        self.prev_time = now

        cmd = JointState()
        cmd.header.stamp = now.to_msg()
        cmd.name = []
        cmd.position = []

        for name in JOINT_NAMES:
            if name not in self.target or name not in self.actual:
                continue

            Kp, Ki, Kd = GAINS[name]
            error = self.target[name] - self.actual[name]

            # dead band: 오차가 충분히 작으면 현재 위치 유지
            if abs(error) < DEAD_BAND:
                self.integral[name] = 0.0
                self.filtered_deriv[name] = 0.0
                self.prev_error[name] = error
                cmd.name.append(name)
                cmd.position.append(self.actual[name])
                continue

            # 적분
            self.integral[name] += error * dt

            # derivative 저역통과 필터 (노이즈 억제)
            raw_deriv = (error - self.prev_error[name]) / dt
            self.filtered_deriv[name] = (
                DERIV_ALPHA * raw_deriv + (1.0 - DERIV_ALPHA) * self.filtered_deriv[name]
            )
            self.prev_error[name] = error

            velocity = (Kp * error
                        + Ki * self.integral[name]
                        + Kd * self.filtered_deriv[name])
            velocity = max(-MAX_VELOCITY, min(MAX_VELOCITY, velocity))

            cmd.name.append(name)
            cmd.position.append(self.actual[name] + velocity * dt)

        if cmd.name:
            self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = PIDJointController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
