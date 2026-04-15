import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
import time

class IsaacMoveItBridge(Node):
    def __init__(self):
        super().__init__('isaac_moveit_bridge')
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
            self.execute_callback
        )
        self.joint_pub = self.create_publisher(
            JointState, '/joint_target', 10
        )
        self.get_logger().info('Isaac MoveIt2 Bridge 시작!')

    async def execute_callback(self, goal_handle):
        trajectory = goal_handle.request.trajectory
        for point in trajectory.points:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = trajectory.joint_names
            msg.position = list(point.positions)
            self.joint_pub.publish(msg)
            time.sleep(0.05)
        goal_handle.succeed()
        return FollowJointTrajectory.Result()

def main():
    rclpy.init()
    node = IsaacMoveItBridge()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
