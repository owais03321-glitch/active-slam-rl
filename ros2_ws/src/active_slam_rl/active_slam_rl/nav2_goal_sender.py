import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from nav2_msgs.action import NavigateToPose


class Nav2GoalSender(Node):

    def __init__(self):
        super().__init__('nav2_goal_sender')

        self.declare_parameter('goal_x', 0.0)
        self.declare_parameter('goal_y', 0.0)

        self.goal_x = self.get_parameter('goal_x').value
        self.goal_y = self.get_parameter('goal_y').value

        self.last_reported_distance = None

        self.action_client = ActionClient(
            self,
            NavigateToPose,
            '/navigate_to_pose'
        )

        self.send_goal()

    def send_goal(self):
        self.get_logger().info('Waiting for Nav2...')

        self.action_client.wait_for_server()

        goal = NavigateToPose.Goal()

        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()

        goal.pose.pose.position.x = self.goal_x
        goal.pose.pose.position.y = self.goal_y
        goal.pose.pose.orientation.w = 1.0

        self.get_logger().info(
            f'Sending goal: x={self.goal_x:.2f}, y={self.goal_y:.2f}'
        )

        future = self.action_client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback
        )

        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Navigation goal rejected.')
            return

        self.get_logger().info('Navigation goal accepted.')

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        distance = feedback_msg.feedback.distance_remaining

        # Print only when distance changes by about 0.1 m
        if (
            self.last_reported_distance is None
            or abs(distance - self.last_reported_distance) >= 0.1
        ):
            self.get_logger().info(
                f'Distance remaining: {distance:.2f} m'
            )
            self.last_reported_distance = distance

    def result_callback(self, future):
        result = future.result()

        self.get_logger().info(
            f'Navigation finished with status: {result.status}'
        )

        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    node = Nav2GoalSender()

    rclpy.spin(node)


if __name__ == '__main__':
    main()
