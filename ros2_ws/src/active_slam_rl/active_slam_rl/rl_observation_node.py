from collections import deque

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import (
    Buffer,
    TransformException,
    TransformListener,
)

from active_slam_rl.rl_env import ActiveSlamEnv
from active_slam_rl.rl_observation import (
    DEFAULT_MAX_CANDIDATES,
)
from active_slam_rl.rl_ros_map import (
    eligible_frontier_candidates_from_occupancy_grid,
)


class RlObservationNode(Node):
    """Build live RL frontier observations from ROS map and TF data."""

    def __init__(
        self,
        *,
        max_candidates=DEFAULT_MAX_CANDIDATES,
    ):
        super().__init__('rl_observation_node')

        self.env = ActiveSlamEnv(
            max_candidates=max_candidates,
        )

        self.visited_goals = deque(
            maxlen=100,
        )

        self.latest_candidates = []

        self.latest_observation, _ = (
            self.env.reset()
        )

        self.map_subscription = (
            self.create_subscription(
                OccupancyGrid,
                '/map',
                self.map_callback,
                10,
            )
        )

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.get_logger().info(
            'RL observation node started.'
        )

    def _lookup_robot_position(self):
        transform = self.tf_buffer.lookup_transform(
            'map',
            'base_footprint',
            Time(),
        )

        return (
            transform.transform.translation.x,
            transform.transform.translation.y,
        )

    def update_from_map(
        self,
        msg,
        *,
        robot_x,
        robot_y,
    ):
        """Update the RL state from one map and known robot position."""

        candidates = (
            eligible_frontier_candidates_from_occupancy_grid(
                msg,
                robot_x=robot_x,
                robot_y=robot_y,
                visited_goals=self.visited_goals,
            )
        )

        observation = self.env.set_frontier_state(
            candidates=candidates,
            robot_x=robot_x,
            robot_y=robot_y,
        )

        self.latest_candidates = list(
            candidates
        )

        self.latest_observation = observation

        return observation

    def map_callback(self, msg):
        try:
            robot_x, robot_y = (
                self._lookup_robot_position()
            )

        except TransformException as exc:
            self.get_logger().warning(
                f'Robot pose unavailable: {exc}'
            )
            return

        observation = self.update_from_map(
            msg,
            robot_x=robot_x,
            robot_y=robot_y,
        )

        valid_actions = int(
            observation['action_mask'].sum()
        )

        self.get_logger().info(
            'RL frontier observation updated: '
            f'candidates={valid_actions}'
        )


def main(args=None):
    rclpy.init(args=args)

    node = RlObservationNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
