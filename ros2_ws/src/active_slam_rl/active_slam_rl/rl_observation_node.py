from collections import deque
from threading import RLock

import rclpy
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import (
    Buffer,
    TransformException,
    TransformListener,
)

from active_slam_rl.rl_env import ActiveSlamEnv
from active_slam_rl.rl_execution import (
    RlActionCoordinator,
)
from active_slam_rl.rl_nav2 import (
    Nav2GoalExecutor,
)
from active_slam_rl.rl_observation import (
    DEFAULT_MAX_CANDIDATES,
    encode_frontier_observation,
)
from active_slam_rl.rl_path import (
    PathLengthTracker,
)
from active_slam_rl.rl_ros_map import (
    eligible_frontier_candidates_from_occupancy_grid,
    explored_area_m2_from_occupancy_grid,
)
from active_slam_rl.rl_transition import (
    GoalTransitionTracker,
)


class RlObservationNode(Node):
    """Build live RL frontier observations and execution state."""

    def __init__(
        self,
        *,
        max_candidates=DEFAULT_MAX_CANDIDATES,
    ):
        super().__init__('rl_observation_node')

        self._state_lock = RLock()

        self.env = ActiveSlamEnv(
            max_candidates=max_candidates,
        )

        self.path_tracker = PathLengthTracker()

        self.latest_explored_area_m2 = None

        self.visited_goals = deque(
            maxlen=100,
        )

        self.latest_candidates = []
        self.latest_robot_xy = None

        self.latest_observation, _ = (
            self.env.reset()
        )

        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            '/navigate_to_pose',
        )

        self.nav_executor = Nav2GoalExecutor(
            action_client=self.nav_client,
        )

        self.transition_tracker = (
            GoalTransitionTracker()
        )

        self.action_coordinator = (
            RlActionCoordinator(
                env=self.env,
                transition_tracker=(
                    self.transition_tracker
                ),
                nav_executor=self.nav_executor,
                visited_goals=self.visited_goals,
                visited_goals_lock=self._state_lock,
            )
        )

        self.map_subscription = (
            self.create_subscription(
                OccupancyGrid,
                '/map',
                self.map_callback,
                10,
            )
        )

        self.odom_subscription = (
            self.create_subscription(
                Odometry,
                '/odom',
                self.odom_callback,
                50,
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
        """Update the latest live ROS frontier snapshot."""

        with self._state_lock:
            visited_goals = tuple(
                self.visited_goals
            )

        explored_area_m2 = (
            explored_area_m2_from_occupancy_grid(
                msg
            )
        )

        candidates = (
            eligible_frontier_candidates_from_occupancy_grid(
                msg,
                robot_x=robot_x,
                robot_y=robot_y,
                visited_goals=visited_goals,
            )
        )

        observation = encode_frontier_observation(
            candidates=candidates,
            robot_x=robot_x,
            robot_y=robot_y,
            max_candidates=self.env.max_candidates,
        )

        with self._state_lock:
            self.latest_explored_area_m2 = (
                explored_area_m2
            )

            self.latest_candidates = list(
                candidates
            )

            self.latest_robot_xy = (
                float(robot_x),
                float(robot_y),
            )

            self.latest_observation = {
                key: value.copy()
                for key, value
                in observation.items()
            }

            return {
                key: value.copy()
                for key, value
                in observation.items()
            }

    def sync_env_to_latest_frontier(self):
        """Freeze the latest ROS frontier snapshot into Gym state."""

        with self._state_lock:
            if self.latest_robot_xy is None:
                raise RuntimeError(
                    'Frontier state is not available yet.'
                )

            candidates = tuple(
                self.latest_candidates
            )

            robot_x, robot_y = (
                self.latest_robot_xy
            )

        return self.env.set_frontier_state(
            candidates=candidates,
            robot_x=robot_x,
            robot_y=robot_y,
        )

    def current_transition_measurements(self):
        """Return cumulative measurements required to start/finish RL."""

        with self._state_lock:
            if self.latest_explored_area_m2 is None:
                raise RuntimeError(
                    'Explored-area measurement is not available yet.'
                )

            if self.path_tracker.last_xy is None:
                raise RuntimeError(
                    'Odometry measurement is not available yet.'
                )

            return (
                float(self.latest_explored_area_m2),
                float(self.path_tracker.path_length_m),
            )

    def start_rl_action(
        self,
        action,
    ):
        """Start one RL-selected frontier using current live state."""

        area_m2, path_m = (
            self.current_transition_measurements()
        )

        stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        return self.action_coordinator.start_action(
            action=action,
            area_m2=area_m2,
            path_m=path_m,
            stamp=stamp,
        )

    def complete_rl_action(
        self,
        *,
        timeout=None,
    ):
        """Wait for Nav2 and complete the active RL transition."""

        return self.action_coordinator.complete_action(
            measurement_provider=(
                self.current_transition_measurements
            ),
            timeout=timeout,
        )

    def odom_callback(self, msg):
        with self._state_lock:
            self.path_tracker.update(
                x=msg.pose.pose.position.x,
                y=msg.pose.pose.position.y,
            )

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
