import math
from collections import deque

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from tf2_ros import Buffer, TransformException, TransformListener


class FrontierExplorer(Node):

    def __init__(self):
        super().__init__('frontier_explorer')

        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            '/navigate_to_pose'
        )

        self.goal_active = False
        self.last_reported_distance = None

        self.get_logger().info(
            'Frontier Explorer started.'
        )

    def map_callback(self, msg):

        # Do not choose another frontier while Nav2 is busy.
        if self.goal_active:
            return

        goals = self.find_frontier_goals(msg)

        if not goals:
            self.get_logger().info(
                'No frontier goals found.'
            )
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_footprint',
                Time()
            )

            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y

        except TransformException as exc:
            self.get_logger().warning(
                f'Robot pose unavailable: {exc}'
            )
            return

        # Ignore goals that are already extremely close.
        valid_goals = [
            goal for goal in goals
            if math.hypot(
                goal[0] - robot_x,
                goal[1] - robot_y
            ) > 0.35
        ]

        if not valid_goals:
            self.get_logger().info(
                'No useful frontier goals available.'
            )
            return

        nearest_goal = min(
            valid_goals,
            key=lambda goal: math.hypot(
                goal[0] - robot_x,
                goal[1] - robot_y
            )
        )

        distance = math.hypot(
            nearest_goal[0] - robot_x,
            nearest_goal[1] - robot_y
        )

        self.get_logger().info(
            f'Selected frontier: '
            f'x={nearest_goal[0]:.2f}, '
            f'y={nearest_goal[1]:.2f}, '
            f'distance={distance:.2f} m'
        )

        self.send_navigation_goal(
            nearest_goal[0],
            nearest_goal[1]
        )

    def find_frontier_goals(self, msg):

        width = msg.info.width
        height = msg.info.height
        data = msg.data

        frontier_cells = set()

        # Detect free cells touching unknown space.
        for y in range(1, height - 1):
            for x in range(1, width - 1):

                index = y * width + x

                if data[index] != 0:
                    continue

                neighbors = [
                    data[index - 1],
                    data[index + 1],
                    data[index - width],
                    data[index + width]
                ]

                if -1 in neighbors:
                    frontier_cells.add((x, y))

        clusters = self.cluster_frontiers(
            frontier_cells
        )

        goals = []

        for cluster in clusters:

            avg_x = sum(
                point[0] for point in cluster
            ) / len(cluster)

            avg_y = sum(
                point[1] for point in cluster
            ) / len(cluster)

            # Pick a real frontier cell nearest the
            # geometric center of the cluster.
            cell_x, cell_y = min(
                cluster,
                key=lambda point:
                (point[0] - avg_x) ** 2
                + (point[1] - avg_y) ** 2
            )

            world_x = (
                msg.info.origin.position.x
                + (cell_x + 0.5)
                * msg.info.resolution
            )

            world_y = (
                msg.info.origin.position.y
                + (cell_y + 0.5)
                * msg.info.resolution
            )

            goals.append(
                (world_x, world_y)
            )

        self.get_logger().info(
            f'Frontier cells: {len(frontier_cells)} | '
            f'Clusters: {len(clusters)}'
        )

        return goals

    def cluster_frontiers(
        self,
        frontier_cells
    ):

        remaining = set(frontier_cells)
        clusters = []

        while remaining:

            start = remaining.pop()

            queue = deque([start])
            cluster = [start]

            while queue:

                x, y = queue.popleft()

                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):

                        if dx == 0 and dy == 0:
                            continue

                        neighbor = (
                            x + dx,
                            y + dy
                        )

                        if neighbor in remaining:

                            remaining.remove(
                                neighbor
                            )

                            queue.append(
                                neighbor
                            )

                            cluster.append(
                                neighbor
                            )

            # Filter tiny noisy frontiers.
            if len(cluster) >= 5:
                clusters.append(cluster)

        return clusters

    def send_navigation_goal(
        self,
        x,
        y
    ):

        if not self.nav_client.server_is_ready():

            self.get_logger().warning(
                'Nav2 action server not ready.'
            )
            return

        goal = NavigateToPose.Goal()

        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = (
            self.get_clock().now().to_msg()
        )

        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.w = 1.0

        # Set this before sending so another map
        # callback cannot send another goal.
        self.goal_active = True
        self.last_reported_distance = None

        future = self.nav_client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback
        )

        future.add_done_callback(
            self.goal_response_callback
        )

    def goal_response_callback(
        self,
        future
    ):

        goal_handle = future.result()

        if not goal_handle.accepted:

            self.get_logger().warning(
                'Frontier goal rejected.'
            )

            self.goal_active = False
            return

        self.get_logger().info(
            'Frontier goal accepted.'
        )

        result_future = (
            goal_handle.get_result_async()
        )

        result_future.add_done_callback(
            self.navigation_result_callback
        )

    def feedback_callback(
        self,
        feedback_msg
    ):

        distance = (
            feedback_msg.feedback
            .distance_remaining
        )

        # Avoid terminal spam.
        if (
            self.last_reported_distance is None
            or abs(
                distance
                - self.last_reported_distance
            ) >= 0.20
        ):

            self.get_logger().info(
                f'Distance remaining: '
                f'{distance:.2f} m'
            )

            self.last_reported_distance = (
                distance
            )

    def navigation_result_callback(
        self,
        future
    ):

        result = future.result()

        self.get_logger().info(
            f'Navigation finished. '
            f'Status: {result.status}'
        )

        self.goal_active = False
        self.last_reported_distance = None

        self.get_logger().info(
            'Waiting for updated map '
            'before selecting next frontier...'
        )


def main(args=None):

    rclpy.init(args=args)

    node = FrontierExplorer()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
