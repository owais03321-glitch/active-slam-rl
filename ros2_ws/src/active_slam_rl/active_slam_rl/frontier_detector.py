import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformException, TransformListener


class FrontierDetector(Node):

    def __init__(self):
        super().__init__('frontier_detector')

        self.subscription = self.create_subscription(
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

        self.get_logger().info(
            'Frontier Detector started. Waiting for /map...'
        )

    def map_callback(self, msg):
        width = msg.info.width
        height = msg.info.height
        data = msg.data

        frontier_cells = set()

        # 1. Detect raw frontier cells
        for y in range(1, height - 1):
            for x in range(1, width - 1):

                index = y * width + x

                # Frontier candidates must be free cells
                if data[index] != 0:
                    continue

                neighbors = [
                    data[index - 1],
                    data[index + 1],
                    data[index - width],
                    data[index + width]
                ]

                # Free cell touching unknown space
                if -1 in neighbors:
                    frontier_cells.add((x, y))

        # 2. Cluster neighboring frontier cells
        clusters = self.cluster_frontiers(frontier_cells)

        # 3. Convert clusters into world-coordinate goals
        goals = []

        for cluster in clusters:

            avg_x = sum(
                point[0] for point in cluster
            ) / len(cluster)

            avg_y = sum(
                point[1] for point in cluster
            ) / len(cluster)

            cell_x, cell_y = min(
                cluster,
                key=lambda point:
                (point[0] - avg_x) ** 2
                + (point[1] - avg_y) ** 2
            )

            world_x = (
                msg.info.origin.position.x
                + (cell_x + 0.5) * msg.info.resolution
            )

            world_y = (
                msg.info.origin.position.y
                + (cell_y + 0.5) * msg.info.resolution
            )

            goals.append((world_x, world_y))

        self.get_logger().info(
            f'Raw frontiers: {len(frontier_cells)} | '
            f'Clusters: {len(clusters)}'
        )

        for index, (x, y) in enumerate(goals):
            self.get_logger().info(
                f'Frontier {index + 1}: '
                f'x={x:.2f} m, y={y:.2f} m'
            )

        # 4. Get robot pose in the map frame
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
                f'Could not get robot pose: {exc}'
            )
            return

        # 5. Select nearest frontier
        if goals:

            nearest_goal = min(
                goals,
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
                f'Robot: x={robot_x:.2f}, '
                f'y={robot_y:.2f} | '
                f'Nearest frontier: '
                f'x={nearest_goal[0]:.2f}, '
                f'y={nearest_goal[1]:.2f} | '
                f'Distance: {distance:.2f} m'
            )

    def cluster_frontiers(self, frontier_cells):
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

                            remaining.remove(neighbor)
                            queue.append(neighbor)
                            cluster.append(neighbor)

            # Ignore tiny/noisy frontier regions
            if len(cluster) >= 5:
                clusters.append(cluster)

        return clusters


def main(args=None):
    rclpy.init(args=args)

    node = FrontierDetector()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
