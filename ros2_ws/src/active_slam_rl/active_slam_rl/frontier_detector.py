import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from collections import deque


class FrontierDetector(Node):

    def __init__(self):
        super().__init__('frontier_detector')

        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.get_logger().info(
            'Frontier Detector started. Waiting for /map...'
        )

    def map_callback(self, msg):
        width = msg.info.width
        height = msg.info.height
        data = msg.data

        frontier_cells = set()

        # Detect raw frontier cells
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

        clusters = self.cluster_frontiers(frontier_cells)

        self.get_logger().info(
            f'Raw frontiers: {len(frontier_cells)} | '
            f'Clusters: {len(clusters)}'
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

                # 8-connected neighbors
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):

                        if dx == 0 and dy == 0:
                            continue

                        neighbor = (x + dx, y + dy)

                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            queue.append(neighbor)
                            cluster.append(neighbor)

            # Ignore tiny noisy frontier regions
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
