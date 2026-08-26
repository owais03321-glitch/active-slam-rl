import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid


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

        frontier_cells = []

        for y in range(1, height - 1):
            for x in range(1, width - 1):

                index = y * width + x

                # Frontier candidates must be free space
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
                    frontier_cells.append((x, y))

        self.get_logger().info(
            f'Detected {len(frontier_cells)} frontier cells'
        )


def main(args=None):
    rclpy.init(args=args)

    node = FrontierDetector()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
