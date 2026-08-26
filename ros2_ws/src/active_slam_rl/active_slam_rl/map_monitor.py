import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid


class MapMonitor(Node):
    def __init__(self):
        super().__init__('map_monitor')

        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.get_logger().info('Map Monitor started. Waiting for /map...')

    def map_callback(self, msg):
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        total_cells = len(msg.data)
        unknown_cells = msg.data.count(-1)
        free_cells = msg.data.count(0)
        occupied_cells = total_cells - unknown_cells - free_cells

        known_cells = total_cells - unknown_cells

        coverage = (
            known_cells / total_cells * 100.0
            if total_cells > 0 else 0.0
        )

        self.get_logger().info(
            f'Map: {width}x{height} | '
            f'Resolution: {resolution:.2f} m | '
            f'Coverage: {coverage:.2f}% | '
            f'Free: {free_cells} | '
            f'Occupied: {occupied_cells} | '
            f'Unknown: {unknown_cells}'
        )


def main(args=None):
    rclpy.init(args=args)

    node = MapMonitor()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
