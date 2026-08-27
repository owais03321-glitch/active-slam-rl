from threading import Thread

from rclpy.executors import SingleThreadedExecutor

from active_slam_rl.rl_observation_node import (
    RlObservationNode,
)


class RlRosRuntime:
    """Spin ROS independently from synchronous Gym calls."""

    def __init__(
        self,
        *,
        node=None,
        executor=None,
    ):
        self.node = (
            node
            if node is not None
            else RlObservationNode()
        )

        self.executor = (
            executor
            if executor is not None
            else SingleThreadedExecutor()
        )

        self.executor.add_node(
            self.node
        )

        self._spin_thread = None
        self._closed = False

    @property
    def env(self):
        """Return the Gym environment owned by the runtime."""

        return self.node.env

    @property
    def running(self):
        """Return whether the ROS callback thread is alive."""

        return (
            self._spin_thread is not None
            and self._spin_thread.is_alive()
        )

    def start(self):
        """Start ROS callback processing on a dedicated thread."""

        if self._closed:
            raise RuntimeError(
                'RL ROS runtime is already closed.'
            )

        if self.running:
            raise RuntimeError(
                'RL ROS runtime is already running.'
            )

        self._spin_thread = Thread(
            target=self.executor.spin,
            name='active_slam_rl_ros_spin',
            daemon=True,
        )

        self._spin_thread.start()

        return self.env

    def close(self):
        """Stop ROS callback processing and destroy the node."""

        if self._closed:
            return

        self.executor.shutdown()

        if self._spin_thread is not None:
            self._spin_thread.join()

        self.executor.remove_node(
            self.node
        )

        self.node.destroy_node()

        self._closed = True
