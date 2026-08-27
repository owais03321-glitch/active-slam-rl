import time

import rclpy

from active_slam_rl.rl_observation_node import (
    RlObservationNode,
)
from active_slam_rl.rl_runtime import (
    RlRosRuntime,
)
from active_slam_rl.rl_simulation import (
    SimulationLifecycle,
)


DEFAULT_SESSION_STARTUP_TIMEOUT_S = 120.0
DEFAULT_SESSION_POLL_INTERVAL_S = 0.10


class FreshRlSession:
    """Own one fresh simulator, ROS context, and live RL runtime."""

    def __init__(
        self,
        *,
        simulation=None,
        node_factory=RlObservationNode,
        runtime_factory=RlRosRuntime,
        startup_timeout_s=DEFAULT_SESSION_STARTUP_TIMEOUT_S,
        poll_interval_s=DEFAULT_SESSION_POLL_INTERVAL_S,
        rclpy_init=rclpy.init,
        rclpy_shutdown=rclpy.shutdown,
        rclpy_ok=rclpy.ok,
        monotonic=time.monotonic,
        sleep=time.sleep,
    ):
        if startup_timeout_s <= 0.0:
            raise ValueError(
                'startup_timeout_s must be greater than zero.'
            )

        if poll_interval_s <= 0.0:
            raise ValueError(
                'poll_interval_s must be greater than zero.'
            )

        self.simulation = (
            simulation
            if simulation is not None
            else SimulationLifecycle()
        )

        self._node_factory = node_factory
        self._runtime_factory = runtime_factory

        self.startup_timeout_s = float(
            startup_timeout_s
        )

        self.poll_interval_s = float(
            poll_interval_s
        )

        self._rclpy_init = rclpy_init
        self._rclpy_shutdown = rclpy_shutdown
        self._rclpy_ok = rclpy_ok

        self._monotonic = monotonic
        self._sleep = sleep

        self._runtime = None
        self._initial_observation = None

        self._started = False
        self._closed = False
        self._owns_ros_context = False

    @property
    def runtime(self):
        """Return the live ROS runtime, if started."""

        return self._runtime

    @property
    def node(self):
        """Return the fresh RL observation node, if started."""

        if self._runtime is None:
            return None

        return self._runtime.node

    @property
    def env(self):
        """Return the live Gym environment, if started."""

        if self._runtime is None:
            return None

        return self._runtime.env

    @property
    def initial_observation(self):
        """Return a copy of the synchronized initial observation."""

        observation = self._initial_observation

        if observation is None:
            return None

        return {
            key: value.copy()
            for key, value
            in observation.items()
        }

    @property
    def running(self):
        """Return whether both simulation and ROS runtime are live."""

        return (
            self._started
            and not self._closed
            and self._runtime is not None
            and self._runtime.running
            and self.simulation.running
        )

    def _ready_observation(self):
        node = self.node

        if node is None:
            return None

        if not bool(
            node.get_parameter(
                'use_sim_time'
            ).value
        ):
            raise RuntimeError(
                'Fresh RL session requires use_sim_time=true.'
            )

        ros_time_ns = (
            node.get_clock()
            .now()
            .nanoseconds
        )

        if ros_time_ns <= 0:
            return None

        if not node.nav_client.wait_for_server(
            timeout_sec=0.0
        ):
            return None

        if node.map_revision <= 0:
            return None

        try:
            node.current_transition_measurements()

        except RuntimeError:
            return None

        try:
            observation = (
                node.sync_env_to_latest_frontier()
            )

        except RuntimeError:
            return None

        if int(
            observation[
                'action_mask'
            ].sum()
        ) <= 0:
            return None

        return observation

    def _wait_until_ready(self):
        deadline = (
            self._monotonic()
            + self.startup_timeout_s
        )

        while True:
            if not self.simulation.running:
                raise RuntimeError(
                    'Simulation stopped before the RL session '
                    'became ready.'
                )

            observation = (
                self._ready_observation()
            )

            if observation is not None:
                return observation

            if self._monotonic() >= deadline:
                raise TimeoutError(
                    'Timed out waiting for fresh RL session '
                    'readiness.'
                )

            self._sleep(
                self.poll_interval_s
            )

    def start(
        self,
        *,
        simulation_stdout=None,
    ):
        """Start and synchronize one completely fresh RL session."""

        if self._closed:
            raise RuntimeError(
                'Fresh RL session is already closed.'
            )

        if self._started:
            raise RuntimeError(
                'Fresh RL session is already started.'
            )

        if self._rclpy_ok():
            raise RuntimeError(
                'Fresh RL session requires an uninitialized '
                'rclpy context.'
            )

        try:
            self.simulation.start(
                stdout=simulation_stdout
            )

            self._rclpy_init(
                args=[
                    '--ros-args',
                    '-p',
                    'use_sim_time:=true',
                ]
            )

            self._owns_ros_context = True

            node = self._node_factory()

            self._runtime = (
                self._runtime_factory(
                    node=node
                )
            )

            self._runtime.start()

            self._initial_observation = (
                self._wait_until_ready()
            )

            self._started = True

            return self.env

        except Exception:
            self.close()
            raise

    def close(self):
        """Close ROS first, then the complete simulation process group."""

        if self._closed:
            return

        first_error = None

        if self._runtime is not None:
            try:
                self._runtime.close()

            except Exception as exc:
                first_error = exc

        try:
            self.simulation.stop()

        except Exception as exc:
            if first_error is None:
                first_error = exc

        if (
            self._owns_ros_context
            and self._rclpy_ok()
        ):
            try:
                self._rclpy_shutdown()

            except Exception as exc:
                if first_error is None:
                    first_error = exc

        self._runtime = None
        self._initial_observation = None

        self._owns_ros_context = False
        self._closed = True

        if first_error is not None:
            raise first_error
