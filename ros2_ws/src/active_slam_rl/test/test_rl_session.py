import numpy as np
import pytest

from active_slam_rl.rl_session import (
    FreshRlSession,
)


class FakeSimulation:

    def __init__(
        self,
        events,
    ):
        self.events = events
        self.running = False

    def start(
        self,
        *,
        stdout=None,
    ):
        self.events.append(
            (
                'simulation_start',
                stdout,
            )
        )

        self.running = True

        return object()

    def stop(self):
        self.events.append(
            'simulation_stop'
        )

        was_running = (
            self.running
        )

        self.running = False

        return was_running


class FakeParameter:

    def __init__(
        self,
        value,
    ):
        self.value = value


class FakeTime:

    def __init__(
        self,
        nanoseconds,
    ):
        self.nanoseconds = nanoseconds


class FakeClock:

    def now(self):
        return FakeTime(
            5_000_000_000
        )


class FakeNavClient:

    def __init__(
        self,
        *,
        ready=True,
    ):
        self.ready = ready

    def wait_for_server(
        self,
        *,
        timeout_sec,
    ):
        assert timeout_sec == 0.0
        return self.ready


class FakeEnv:

    pass


class FakeNode:

    def __init__(
        self,
        *,
        use_sim_time=True,
        nav_ready=True,
        map_revision=1,
        measurement_ready=True,
        valid_actions=1,
    ):
        self.env = FakeEnv()

        self._use_sim_time = (
            use_sim_time
        )

        self.nav_client = FakeNavClient(
            ready=nav_ready
        )

        self.map_revision = (
            map_revision
        )

        self.measurement_ready = (
            measurement_ready
        )

        self.valid_actions = (
            valid_actions
        )

    def get_parameter(
        self,
        name,
    ):
        assert name == 'use_sim_time'

        return FakeParameter(
            self._use_sim_time
        )

    def get_clock(self):
        return FakeClock()

    def current_transition_measurements(self):
        if not self.measurement_ready:
            raise RuntimeError(
                'measurements not ready'
            )

        return (
            1.0,
            0.0,
        )

    def sync_env_to_latest_frontier(self):
        if self.map_revision <= 0:
            raise RuntimeError(
                'frontier state unavailable'
            )

        mask = np.zeros(
            4,
            dtype=np.int8,
        )

        mask[
            :self.valid_actions
        ] = 1

        return {
            'candidates': np.zeros(
                (
                    4,
                    4,
                ),
                dtype=np.float32,
            ),
            'action_mask': mask,
        }


class FakeRuntime:

    def __init__(
        self,
        *,
        node,
        events,
    ):
        self.node = node
        self.env = node.env
        self.events = events
        self.running = False

    def start(self):
        self.events.append(
            'runtime_start'
        )

        self.running = True

        return self.env

    def close(self):
        self.events.append(
            'runtime_close'
        )

        self.running = False


class FakeMonotonic:

    def __init__(self):
        self.value = 0.0

    def __call__(self):
        value = self.value
        self.value += 1.0
        return value


def make_ros_context(
    events,
    *,
    initially_ok=False,
):
    state = {
        'ok': initially_ok,
    }

    def ros_ok():
        return state[
            'ok'
        ]

    def ros_init(
        *,
        args,
    ):
        events.append(
            (
                'rclpy_init',
                tuple(args),
            )
        )

        state[
            'ok'
        ] = True

    def ros_shutdown():
        events.append(
            'rclpy_shutdown'
        )

        state[
            'ok'
        ] = False

    return (
        ros_init,
        ros_shutdown,
        ros_ok,
    )


def test_constructor_rejects_invalid_wait_configuration():
    with pytest.raises(
        ValueError,
        match='startup_timeout_s',
    ):
        FreshRlSession(
            startup_timeout_s=0.0
        )

    with pytest.raises(
        ValueError,
        match='poll_interval_s',
    ):
        FreshRlSession(
            poll_interval_s=0.0
        )


def test_start_builds_fresh_ready_session_in_order():
    events = []

    simulation = FakeSimulation(
        events
    )

    node = FakeNode(
        valid_actions=2
    )

    (
        ros_init,
        ros_shutdown,
        ros_ok,
    ) = make_ros_context(
        events
    )

    session = FreshRlSession(
        simulation=simulation,
        node_factory=(
            lambda: node
        ),
        runtime_factory=(
            lambda *,
            node: FakeRuntime(
                node=node,
                events=events,
            )
        ),
        rclpy_init=ros_init,
        rclpy_shutdown=ros_shutdown,
        rclpy_ok=ros_ok,
    )

    stdout = object()

    env = session.start(
        simulation_stdout=stdout
    )

    assert env is node.env
    assert session.env is node.env
    assert session.node is node
    assert session.running is True

    assert (
        session.initial_observation[
            'action_mask'
        ].tolist()
        == [
            1,
            1,
            0,
            0,
        ]
    )

    assert events[:3] == [
        (
            'simulation_start',
            stdout,
        ),
        (
            'rclpy_init',
            (
                '--ros-args',
                '-p',
                'use_sim_time:=true',
            ),
        ),
        'runtime_start',
    ]


def test_start_rejects_preexisting_ros_context():
    events = []

    simulation = FakeSimulation(
        events
    )

    (
        ros_init,
        ros_shutdown,
        ros_ok,
    ) = make_ros_context(
        events,
        initially_ok=True,
    )

    session = FreshRlSession(
        simulation=simulation,
        rclpy_init=ros_init,
        rclpy_shutdown=ros_shutdown,
        rclpy_ok=ros_ok,
    )

    with pytest.raises(
        RuntimeError,
        match='uninitialized rclpy context',
    ):
        session.start()

    assert events == []
    assert simulation.running is False


def test_failed_readiness_cleans_runtime_simulation_and_ros():
    events = []

    simulation = FakeSimulation(
        events
    )

    node = FakeNode(
        nav_ready=False
    )

    (
        ros_init,
        ros_shutdown,
        ros_ok,
    ) = make_ros_context(
        events
    )

    session = FreshRlSession(
        simulation=simulation,
        node_factory=(
            lambda: node
        ),
        runtime_factory=(
            lambda *,
            node: FakeRuntime(
                node=node,
                events=events,
            )
        ),
        startup_timeout_s=1.5,
        poll_interval_s=0.01,
        rclpy_init=ros_init,
        rclpy_shutdown=ros_shutdown,
        rclpy_ok=ros_ok,
        monotonic=FakeMonotonic(),
        sleep=lambda duration: None,
    )

    with pytest.raises(
        TimeoutError,
        match='readiness',
    ):
        session.start()

    assert events[-3:] == [
        'runtime_close',
        'simulation_stop',
        'rclpy_shutdown',
    ]

    assert simulation.running is False
    assert session.runtime is None
    assert session.running is False


def test_close_orders_runtime_before_simulation_before_ros():
    events = []

    simulation = FakeSimulation(
        events
    )

    node = FakeNode()

    (
        ros_init,
        ros_shutdown,
        ros_ok,
    ) = make_ros_context(
        events
    )

    session = FreshRlSession(
        simulation=simulation,
        node_factory=(
            lambda: node
        ),
        runtime_factory=(
            lambda *,
            node: FakeRuntime(
                node=node,
                events=events,
            )
        ),
        rclpy_init=ros_init,
        rclpy_shutdown=ros_shutdown,
        rclpy_ok=ros_ok,
    )

    session.start()

    session.close()
    session.close()

    assert events[-3:] == [
        'runtime_close',
        'simulation_stop',
        'rclpy_shutdown',
    ]

    assert session.runtime is None
    assert session.env is None
    assert session.running is False


def test_started_or_closed_session_cannot_be_reused():
    events = []

    simulation = FakeSimulation(
        events
    )

    node = FakeNode()

    (
        ros_init,
        ros_shutdown,
        ros_ok,
    ) = make_ros_context(
        events
    )

    session = FreshRlSession(
        simulation=simulation,
        node_factory=(
            lambda: node
        ),
        runtime_factory=(
            lambda *,
            node: FakeRuntime(
                node=node,
                events=events,
            )
        ),
        rclpy_init=ros_init,
        rclpy_shutdown=ros_shutdown,
        rclpy_ok=ros_ok,
    )

    session.start()

    with pytest.raises(
        RuntimeError,
        match='already started',
    ):
        session.start()

    session.close()

    with pytest.raises(
        RuntimeError,
        match='already closed',
    ):
        session.start()
