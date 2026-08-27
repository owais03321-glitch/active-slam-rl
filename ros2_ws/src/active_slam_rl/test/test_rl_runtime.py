from threading import Event, get_ident

import pytest

from active_slam_rl.rl_runtime import (
    RlRosRuntime,
)


class FakeNode:

    def __init__(self):
        self.env = object()
        self.destroyed = False

    def destroy_node(self):
        self.destroyed = True


class FakeExecutor:

    def __init__(self):
        self.node = None
        self.removed_node = None
        self.spin_started = Event()
        self.shutdown_requested = Event()
        self.spin_thread_id = None

    def add_node(self, node):
        self.node = node

    def spin(self):
        self.spin_thread_id = get_ident()
        self.spin_started.set()
        self.shutdown_requested.wait()

    def shutdown(self):
        self.shutdown_requested.set()

    def remove_node(self, node):
        self.removed_node = node


def test_runtime_spins_ros_on_dedicated_thread():
    node = FakeNode()
    executor = FakeExecutor()

    runtime = RlRosRuntime(
        node=node,
        executor=executor,
    )

    caller_thread_id = get_ident()

    env = runtime.start()

    assert executor.spin_started.wait(
        timeout=1.0
    )

    assert env is node.env
    assert runtime.running is True
    assert executor.spin_thread_id != caller_thread_id

    runtime.close()

    assert runtime.running is False
    assert executor.removed_node is node
    assert node.destroyed is True


def test_runtime_rejects_duplicate_start():
    runtime = RlRosRuntime(
        node=FakeNode(),
        executor=FakeExecutor(),
    )

    runtime.start()

    assert runtime.executor.spin_started.wait(
        timeout=1.0
    )

    try:
        with pytest.raises(
            RuntimeError,
            match='already running',
        ):
            runtime.start()

    finally:
        runtime.close()


def test_runtime_rejects_restart_after_close():
    runtime = RlRosRuntime(
        node=FakeNode(),
        executor=FakeExecutor(),
    )

    runtime.close()

    with pytest.raises(
        RuntimeError,
        match='already closed',
    ):
        runtime.start()
