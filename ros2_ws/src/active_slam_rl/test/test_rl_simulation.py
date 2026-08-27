import signal
import subprocess

import pytest

from active_slam_rl.rl_simulation import (
    DEFAULT_SIMULATION_COMMAND,
    SimulationLifecycle,
)


class FakeProcess:

    def __init__(
        self,
        *,
        pid=4321,
    ):
        self.pid = pid
        self.returncode = None
        self.wait_timeouts = []

    def poll(self):
        return self.returncode

    def wait(
        self,
        *,
        timeout,
    ):
        self.wait_timeouts.append(
            timeout
        )

        if self.returncode is None:
            raise subprocess.TimeoutExpired(
                cmd='simulation',
                timeout=timeout,
            )

        return self.returncode


class FakeClock:

    def __init__(self):
        self.value = 0.0

    def __call__(self):
        current = self.value
        self.value += 1.0
        return current


def test_default_command_matches_frozen_simulation_contract():
    assert DEFAULT_SIMULATION_COMMAND == (
        'ros2',
        'launch',
        'nav2_bringup',
        'tb3_simulation_launch.py',
        'slam:=True',
        'use_rviz:=False',
        'headless:=True',
    )


def test_start_tracks_new_process_group_independently():
    captured = {}
    process = FakeProcess(
        pid=4321
    )
    groups = {
        4321,
    }

    def fake_popen(
        command,
        **kwargs,
    ):
        captured['command'] = command
        captured['kwargs'] = kwargs
        return process

    lifecycle = SimulationLifecycle(
        popen_factory=fake_popen,
        group_exists=(
            lambda pgid: pgid in groups
        ),
    )

    stdout = object()

    returned = lifecycle.start(
        stdout=stdout
    )

    assert returned is process
    assert lifecycle.process is process
    assert lifecycle.process_group_id == 4321
    assert lifecycle.running is True

    assert captured['command'] == list(
        DEFAULT_SIMULATION_COMMAND
    )

    assert captured['kwargs'] == {
        'stdout': stdout,
        'stderr': subprocess.STDOUT,
        'start_new_session': True,
    }


def test_double_start_is_rejected_while_group_exists():
    process = FakeProcess()
    groups = {
        process.pid,
    }

    lifecycle = SimulationLifecycle(
        popen_factory=(
            lambda *args, **kwargs: process
        ),
        group_exists=(
            lambda pgid: pgid in groups
        ),
    )

    lifecycle.start()

    with pytest.raises(
        RuntimeError,
        match='already running',
    ):
        lifecycle.start()


def test_stop_terminates_whole_group_with_sigint():
    process = FakeProcess(
        pid=9001
    )
    groups = {
        9001,
    }
    signals = []

    def fake_killpg(
        pgid,
        stop_signal,
    ):
        signals.append(
            (
                pgid,
                stop_signal,
            )
        )
        groups.discard(
            pgid
        )
        process.returncode = 0

    lifecycle = SimulationLifecycle(
        popen_factory=(
            lambda *args, **kwargs: process
        ),
        killpg=fake_killpg,
        group_exists=(
            lambda pgid: pgid in groups
        ),
    )

    lifecycle.start()

    assert lifecycle.stop() is True

    assert signals == [
        (
            9001,
            signal.SIGINT,
        )
    ]

    assert lifecycle.process is None
    assert lifecycle.process_group_id is None
    assert lifecycle.running is False


def test_stop_kills_descendants_after_launch_parent_exits():
    process = FakeProcess(
        pid=9002
    )
    groups = {
        9002,
    }
    signals = []

    def fake_killpg(
        pgid,
        stop_signal,
    ):
        signals.append(
            (
                pgid,
                stop_signal,
            )
        )
        groups.discard(
            pgid
        )

    lifecycle = SimulationLifecycle(
        popen_factory=(
            lambda *args, **kwargs: process
        ),
        killpg=fake_killpg,
        group_exists=(
            lambda pgid: pgid in groups
        ),
    )

    lifecycle.start()

    # Simulate the ros2-launch parent dying while Gazebo/Nav2/SLAM
    # descendants remain in the originally created process group.
    process.returncode = 1

    assert process.poll() == 1
    assert lifecycle.running is True

    assert lifecycle.stop() is True

    assert signals == [
        (
            9002,
            signal.SIGINT,
        )
    ]

    assert lifecycle.running is False
    assert lifecycle.process is None
    assert lifecycle.process_group_id is None


def test_stop_is_idempotent_when_parent_and_group_are_gone():
    process = FakeProcess(
        pid=9003
    )
    groups = {
        9003,
    }

    lifecycle = SimulationLifecycle(
        popen_factory=(
            lambda *args, **kwargs: process
        ),
        group_exists=(
            lambda pgid: pgid in groups
        ),
    )

    lifecycle.start()

    process.returncode = 0
    groups.clear()

    assert lifecycle.stop() is False
    assert lifecycle.stop() is False

    assert lifecycle.process is None
    assert lifecycle.process_group_id is None
    assert lifecycle.running is False


def test_stop_escalates_from_sigint_to_sigterm():
    process = FakeProcess(
        pid=9004
    )
    groups = {
        9004,
    }
    signals = []

    def fake_killpg(
        pgid,
        stop_signal,
    ):
        signals.append(
            stop_signal
        )

        if stop_signal == signal.SIGTERM:
            groups.discard(
                pgid
            )
            process.returncode = 0

    lifecycle = SimulationLifecycle(
        shutdown_timeout_s=1.5,
        popen_factory=(
            lambda *args, **kwargs: process
        ),
        killpg=fake_killpg,
        group_exists=(
            lambda pgid: pgid in groups
        ),
        monotonic=FakeClock(),
        sleep=lambda duration: None,
    )

    lifecycle.start()

    assert lifecycle.stop() is True

    assert signals == [
        signal.SIGINT,
        signal.SIGTERM,
    ]


def test_stop_escalates_to_sigkill():
    process = FakeProcess(
        pid=9005
    )
    groups = {
        9005,
    }
    signals = []

    def fake_killpg(
        pgid,
        stop_signal,
    ):
        signals.append(
            stop_signal
        )

        if stop_signal == signal.SIGKILL:
            groups.discard(
                pgid
            )
            process.returncode = -9

    lifecycle = SimulationLifecycle(
        shutdown_timeout_s=1.5,
        popen_factory=(
            lambda *args, **kwargs: process
        ),
        killpg=fake_killpg,
        group_exists=(
            lambda pgid: pgid in groups
        ),
        monotonic=FakeClock(),
        sleep=lambda duration: None,
    )

    lifecycle.start()

    assert lifecycle.stop() is True

    assert signals == [
        signal.SIGINT,
        signal.SIGTERM,
        signal.SIGKILL,
    ]

    assert lifecycle.stop() is False
    assert lifecycle.running is False
