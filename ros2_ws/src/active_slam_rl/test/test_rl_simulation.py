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
        wait_results=None,
    ):
        self.pid = pid
        self.returncode = None
        self.wait_results = list(
            wait_results
            if wait_results is not None
            else [0]
        )
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

        result = self.wait_results.pop(
            0
        )

        if result == 'timeout':
            raise subprocess.TimeoutExpired(
                cmd='simulation',
                timeout=timeout,
            )

        self.returncode = int(
            result
        )

        return self.returncode


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


def test_start_uses_new_process_session_and_redirects_stderr():
    captured = {}
    process = FakeProcess()

    def fake_popen(
        command,
        **kwargs,
    ):
        captured['command'] = command
        captured['kwargs'] = kwargs

        return process

    lifecycle = SimulationLifecycle(
        popen_factory=fake_popen,
    )

    stdout = object()

    returned = lifecycle.start(
        stdout=stdout
    )

    assert returned is process
    assert lifecycle.process is process
    assert lifecycle.running is True

    assert captured['command'] == list(
        DEFAULT_SIMULATION_COMMAND
    )

    assert captured['kwargs'] == {
        'stdout': stdout,
        'stderr': subprocess.STDOUT,
        'start_new_session': True,
    }


def test_double_start_is_rejected():
    process = FakeProcess()

    lifecycle = SimulationLifecycle(
        popen_factory=(
            lambda *args, **kwargs: process
        ),
    )

    lifecycle.start()

    with pytest.raises(
        RuntimeError,
        match='already running',
    ):
        lifecycle.start()


def test_stop_terminates_whole_process_group_with_sigint():
    process = FakeProcess(
        pid=9001,
        wait_results=[
            0,
        ],
    )

    signals = []

    lifecycle = SimulationLifecycle(
        popen_factory=(
            lambda *args, **kwargs: process
        ),
        getpgid=lambda pid: 777,
        killpg=lambda pgid, sig: (
            signals.append(
                (
                    pgid,
                    sig,
                )
            )
        ),
    )

    lifecycle.start()

    assert lifecycle.stop() is True

    assert signals == [
        (
            777,
            signal.SIGINT,
        )
    ]

    assert lifecycle.process is None
    assert lifecycle.running is False


def test_stop_escalates_from_sigint_to_sigterm():
    process = FakeProcess(
        wait_results=[
            'timeout',
            0,
        ],
    )

    signals = []

    lifecycle = SimulationLifecycle(
        shutdown_timeout_s=2.5,
        popen_factory=(
            lambda *args, **kwargs: process
        ),
        getpgid=lambda pid: 888,
        killpg=lambda pgid, sig: (
            signals.append(sig)
        ),
    )

    lifecycle.start()

    assert lifecycle.stop() is True

    assert signals == [
        signal.SIGINT,
        signal.SIGTERM,
    ]

    assert process.wait_timeouts == [
        pytest.approx(2.5),
        pytest.approx(2.5),
    ]


def test_stop_escalates_to_sigkill_and_is_then_idempotent():
    process = FakeProcess(
        wait_results=[
            'timeout',
            'timeout',
            0,
        ],
    )

    signals = []

    lifecycle = SimulationLifecycle(
        popen_factory=(
            lambda *args, **kwargs: process
        ),
        getpgid=lambda pid: 999,
        killpg=lambda pgid, sig: (
            signals.append(sig)
        ),
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
