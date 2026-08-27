import os
import signal
import subprocess


DEFAULT_SIMULATION_COMMAND = (
    'ros2',
    'launch',
    'nav2_bringup',
    'tb3_simulation_launch.py',
    'slam:=True',
    'use_rviz:=False',
    'headless:=True',
)


class SimulationLifecycle:
    """Own one complete Gazebo/Nav2/SLAM launch process group."""

    def __init__(
        self,
        *,
        command=DEFAULT_SIMULATION_COMMAND,
        shutdown_timeout_s=5.0,
        popen_factory=subprocess.Popen,
        getpgid=os.getpgid,
        killpg=os.killpg,
    ):
        if shutdown_timeout_s <= 0.0:
            raise ValueError(
                'shutdown_timeout_s must be greater than zero.'
            )

        self.command = tuple(
            command
        )

        if not self.command:
            raise ValueError(
                'command must not be empty.'
            )

        self.shutdown_timeout_s = float(
            shutdown_timeout_s
        )

        self._popen_factory = popen_factory
        self._getpgid = getpgid
        self._killpg = killpg
        self._process = None

    @property
    def process(self):
        """Return the currently tracked launch process, if any."""

        return self._process

    @property
    def running(self):
        """Return whether the tracked launch process is alive."""

        return (
            self._process is not None
            and self._process.poll() is None
        )

    def start(
        self,
        *,
        stdout=None,
    ):
        """Start one fresh simulation in an isolated process group."""

        if self.running:
            raise RuntimeError(
                'Simulation is already running.'
            )

        if (
            self._process is not None
            and self._process.poll() is not None
        ):
            self._process = None

        output = (
            stdout
            if stdout is not None
            else subprocess.DEVNULL
        )

        self._process = self._popen_factory(
            list(
                self.command
            ),
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        return self._process

    def _wait_for_exit(self):
        try:
            self._process.wait(
                timeout=self.shutdown_timeout_s
            )

        except subprocess.TimeoutExpired:
            return False

        return True

    def stop(self):
        """Stop the complete simulation process group idempotently."""

        process = self._process

        if process is None:
            return False

        if process.poll() is not None:
            self._process = None
            return False

        process_group = self._getpgid(
            process.pid
        )

        for stop_signal in (
            signal.SIGINT,
            signal.SIGTERM,
            signal.SIGKILL,
        ):
            self._killpg(
                process_group,
                stop_signal,
            )

            if self._wait_for_exit():
                self._process = None
                return True

        raise RuntimeError(
            'Simulation process group did not exit '
            'after SIGKILL.'
        )
