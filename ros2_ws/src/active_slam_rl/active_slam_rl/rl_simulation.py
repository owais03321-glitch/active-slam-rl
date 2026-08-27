import os
import signal
import subprocess
import time


DEFAULT_SIMULATION_COMMAND = (
    'ros2',
    'launch',
    'nav2_bringup',
    'tb3_simulation_launch.py',
    'slam:=True',
    'use_rviz:=False',
    'headless:=True',
)


def _process_group_exists(process_group_id):
    """Return whether any process still belongs to the group."""

    try:
        os.killpg(
            process_group_id,
            0,
        )

    except ProcessLookupError:
        return False

    except PermissionError:
        return True

    return True


class SimulationLifecycle:
    """Own one complete Gazebo/Nav2/SLAM launch process group."""

    def __init__(
        self,
        *,
        command=DEFAULT_SIMULATION_COMMAND,
        shutdown_timeout_s=5.0,
        shutdown_poll_interval_s=0.05,
        popen_factory=subprocess.Popen,
        killpg=os.killpg,
        group_exists=_process_group_exists,
        monotonic=time.monotonic,
        sleep=time.sleep,
    ):
        if shutdown_timeout_s <= 0.0:
            raise ValueError(
                'shutdown_timeout_s must be greater than zero.'
            )

        if shutdown_poll_interval_s <= 0.0:
            raise ValueError(
                'shutdown_poll_interval_s must be greater than zero.'
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

        self.shutdown_poll_interval_s = float(
            shutdown_poll_interval_s
        )

        self._popen_factory = popen_factory
        self._killpg = killpg
        self._group_exists = group_exists
        self._monotonic = monotonic
        self._sleep = sleep

        self._process = None
        self._process_group_id = None

    @property
    def process(self):
        """Return the tracked ros2-launch parent, if any."""

        return self._process

    @property
    def process_group_id(self):
        """Return the independently tracked simulation process group."""

        return self._process_group_id

    @property
    def running(self):
        """Return whether any tracked simulation process remains alive."""

        return (
            self._process_group_id is not None
            and self._group_exists(
                self._process_group_id
            )
        )

    def _clear_tracking(self):
        self._process = None
        self._process_group_id = None

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

        self._clear_tracking()

        output = (
            stdout
            if stdout is not None
            else subprocess.DEVNULL
        )

        process = self._popen_factory(
            list(
                self.command
            ),
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        self._process = process

        # start_new_session=True makes the child the leader of a new
        # session and process group, therefore PGID == child PID.
        # Store it now so descendants can still be terminated even if
        # the ros2-launch parent exits first.
        self._process_group_id = int(
            process.pid
        )

        return process

    def _reap_parent_if_exited(self):
        process = self._process

        if process is None:
            return

        try:
            process.wait(
                timeout=0.0
            )

        except subprocess.TimeoutExpired:
            pass

    def _wait_for_group_exit(
        self,
        process_group_id,
    ):
        deadline = (
            self._monotonic()
            + self.shutdown_timeout_s
        )

        while self._group_exists(
            process_group_id
        ):
            remaining_s = (
                deadline
                - self._monotonic()
            )

            if remaining_s <= 0.0:
                return False

            self._sleep(
                min(
                    self.shutdown_poll_interval_s,
                    remaining_s,
                )
            )

        return True

    def stop(self):
        """Stop every process in the tracked simulation group."""

        process_group_id = (
            self._process_group_id
        )

        if process_group_id is None:
            self._process = None
            return False

        if not self._group_exists(
            process_group_id
        ):
            self._reap_parent_if_exited()
            self._clear_tracking()
            return False

        for stop_signal in (
            signal.SIGINT,
            signal.SIGTERM,
            signal.SIGKILL,
        ):
            try:
                self._killpg(
                    process_group_id,
                    stop_signal,
                )

            except ProcessLookupError:
                self._reap_parent_if_exited()
                self._clear_tracking()
                return True

            if self._wait_for_group_exit(
                process_group_id
            ):
                self._reap_parent_if_exited()
                self._clear_tracking()
                return True

        raise RuntimeError(
            'Simulation process group did not exit '
            'after SIGKILL.'
        )
