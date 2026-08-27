from dataclasses import dataclass
from threading import Event, RLock

from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose


@dataclass(frozen=True)
class NavigationCompletion:
    """Terminal Nav2 outcome for one selected RL frontier."""

    goal_x: float
    goal_y: float
    accepted: bool
    status: int | None

    @property
    def succeeded(self):
        """Return whether Nav2 reported successful completion."""

        return (
            self.accepted
            and self.status
            == GoalStatus.STATUS_SUCCEEDED
        )


def make_navigation_goal(
    *,
    x,
    y,
    stamp,
):
    """Build a Nav2 goal for one selected RL frontier."""

    goal = NavigateToPose.Goal()

    goal.pose.header.frame_id = 'map'
    goal.pose.header.stamp = stamp

    goal.pose.pose.position.x = float(x)
    goal.pose.pose.position.y = float(y)

    goal.pose.pose.orientation.w = 1.0

    return goal


class Nav2GoalExecutor:
    """Track one asynchronous Nav2 goal lifecycle."""

    def __init__(
        self,
        *,
        action_client,
    ):
        self.action_client = action_client

        self._active = False
        self._current_goal = None
        self._completion = None
        self._completion_event = Event()
        self._state_lock = RLock()

    @property
    def active(self):
        """Return whether a Nav2 goal lifecycle is active."""

        with self._state_lock:
            return self._active

    @property
    def current_goal(self):
        """Return the active goal coordinates, if any."""

        with self._state_lock:
            return self._current_goal

    @property
    def completion(self):
        """Return the latest terminal completion, if any."""

        with self._state_lock:
            return self._completion

    def start(
        self,
        *,
        x,
        y,
        stamp,
    ):
        """Start one asynchronous Nav2 goal."""

        with self._state_lock:
            if self._active:
                raise RuntimeError(
                    'A Nav2 goal is already active.'
                )

        if not self.action_client.server_is_ready():
            raise RuntimeError(
                'Nav2 action server is not ready.'
            )

        goal_x = float(x)
        goal_y = float(y)

        goal = make_navigation_goal(
            x=goal_x,
            y=goal_y,
            stamp=stamp,
        )

        with self._state_lock:
            if self._active:
                raise RuntimeError(
                    'A Nav2 goal is already active.'
                )

            self._active = True
            self._current_goal = (
                goal_x,
                goal_y,
            )
            self._completion = None
            self._completion_event.clear()

        try:
            future = (
                self.action_client.send_goal_async(
                    goal
                )
            )

        except Exception:
            with self._state_lock:
                self._active = False
                self._current_goal = None
            raise

        future.add_done_callback(
            self._goal_response_callback
        )

        return future

    def _finish(
        self,
        *,
        accepted,
        status,
    ):
        with self._state_lock:
            goal_x, goal_y = self._current_goal

            completion = NavigationCompletion(
                goal_x=goal_x,
                goal_y=goal_y,
                accepted=bool(accepted),
                status=status,
            )

            self._completion = completion
            self._active = False
            self._current_goal = None

            self._completion_event.set()

            return completion

    def _goal_response_callback(
        self,
        future,
    ):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self._finish(
                accepted=False,
                status=None,
            )
            return

        result_future = (
            goal_handle.get_result_async()
        )

        result_future.add_done_callback(
            self._navigation_result_callback
        )

    def _navigation_result_callback(
        self,
        future,
    ):
        result = future.result()

        self._finish(
            accepted=True,
            status=result.status,
        )

    def wait_for_completion(
        self,
        timeout=None,
    ):
        """Wait for the active goal to reach a terminal outcome."""

        completed = self._completion_event.wait(
            timeout=timeout
        )

        if not completed:
            return None

        with self._state_lock:
            return self._completion
