from threading import Thread

import pytest
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time

from active_slam_rl.rl_nav2 import (
    Nav2GoalExecutor,
    make_navigation_goal,
)


class FakeFuture:

    def __init__(self):
        self._result = None
        self._callbacks = []

    def add_done_callback(
        self,
        callback,
    ):
        self._callbacks.append(callback)

    def result(self):
        return self._result

    def resolve(
        self,
        result,
    ):
        self._result = result

        for callback in list(
            self._callbacks
        ):
            callback(self)


class FakeGoalHandle:

    def __init__(
        self,
        *,
        accepted,
        result_future=None,
    ):
        self.accepted = accepted
        self.result_future = result_future
        self.cancel_calls = 0

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancel_calls += 1
        return FakeFuture()


class FakeActionClient:

    def __init__(
        self,
        *,
        ready=True,
    ):
        self.ready = ready
        self.sent_goals = []
        self.send_future = FakeFuture()

    def server_is_ready(self):
        return self.ready

    def send_goal_async(
        self,
        goal,
    ):
        self.sent_goals.append(goal)

        return self.send_future


class FakeResult:

    def __init__(
        self,
        *,
        status,
    ):
        self.status = status


def test_navigation_goal_matches_frozen_baseline_contract():
    stamp = Time(
        sec=123,
        nanosec=456,
    )

    goal = make_navigation_goal(
        x=1.25,
        y=-0.75,
        stamp=stamp,
    )

    assert goal.pose.header.frame_id == 'map'
    assert goal.pose.header.stamp == stamp

    assert (
        goal.pose.pose.position.x
        == pytest.approx(1.25)
    )

    assert (
        goal.pose.pose.position.y
        == pytest.approx(-0.75)
    )

    assert (
        goal.pose.pose.orientation.x
        == pytest.approx(0.0)
    )

    assert (
        goal.pose.pose.orientation.y
        == pytest.approx(0.0)
    )

    assert (
        goal.pose.pose.orientation.z
        == pytest.approx(0.0)
    )

    assert (
        goal.pose.pose.orientation.w
        == pytest.approx(1.0)
    )


def test_navigation_goal_converts_coordinates_to_float():
    stamp = Time()

    goal = make_navigation_goal(
        x=2,
        y=3,
        stamp=stamp,
    )

    assert goal.pose.pose.position.x == 2.0
    assert goal.pose.pose.position.y == 3.0


def test_executor_rejects_start_when_server_not_ready():
    executor = Nav2GoalExecutor(
        action_client=FakeActionClient(
            ready=False,
        )
    )

    with pytest.raises(
        RuntimeError,
        match='Nav2 action server is not ready',
    ):
        executor.start(
            x=1.0,
            y=2.0,
            stamp=Time(),
        )

    assert executor.active is False
    assert executor.current_goal is None


def test_executor_rejects_overlapping_goal():
    action_client = FakeActionClient()

    executor = Nav2GoalExecutor(
        action_client=action_client
    )

    executor.start(
        x=1.0,
        y=2.0,
        stamp=Time(),
    )

    assert executor.active is True
    assert executor.current_goal == (
        1.0,
        2.0,
    )

    with pytest.raises(
        RuntimeError,
        match='already active',
    ):
        executor.start(
            x=3.0,
            y=4.0,
            stamp=Time(),
        )


def test_rejected_goal_produces_terminal_completion():
    action_client = FakeActionClient()

    executor = Nav2GoalExecutor(
        action_client=action_client
    )

    executor.start(
        x=1.25,
        y=-0.75,
        stamp=Time(),
    )

    action_client.send_future.resolve(
        FakeGoalHandle(
            accepted=False,
        )
    )

    completion = executor.wait_for_completion(
        timeout=0.0
    )

    assert completion is not None
    assert completion.accepted is False
    assert completion.status is None
    assert completion.succeeded is False

    assert completion.goal_x == pytest.approx(
        1.25
    )
    assert completion.goal_y == pytest.approx(
        -0.75
    )

    assert executor.active is False
    assert executor.current_goal is None


@pytest.mark.parametrize(
    (
        'status',
        'expected_succeeded',
    ),
    [
        (
            GoalStatus.STATUS_SUCCEEDED,
            True,
        ),
        (
            GoalStatus.STATUS_ABORTED,
            False,
        ),
        (
            GoalStatus.STATUS_CANCELED,
            False,
        ),
    ],
)
def test_accepted_goal_preserves_terminal_nav2_status(
    status,
    expected_succeeded,
):
    action_client = FakeActionClient()
    result_future = FakeFuture()

    executor = Nav2GoalExecutor(
        action_client=action_client
    )

    executor.start(
        x=3.0,
        y=4.0,
        stamp=Time(),
    )

    action_client.send_future.resolve(
        FakeGoalHandle(
            accepted=True,
            result_future=result_future,
        )
    )

    assert executor.active is True

    result_future.resolve(
        FakeResult(
            status=status,
        )
    )

    completion = executor.wait_for_completion(
        timeout=0.0
    )

    assert completion is not None
    assert completion.accepted is True
    assert completion.status == status
    assert (
        completion.succeeded
        is expected_succeeded
    )

    assert executor.active is False
    assert executor.current_goal is None


def test_wait_for_completion_unblocks_across_threads():
    action_client = FakeActionClient()

    executor = Nav2GoalExecutor(
        action_client=action_client
    )

    executor.start(
        x=1.0,
        y=2.0,
        stamp=Time(),
    )

    completions = []

    wait_thread = Thread(
        target=lambda: completions.append(
            executor.wait_for_completion(
                timeout=1.0
            )
        )
    )

    wait_thread.start()

    action_client.send_future.resolve(
        FakeGoalHandle(
            accepted=False,
        )
    )

    wait_thread.join(
        timeout=1.0
    )

    assert wait_thread.is_alive() is False
    assert len(completions) == 1

    completion = completions[0]

    assert completion is not None
    assert completion.accepted is False
    assert completion.goal_x == pytest.approx(
        1.0
    )
    assert completion.goal_y == pytest.approx(
        2.0
    )


def test_cancel_request_is_false_without_active_goal():
    executor = Nav2GoalExecutor(
        action_client=FakeActionClient()
    )

    assert executor.request_cancel() is False
    assert executor.active is False


def test_accepted_goal_cancel_request_is_idempotent():
    action_client = FakeActionClient()
    result_future = FakeFuture()

    goal_handle = FakeGoalHandle(
        accepted=True,
        result_future=result_future,
    )

    executor = Nav2GoalExecutor(
        action_client=action_client
    )

    executor.start(
        x=1.0,
        y=2.0,
        stamp=Time(),
    )

    action_client.send_future.resolve(
        goal_handle
    )

    assert executor.active is True

    assert executor.request_cancel() is True
    assert executor.request_cancel() is True

    assert goal_handle.cancel_calls == 1
    assert executor.active is True

    result_future.resolve(
        FakeResult(
            status=GoalStatus.STATUS_CANCELED,
        )
    )

    completion = executor.wait_for_completion(
        timeout=0.0
    )

    assert completion is not None
    assert completion.accepted is True
    assert (
        completion.status
        == GoalStatus.STATUS_CANCELED
    )
    assert completion.succeeded is False

    assert executor.active is False
    assert executor.request_cancel() is False


def test_cancel_before_goal_acceptance_is_deferred():
    action_client = FakeActionClient()
    result_future = FakeFuture()

    goal_handle = FakeGoalHandle(
        accepted=True,
        result_future=result_future,
    )

    executor = Nav2GoalExecutor(
        action_client=action_client
    )

    executor.start(
        x=3.0,
        y=4.0,
        stamp=Time(),
    )

    assert executor.request_cancel() is True

    assert goal_handle.cancel_calls == 0
    assert executor.active is True

    action_client.send_future.resolve(
        goal_handle
    )

    assert goal_handle.cancel_calls == 1
    assert executor.active is True

    result_future.resolve(
        FakeResult(
            status=GoalStatus.STATUS_CANCELED,
        )
    )

    completion = executor.wait_for_completion(
        timeout=0.0
    )

    assert completion is not None
    assert (
        completion.status
        == GoalStatus.STATUS_CANCELED
    )
    assert executor.active is False
