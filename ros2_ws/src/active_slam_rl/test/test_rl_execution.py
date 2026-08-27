import pytest
from builtin_interfaces.msg import Time

from active_slam_rl.rl_env import ActiveSlamEnv
from active_slam_rl.rl_execution import (
    RlActionCoordinator,
)
from active_slam_rl.rl_frontier import (
    FrontierCandidate,
)
from active_slam_rl.rl_transition import (
    GoalTransitionTracker,
)


class FakeCompletion:

    def __init__(
        self,
        *,
        succeeded,
    ):
        self.succeeded = succeeded


class FakeNavExecutor:

    def __init__(
        self,
        *,
        error=None,
        completion=None,
    ):
        self.error = error
        self.completion = completion
        self.calls = []

    def start(
        self,
        *,
        x,
        y,
        stamp,
    ):
        if self.error is not None:
            raise self.error

        self.calls.append(
            {
                'x': x,
                'y': y,
                'stamp': stamp,
            }
        )

    def wait_for_completion(
        self,
        timeout=None,
    ):
        return self.completion


@pytest.fixture
def configured_env():
    env = ActiveSlamEnv(
        max_candidates=4,
    )

    candidates = [
        FrontierCandidate(
            cell_x=1,
            cell_y=2,
            world_x=1.25,
            world_y=-0.75,
            cluster_size=5,
        ),
        FrontierCandidate(
            cell_x=3,
            cell_y=4,
            world_x=4.5,
            world_y=2.25,
            cluster_size=8,
        ),
    ]

    env.set_frontier_state(
        candidates=candidates,
        robot_x=0.0,
        robot_y=0.0,
    )

    try:
        yield env, candidates

    finally:
        env.close()


def test_start_action_freezes_exact_candidate_slot(
    configured_env,
):
    env, candidates = configured_env

    tracker = GoalTransitionTracker()
    nav_executor = FakeNavExecutor()

    coordinator = RlActionCoordinator(
        env=env,
        transition_tracker=tracker,
        nav_executor=nav_executor,
        visited_goals=[],
    )

    stamp = Time(
        sec=12,
        nanosec=34,
    )

    start = coordinator.start_action(
        action=1,
        area_m2=7.5,
        path_m=3.25,
        stamp=stamp,
    )

    assert start.action == 1

    assert start.goal_x == pytest.approx(
        candidates[1].world_x
    )

    assert start.goal_y == pytest.approx(
        candidates[1].world_y
    )

    assert start.area_m2 == pytest.approx(
        7.5
    )

    assert start.path_m == pytest.approx(
        3.25
    )

    assert tracker.active is True
    assert tracker.start is start

    assert nav_executor.calls == [
        {
            'x': candidates[1].world_x,
            'y': candidates[1].world_y,
            'stamp': stamp,
        }
    ]


def test_candidate_snapshot_survives_later_env_update(
    configured_env,
):
    env, candidates = configured_env

    tracker = GoalTransitionTracker()

    coordinator = RlActionCoordinator(
        env=env,
        transition_tracker=tracker,
        nav_executor=FakeNavExecutor(),
        visited_goals=[],
    )

    start = coordinator.start_action(
        action=0,
        area_m2=2.0,
        path_m=1.0,
        stamp=Time(),
    )

    env.set_frontier_state(
        candidates=[
            FrontierCandidate(
                cell_x=9,
                cell_y=9,
                world_x=99.0,
                world_y=100.0,
                cluster_size=20,
            )
        ],
        robot_x=0.0,
        robot_y=0.0,
    )

    assert start.goal_x == pytest.approx(
        candidates[0].world_x
    )

    assert start.goal_y == pytest.approx(
        candidates[0].world_y
    )


def test_unavailable_action_never_starts_transition_or_nav(
    configured_env,
):
    env, _ = configured_env

    tracker = GoalTransitionTracker()
    nav_executor = FakeNavExecutor()

    coordinator = RlActionCoordinator(
        env=env,
        transition_tracker=tracker,
        nav_executor=nav_executor,
        visited_goals=[],
    )

    with pytest.raises(
        ValueError,
        match='unavailable frontier candidate slot',
    ):
        coordinator.start_action(
            action=2,
            area_m2=1.0,
            path_m=0.0,
            stamp=Time(),
        )

    assert tracker.active is False
    assert nav_executor.calls == []


def test_nav_start_failure_rolls_back_transition(
    configured_env,
):
    env, _ = configured_env

    tracker = GoalTransitionTracker()

    coordinator = RlActionCoordinator(
        env=env,
        transition_tracker=tracker,
        nav_executor=FakeNavExecutor(
            error=RuntimeError(
                'Nav2 action server is not ready.'
            )
        ),
        visited_goals=[],
    )

    with pytest.raises(
        RuntimeError,
        match='Nav2 action server is not ready',
    ):
        coordinator.start_action(
            action=0,
            area_m2=4.0,
            path_m=2.0,
            stamp=Time(),
        )

    assert tracker.active is False
    assert tracker.start is None


def test_overlapping_action_is_rejected(
    configured_env,
):
    env, _ = configured_env

    tracker = GoalTransitionTracker()
    nav_executor = FakeNavExecutor()

    coordinator = RlActionCoordinator(
        env=env,
        transition_tracker=tracker,
        nav_executor=nav_executor,
        visited_goals=[],
    )

    coordinator.start_action(
        action=0,
        area_m2=1.0,
        path_m=0.0,
        stamp=Time(),
    )

    with pytest.raises(
        RuntimeError,
        match='already active',
    ):
        coordinator.start_action(
            action=1,
            area_m2=2.0,
            path_m=1.0,
            stamp=Time(),
        )

    assert len(nav_executor.calls) == 1


def test_complete_action_computes_goal_aligned_transition(
    configured_env,
):
    env, candidates = configured_env

    tracker = GoalTransitionTracker()
    visited_goals = []

    coordinator = RlActionCoordinator(
        env=env,
        transition_tracker=tracker,
        nav_executor=FakeNavExecutor(
            completion=FakeCompletion(
                succeeded=True,
            )
        ),
        visited_goals=visited_goals,
    )

    coordinator.start_action(
        action=1,
        area_m2=5.0,
        path_m=2.0,
        stamp=Time(),
    )

    outcome = coordinator.complete_action(
        end_area_m2=6.5,
        end_path_m=3.0,
        timeout=0.0,
    )

    assert outcome is not None

    assert (
        outcome.transition.start.action
        == 1
    )

    assert (
        outcome.transition.metrics.area_gain_m2
        == pytest.approx(1.5)
    )

    assert (
        outcome.transition.metrics.path_delta_m
        == pytest.approx(1.0)
    )

    assert (
        outcome.transition.metrics.reward
        == pytest.approx(1.4)
    )

    assert visited_goals == [
        (
            candidates[1].world_x,
            candidates[1].world_y,
        )
    ]

    assert tracker.active is False


def test_failed_navigation_is_not_recorded_visited(
    configured_env,
):
    env, _ = configured_env

    tracker = GoalTransitionTracker()
    visited_goals = []

    coordinator = RlActionCoordinator(
        env=env,
        transition_tracker=tracker,
        nav_executor=FakeNavExecutor(
            completion=FakeCompletion(
                succeeded=False,
            )
        ),
        visited_goals=visited_goals,
    )

    coordinator.start_action(
        action=0,
        area_m2=2.0,
        path_m=1.0,
        stamp=Time(),
    )

    outcome = coordinator.complete_action(
        end_area_m2=2.5,
        end_path_m=1.5,
        timeout=0.0,
    )

    assert outcome is not None

    assert (
        outcome.transition.metrics.reward
        == pytest.approx(0.45)
    )

    assert visited_goals == []
    assert tracker.active is False


def test_completion_timeout_preserves_active_transition(
    configured_env,
):
    env, _ = configured_env

    tracker = GoalTransitionTracker()

    coordinator = RlActionCoordinator(
        env=env,
        transition_tracker=tracker,
        nav_executor=FakeNavExecutor(
            completion=None,
        ),
        visited_goals=[],
    )

    start = coordinator.start_action(
        action=0,
        area_m2=3.0,
        path_m=1.0,
        stamp=Time(),
    )

    outcome = coordinator.complete_action(
        end_area_m2=4.0,
        end_path_m=2.0,
        timeout=0.0,
    )

    assert outcome is None
    assert tracker.active is True
    assert tracker.start is start


def test_complete_action_requires_active_transition(
    configured_env,
):
    env, _ = configured_env

    coordinator = RlActionCoordinator(
        env=env,
        transition_tracker=GoalTransitionTracker(),
        nav_executor=FakeNavExecutor(),
        visited_goals=[],
    )

    with pytest.raises(
        RuntimeError,
        match='No RL action transition is active',
    ):
        coordinator.complete_action(
            end_area_m2=1.0,
            end_path_m=0.0,
            timeout=0.0,
        )
