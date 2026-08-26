import pytest

from active_slam_rl.rl_transition import (
    GoalTransitionTracker,
    TransitionMetrics,
    TransitionStart,
    compute_transition_metrics,
)


def test_transition_computes_goal_aligned_deltas():
    metrics = compute_transition_metrics(
        start_area_m2=10.0,
        end_area_m2=11.5,
        start_path_m=5.0,
        end_path_m=7.0,
    )

    assert isinstance(
        metrics,
        TransitionMetrics,
    )

    assert metrics.area_gain_m2 == pytest.approx(
        1.5
    )

    assert metrics.path_delta_m == pytest.approx(
        2.0
    )

    assert metrics.reward == pytest.approx(
        1.3
    )


def test_area_regression_is_clamped_to_zero_gain():
    metrics = compute_transition_metrics(
        start_area_m2=10.0,
        end_area_m2=9.8,
        start_path_m=2.0,
        end_path_m=3.0,
    )

    assert metrics.area_gain_m2 == pytest.approx(
        0.0
    )

    assert metrics.path_delta_m == pytest.approx(
        1.0
    )

    assert metrics.reward == pytest.approx(
        -0.1
    )


def test_unchanged_measurements_produce_zero_reward():
    metrics = compute_transition_metrics(
        start_area_m2=4.0,
        end_area_m2=4.0,
        start_path_m=1.0,
        end_path_m=1.0,
    )

    assert metrics.area_gain_m2 == pytest.approx(
        0.0
    )

    assert metrics.path_delta_m == pytest.approx(
        0.0
    )

    assert metrics.reward == pytest.approx(
        0.0
    )


def test_path_counter_regression_is_rejected():
    with pytest.raises(
        ValueError,
        match='end_path_m',
    ):
        compute_transition_metrics(
            start_area_m2=1.0,
            end_area_m2=2.0,
            start_path_m=5.0,
            end_path_m=4.0,
        )


@pytest.mark.parametrize(
    (
        'field',
        'kwargs',
    ),
    [
        (
            'start_area_m2',
            {
                'start_area_m2': -0.1,
                'end_area_m2': 1.0,
                'start_path_m': 0.0,
                'end_path_m': 0.0,
            },
        ),
        (
            'end_area_m2',
            {
                'start_area_m2': 0.0,
                'end_area_m2': -0.1,
                'start_path_m': 0.0,
                'end_path_m': 0.0,
            },
        ),
        (
            'start_path_m',
            {
                'start_area_m2': 0.0,
                'end_area_m2': 1.0,
                'start_path_m': -0.1,
                'end_path_m': 0.0,
            },
        ),
        (
            'end_path_m',
            {
                'start_area_m2': 0.0,
                'end_area_m2': 1.0,
                'start_path_m': 0.0,
                'end_path_m': -0.1,
            },
        ),
    ],
)
def test_negative_measurements_are_rejected(
    field,
    kwargs,
):
    with pytest.raises(
        ValueError,
        match=field,
    ):
        compute_transition_metrics(
            **kwargs
        )


def test_goal_transition_begin_freezes_start_state():
    tracker = GoalTransitionTracker()

    start = tracker.begin(
        action=3,
        goal_x=1.25,
        goal_y=-0.5,
        area_m2=4.0,
        path_m=2.0,
    )

    assert tracker.active is True
    assert tracker.start is start

    assert isinstance(
        start,
        TransitionStart,
    )

    assert start.action == 3

    assert start.goal_x == pytest.approx(
        1.25
    )

    assert start.goal_y == pytest.approx(
        -0.5
    )

    assert start.area_m2 == pytest.approx(
        4.0
    )

    assert start.path_m == pytest.approx(
        2.0
    )


def test_goal_transition_rejects_overlapping_begin():
    tracker = GoalTransitionTracker()

    tracker.begin(
        action=0,
        goal_x=1.0,
        goal_y=2.0,
        area_m2=3.0,
        path_m=4.0,
    )

    with pytest.raises(
        RuntimeError,
        match='already active',
    ):
        tracker.begin(
            action=1,
            goal_x=2.0,
            goal_y=3.0,
            area_m2=4.0,
            path_m=5.0,
        )


def test_goal_transition_complete_uses_frozen_start():
    tracker = GoalTransitionTracker()

    tracker.begin(
        action=2,
        goal_x=3.0,
        goal_y=4.0,
        area_m2=5.0,
        path_m=2.0,
    )

    completed = tracker.complete(
        end_area_m2=5.5,
        end_path_m=3.0,
    )

    assert completed.start.action == 2

    assert completed.start.goal_x == pytest.approx(
        3.0
    )

    assert completed.start.goal_y == pytest.approx(
        4.0
    )

    assert completed.metrics.area_gain_m2 == pytest.approx(
        0.5
    )

    assert completed.metrics.path_delta_m == pytest.approx(
        1.0
    )

    assert completed.metrics.reward == pytest.approx(
        0.4
    )

    assert tracker.active is False
    assert tracker.start is None


def test_goal_transition_complete_requires_active_transition():
    tracker = GoalTransitionTracker()

    with pytest.raises(
        RuntimeError,
        match='No transition is active',
    ):
        tracker.complete(
            end_area_m2=1.0,
            end_path_m=1.0,
        )


def test_invalid_completion_preserves_active_transition():
    tracker = GoalTransitionTracker()

    start = tracker.begin(
        action=0,
        goal_x=1.0,
        goal_y=1.0,
        area_m2=2.0,
        path_m=5.0,
    )

    with pytest.raises(
        ValueError,
        match='end_path_m',
    ):
        tracker.complete(
            end_area_m2=3.0,
            end_path_m=4.0,
        )

    assert tracker.active is True
    assert tracker.start is start


def test_goal_transition_reset_discards_active_transition():
    tracker = GoalTransitionTracker()

    tracker.begin(
        action=0,
        goal_x=1.0,
        goal_y=1.0,
        area_m2=2.0,
        path_m=3.0,
    )

    tracker.reset()

    assert tracker.active is False
    assert tracker.start is None
