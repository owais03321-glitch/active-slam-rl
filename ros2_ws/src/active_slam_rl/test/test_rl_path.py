import pytest

from active_slam_rl.rl_path import (
    DEFAULT_MAX_ODOM_STEP_M,
    PathLengthTracker,
)


def test_default_max_odom_step_matches_baseline():
    assert DEFAULT_MAX_ODOM_STEP_M == pytest.approx(
        1.0
    )


def test_first_position_establishes_origin():
    tracker = PathLengthTracker()

    counted = tracker.update(
        x=2.0,
        y=3.0,
    )

    assert counted == pytest.approx(
        0.0
    )

    assert tracker.path_length_m == pytest.approx(
        0.0
    )


def test_tracker_accumulates_consecutive_motion():
    tracker = PathLengthTracker()

    tracker.update(
        x=0.0,
        y=0.0,
    )

    first = tracker.update(
        x=0.3,
        y=0.4,
    )

    second = tracker.update(
        x=0.6,
        y=0.8,
    )

    assert first == pytest.approx(
        0.5
    )

    assert second == pytest.approx(
        0.5
    )

    assert tracker.path_length_m == pytest.approx(
        1.0
    )


def test_step_exactly_at_limit_is_counted():
    tracker = PathLengthTracker(
        max_step_m=1.0,
    )

    tracker.update(
        x=0.0,
        y=0.0,
    )

    counted = tracker.update(
        x=1.0,
        y=0.0,
    )

    assert counted == pytest.approx(
        1.0
    )

    assert tracker.path_length_m == pytest.approx(
        1.0
    )


def test_large_jump_is_ignored_and_reanchors():
    tracker = PathLengthTracker(
        max_step_m=1.0,
    )

    tracker.update(
        x=0.0,
        y=0.0,
    )

    ignored = tracker.update(
        x=10.0,
        y=0.0,
    )

    counted = tracker.update(
        x=10.3,
        y=0.4,
    )

    assert ignored == pytest.approx(
        0.0
    )

    assert counted == pytest.approx(
        0.5
    )

    assert tracker.path_length_m == pytest.approx(
        0.5
    )


def test_reset_clears_path_and_origin():
    tracker = PathLengthTracker()

    tracker.update(
        x=0.0,
        y=0.0,
    )

    tracker.update(
        x=0.3,
        y=0.4,
    )

    tracker.reset()

    assert tracker.last_xy is None

    assert tracker.path_length_m == pytest.approx(
        0.0
    )

    counted = tracker.update(
        x=5.0,
        y=5.0,
    )

    assert counted == pytest.approx(
        0.0
    )


def test_nonpositive_max_step_is_rejected():
    with pytest.raises(
        ValueError,
        match='max_step_m',
    ):
        PathLengthTracker(
            max_step_m=0.0,
        )
