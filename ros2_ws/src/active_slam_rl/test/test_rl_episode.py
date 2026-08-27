import pytest

from active_slam_rl.rl_episode import (
    DEFAULT_EPISODE_HORIZON_S,
    EpisodeTimeLimit,
)


class FakeClock:

    def __init__(
        self,
        value=0.0,
    ):
        self.value = float(
            value
        )

    def __call__(self):
        return self.value


def test_default_episode_horizon_is_300_seconds():
    limit = EpisodeTimeLimit()

    assert DEFAULT_EPISODE_HORIZON_S == pytest.approx(
        300.0
    )

    assert limit.horizon_s == pytest.approx(
        300.0
    )


def test_nonpositive_episode_horizon_is_rejected():
    with pytest.raises(
        ValueError,
        match='horizon_s must be greater than zero',
    ):
        EpisodeTimeLimit(
            horizon_s=0.0,
        )


def test_episode_clock_is_inactive_before_first_action():
    clock = FakeClock(
        value=100.0,
    )

    limit = EpisodeTimeLimit(
        clock=clock,
    )

    assert limit.active is False
    assert limit.start_time is None
    assert limit.elapsed_s == pytest.approx(
        0.0
    )
    assert limit.truncated is False


def test_episode_start_is_idempotent():
    clock = FakeClock(
        value=10.0,
    )

    limit = EpisodeTimeLimit(
        clock=clock,
    )

    first_start = limit.start()

    clock.value = 25.0

    second_start = limit.start()

    assert first_start == pytest.approx(
        10.0
    )

    assert second_start == pytest.approx(
        10.0
    )

    assert limit.elapsed_s == pytest.approx(
        15.0
    )


def test_episode_truncates_at_300_seconds_not_before():
    clock = FakeClock(
        value=50.0,
    )

    limit = EpisodeTimeLimit(
        clock=clock,
    )

    limit.start()

    clock.value = 349.999

    assert limit.elapsed_s == pytest.approx(
        299.999
    )
    assert limit.truncated is False

    clock.value = 350.0

    assert limit.elapsed_s == pytest.approx(
        300.0
    )
    assert limit.truncated is True


def test_episode_reset_rearms_time_limit():
    clock = FakeClock(
        value=5.0,
    )

    limit = EpisodeTimeLimit(
        clock=clock,
    )

    limit.start()

    clock.value = 305.0

    assert limit.truncated is True

    limit.reset()

    assert limit.active is False
    assert limit.start_time is None
    assert limit.elapsed_s == pytest.approx(
        0.0
    )
    assert limit.truncated is False

    clock.value = 500.0

    assert limit.start() == pytest.approx(
        500.0
    )


def test_remaining_budget_is_full_before_episode_start():
    clock = FakeClock(
        value=500.0,
    )

    limit = EpisodeTimeLimit(
        horizon_s=300.0,
        clock=clock,
    )

    assert limit.active is False

    assert limit.remaining_s == pytest.approx(
        300.0
    )


def test_remaining_budget_tracks_elapsed_episode_time():
    clock = FakeClock(
        value=100.0,
    )

    limit = EpisodeTimeLimit(
        horizon_s=300.0,
        clock=clock,
    )

    limit.start()

    clock.value = 225.5

    assert limit.elapsed_s == pytest.approx(
        125.5
    )

    assert limit.remaining_s == pytest.approx(
        174.5
    )

    assert limit.truncated is False


def test_remaining_budget_clamps_to_zero_at_and_after_horizon():
    clock = FakeClock(
        value=20.0,
    )

    limit = EpisodeTimeLimit(
        horizon_s=300.0,
        clock=clock,
    )

    limit.start()

    clock.value = 320.0

    assert limit.remaining_s == pytest.approx(
        0.0
    )
    assert limit.truncated is True

    clock.value = 350.0

    assert limit.remaining_s == pytest.approx(
        0.0
    )
    assert limit.truncated is True
