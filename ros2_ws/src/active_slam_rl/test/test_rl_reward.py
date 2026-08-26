import pytest

from active_slam_rl.rl_reward import (
    DEFAULT_PATH_COST_PER_M,
    compute_exploration_reward,
)


def test_default_path_cost_is_explicit():
    assert DEFAULT_PATH_COST_PER_M == pytest.approx(
        0.10
    )


def test_reward_combines_area_gain_and_path_cost():
    reward = compute_exploration_reward(
        area_gain_m2=0.745,
        path_delta_m=1.311948,
    )

    assert reward == pytest.approx(
        0.745 - 0.10 * 1.311948
    )


def test_zero_area_gain_still_penalizes_motion():
    reward = compute_exploration_reward(
        area_gain_m2=0.0,
        path_delta_m=1.5,
    )

    assert reward == pytest.approx(
        -0.15
    )


def test_zero_motion_preserves_area_gain():
    reward = compute_exploration_reward(
        area_gain_m2=2.5,
        path_delta_m=0.0,
    )

    assert reward == pytest.approx(
        2.5
    )


def test_custom_path_cost_is_supported():
    reward = compute_exploration_reward(
        area_gain_m2=1.0,
        path_delta_m=2.0,
        path_cost_per_m=0.25,
    )

    assert reward == pytest.approx(
        0.5
    )


@pytest.mark.parametrize(
    (
        'area_gain_m2',
        'path_delta_m',
        'path_cost_per_m',
        'message',
    ),
    [
        (
            -0.01,
            0.0,
            0.10,
            'area_gain_m2',
        ),
        (
            0.0,
            -0.01,
            0.10,
            'path_delta_m',
        ),
        (
            0.0,
            0.0,
            -0.01,
            'path_cost_per_m',
        ),
    ],
)
def test_negative_inputs_are_rejected(
    area_gain_m2,
    path_delta_m,
    path_cost_per_m,
    message,
):
    with pytest.raises(
        ValueError,
        match=message,
    ):
        compute_exploration_reward(
            area_gain_m2=area_gain_m2,
            path_delta_m=path_delta_m,
            path_cost_per_m=path_cost_per_m,
        )
