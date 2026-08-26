import pytest

from active_slam_rl.rl_transition import (
    TransitionMetrics,
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
