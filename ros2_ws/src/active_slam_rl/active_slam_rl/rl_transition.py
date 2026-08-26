from dataclasses import dataclass

from active_slam_rl.rl_reward import (
    DEFAULT_PATH_COST_PER_M,
    compute_exploration_reward,
)


@dataclass(frozen=True)
class TransitionMetrics:
    """Goal-aligned physical measurements for one RL transition."""

    area_gain_m2: float
    path_delta_m: float
    reward: float


def compute_transition_metrics(
    *,
    start_area_m2,
    end_area_m2,
    start_path_m,
    end_path_m,
    path_cost_per_m=DEFAULT_PATH_COST_PER_M,
):
    """Compute one goal-aligned exploration transition."""

    values = {
        'start_area_m2': start_area_m2,
        'end_area_m2': end_area_m2,
        'start_path_m': start_path_m,
        'end_path_m': end_path_m,
    }

    for name, value in values.items():
        if value < 0.0:
            raise ValueError(
                f'{name} must be nonnegative.'
            )

    if end_path_m < start_path_m:
        raise ValueError(
            'end_path_m must not be less than start_path_m.'
        )

    area_gain_m2 = max(
        0.0,
        float(end_area_m2)
        - float(start_area_m2),
    )

    path_delta_m = (
        float(end_path_m)
        - float(start_path_m)
    )

    reward = compute_exploration_reward(
        area_gain_m2=area_gain_m2,
        path_delta_m=path_delta_m,
        path_cost_per_m=path_cost_per_m,
    )

    return TransitionMetrics(
        area_gain_m2=area_gain_m2,
        path_delta_m=path_delta_m,
        reward=reward,
    )
