DEFAULT_PATH_COST_PER_M = 0.10


def compute_exploration_reward(
    *,
    area_gain_m2,
    path_delta_m,
    path_cost_per_m=DEFAULT_PATH_COST_PER_M,
):
    """Return area-gain reward minus physical path cost."""

    if area_gain_m2 < 0.0:
        raise ValueError(
            'area_gain_m2 must be nonnegative.'
        )

    if path_delta_m < 0.0:
        raise ValueError(
            'path_delta_m must be nonnegative.'
        )

    if path_cost_per_m < 0.0:
        raise ValueError(
            'path_cost_per_m must be nonnegative.'
        )

    return (
        float(area_gain_m2)
        - float(path_cost_per_m)
        * float(path_delta_m)
    )
