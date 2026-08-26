import math

import numpy as np


DEFAULT_MAX_CANDIDATES = 32
CANDIDATE_FEATURE_COUNT = 4


def encode_frontier_observation(
    *,
    candidates,
    robot_x,
    robot_y,
    max_candidates=DEFAULT_MAX_CANDIDATES,
):
    """Encode variable frontier candidates into a fixed-size RL observation."""

    if max_candidates <= 0:
        raise ValueError(
            'max_candidates must be greater than zero.'
        )

    candidate_count = len(candidates)

    if candidate_count > max_candidates:
        raise ValueError(
            f'Frontier candidate count {candidate_count} exceeds '
            f'max_candidates={max_candidates}.'
        )

    features = np.zeros(
        (
            max_candidates,
            CANDIDATE_FEATURE_COUNT,
        ),
        dtype=np.float32,
    )

    action_mask = np.zeros(
        max_candidates,
        dtype=np.int8,
    )

    for index, candidate in enumerate(candidates):
        relative_x = candidate.world_x - robot_x
        relative_y = candidate.world_y - robot_y

        distance = math.hypot(
            relative_x,
            relative_y,
        )

        features[index] = (
            relative_x,
            relative_y,
            distance,
            float(candidate.cluster_size),
        )

        action_mask[index] = 1

    return {
        'candidates': features,
        'action_mask': action_mask,
    }
