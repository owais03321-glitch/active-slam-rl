import numpy as np
import pytest

from active_slam_rl.rl_frontier import FrontierCandidate
from active_slam_rl.rl_observation import (
    CANDIDATE_FEATURE_COUNT,
    DEFAULT_MAX_CANDIDATES,
    encode_frontier_observation,
)


def make_candidate(
    *,
    cell_x,
    cell_y,
    world_x,
    world_y,
    cluster_size,
):
    return FrontierCandidate(
        cell_x=cell_x,
        cell_y=cell_y,
        world_x=world_x,
        world_y=world_y,
        cluster_size=cluster_size,
    )


def test_encodes_relative_candidate_features():
    candidates = [
        make_candidate(
            cell_x=10,
            cell_y=20,
            world_x=2.0,
            world_y=3.0,
            cluster_size=7,
        ),
        make_candidate(
            cell_x=30,
            cell_y=40,
            world_x=-1.0,
            world_y=1.0,
            cluster_size=12,
        ),
    ]

    observation = encode_frontier_observation(
        candidates=candidates,
        robot_x=1.0,
        robot_y=1.0,
    )

    features = observation['candidates']
    mask = observation['action_mask']

    assert features.shape == (
        DEFAULT_MAX_CANDIDATES,
        CANDIDATE_FEATURE_COUNT,
    )
    assert features.dtype == np.float32

    assert mask.shape == (DEFAULT_MAX_CANDIDATES,)
    assert mask.dtype == np.int8

    np.testing.assert_allclose(
        features[0],
        [
            1.0,
            2.0,
            np.sqrt(5.0),
            7.0,
        ],
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        features[1],
        [
            -2.0,
            0.0,
            2.0,
            12.0,
        ],
        rtol=1e-6,
    )

    assert mask[0] == 1
    assert mask[1] == 1


def test_unused_candidate_slots_are_zero_padded():
    candidate = make_candidate(
        cell_x=1,
        cell_y=1,
        world_x=0.5,
        world_y=0.5,
        cluster_size=5,
    )

    observation = encode_frontier_observation(
        candidates=[candidate],
        robot_x=0.0,
        robot_y=0.0,
        max_candidates=4,
    )

    features = observation['candidates']
    mask = observation['action_mask']

    np.testing.assert_array_equal(
        features[1:],
        np.zeros(
            (3, CANDIDATE_FEATURE_COUNT),
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        mask,
        np.array(
            [1, 0, 0, 0],
            dtype=np.int8,
        ),
    )


def test_empty_candidate_list_produces_empty_mask():
    observation = encode_frontier_observation(
        candidates=[],
        robot_x=0.0,
        robot_y=0.0,
        max_candidates=4,
    )

    np.testing.assert_array_equal(
        observation['candidates'],
        np.zeros(
            (4, CANDIDATE_FEATURE_COUNT),
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        observation['action_mask'],
        np.zeros(
            4,
            dtype=np.int8,
        ),
    )


def test_candidate_overflow_fails_loudly():
    candidates = [
        make_candidate(
            cell_x=index,
            cell_y=0,
            world_x=float(index),
            world_y=0.0,
            cluster_size=5,
        )
        for index in range(3)
    ]

    with pytest.raises(
        ValueError,
        match=(
            'Frontier candidate count 3 exceeds '
            'max_candidates=2'
        ),
    ):
        encode_frontier_observation(
            candidates=candidates,
            robot_x=0.0,
            robot_y=0.0,
            max_candidates=2,
        )


def test_rejects_nonpositive_candidate_capacity():
    with pytest.raises(
        ValueError,
        match='max_candidates must be greater than zero',
    ):
        encode_frontier_observation(
            candidates=[],
            robot_x=0.0,
            robot_y=0.0,
            max_candidates=0,
        )
