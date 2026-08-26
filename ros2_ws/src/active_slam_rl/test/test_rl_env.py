import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from active_slam_rl.rl_env import ActiveSlamEnv
from active_slam_rl.rl_observation import (
    CANDIDATE_FEATURE_COUNT,
    DEFAULT_MAX_CANDIDATES,
)


def test_environment_passes_gymnasium_checker():
    env = ActiveSlamEnv()

    check_env(
        env,
        skip_render_check=True,
    )

    env.close()


def test_environment_uses_frontier_contract():
    env = ActiveSlamEnv()

    observation, info = env.reset(seed=123)

    assert env.action_space.n == (
        DEFAULT_MAX_CANDIDATES
    )

    assert observation['candidates'].shape == (
        DEFAULT_MAX_CANDIDATES,
        CANDIDATE_FEATURE_COUNT,
    )

    assert observation['candidates'].dtype == (
        np.float32
    )

    assert observation['action_mask'].shape == (
        DEFAULT_MAX_CANDIDATES,
    )

    assert observation['action_mask'].dtype == (
        np.int8
    )

    assert env.observation_space.contains(
        observation
    )

    assert info == {}

    env.close()


def test_custom_candidate_capacity_changes_spaces():
    env = ActiveSlamEnv(
        max_candidates=8,
    )

    observation, _ = env.reset()

    assert env.action_space.n == 8
    assert observation['candidates'].shape == (
        8,
        CANDIDATE_FEATURE_COUNT,
    )
    assert observation['action_mask'].shape == (8,)
    assert env.observation_space.contains(
        observation
    )

    env.close()


def test_invalid_action_is_rejected():
    env = ActiveSlamEnv(
        max_candidates=4,
    )

    env.reset()

    with pytest.raises(
        ValueError,
        match='Invalid action 4',
    ):
        env.step(4)

    env.close()


def test_nonpositive_candidate_capacity_is_rejected():
    with pytest.raises(
        ValueError,
        match='max_candidates must be greater than zero',
    ):
        ActiveSlamEnv(
            max_candidates=0,
        )
