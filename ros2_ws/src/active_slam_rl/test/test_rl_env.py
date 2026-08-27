import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from active_slam_rl.rl_env import ActiveSlamEnv
from active_slam_rl.rl_observation import (
    CANDIDATE_FEATURE_COUNT,
    DEFAULT_MAX_CANDIDATES,
)


class FakeStepBridge:

    def __init__(
        self,
        *,
        env,
        result=None,
    ):
        self.env = env
        self.result = result
        self.calls = []

    def step(
        self,
        action,
    ):
        self.calls.append(
            action
        )

        if self.result is not None:
            return self.result

        return (
            self.env._copy_observation(),
            0.0,
            False,
            False,
            {},
        )


def test_environment_passes_gymnasium_checker():
    env = ActiveSlamEnv()

    env.bind_step_bridge(
        FakeStepBridge(
            env=env,
        )
    )

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


def test_valid_step_requires_synchronous_bridge():
    env = ActiveSlamEnv(
        max_candidates=4,
    )

    env.reset()

    with pytest.raises(
        RuntimeError,
        match='no synchronous step bridge',
    ):
        env.step(0)

    env.close()


def test_step_delegates_to_bound_bridge():
    env = ActiveSlamEnv(
        max_candidates=4,
    )

    env.reset()

    expected = (
        env._copy_observation(),
        1.25,
        False,
        False,
        {
            'source': 'fake',
        },
    )

    bridge = FakeStepBridge(
        env=env,
        result=expected,
    )

    env.bind_step_bridge(
        bridge
    )

    result = env.step(2)

    assert result is expected
    assert bridge.calls == [2]
    assert env.step_bridge is bridge

    env.close()


def test_bind_step_bridge_rejects_invalid_interface():
    env = ActiveSlamEnv()

    with pytest.raises(
        TypeError,
        match='callable step',
    ):
        env.bind_step_bridge(
            object()
        )

    env.close()


def test_nonpositive_candidate_capacity_is_rejected():
    with pytest.raises(
        ValueError,
        match='max_candidates must be greater than zero',
    ):
        ActiveSlamEnv(
            max_candidates=0,
        )


def test_frontier_state_updates_environment_observation():
    from active_slam_rl.rl_frontier import FrontierCandidate

    env = ActiveSlamEnv(
        max_candidates=4,
    )

    candidates = [
        FrontierCandidate(
            cell_x=10,
            cell_y=20,
            world_x=3.0,
            world_y=4.0,
            cluster_size=8,
        ),
        FrontierCandidate(
            cell_x=30,
            cell_y=40,
            world_x=0.0,
            world_y=2.0,
            cluster_size=11,
        ),
    ]

    observation = env.set_frontier_state(
        candidates=candidates,
        robot_x=1.0,
        robot_y=1.0,
    )

    np.testing.assert_allclose(
        observation['candidates'][0],
        [
            2.0,
            3.0,
            np.sqrt(13.0),
            8.0,
        ],
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        observation['candidates'][1],
        [
            -1.0,
            1.0,
            np.sqrt(2.0),
            11.0,
        ],
        rtol=1e-6,
    )

    np.testing.assert_array_equal(
        observation['action_mask'],
        np.array(
            [1, 1, 0, 0],
            dtype=np.int8,
        ),
    )

    assert env.observation_space.contains(
        observation
    )

    env.close()


def test_frontier_state_return_is_independent_copy():
    from active_slam_rl.rl_frontier import FrontierCandidate

    env = ActiveSlamEnv(
        max_candidates=2,
    )

    candidate = FrontierCandidate(
        cell_x=1,
        cell_y=1,
        world_x=1.0,
        world_y=0.0,
        cluster_size=5,
    )

    observation = env.set_frontier_state(
        candidates=[candidate],
        robot_x=0.0,
        robot_y=0.0,
    )

    observation['action_mask'][0] = 0
    observation['candidates'][0, 0] = 999.0

    internal_copy = env._copy_observation()

    assert internal_copy['action_mask'][0] == 1
    assert internal_copy['candidates'][0, 0] == pytest.approx(1.0)

    env.close()


def test_frontier_state_propagates_candidate_overflow():
    from active_slam_rl.rl_frontier import FrontierCandidate

    env = ActiveSlamEnv(
        max_candidates=2,
    )

    candidates = [
        FrontierCandidate(
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
        env.set_frontier_state(
            candidates=candidates,
            robot_x=0.0,
            robot_y=0.0,
        )

    env.close()


def test_action_resolves_to_matching_frontier_candidate():
    from active_slam_rl.rl_frontier import FrontierCandidate

    env = ActiveSlamEnv(
        max_candidates=4,
    )

    candidates = [
        FrontierCandidate(
            cell_x=2,
            cell_y=3,
            world_x=1.0,
            world_y=2.0,
            cluster_size=5,
        ),
        FrontierCandidate(
            cell_x=7,
            cell_y=8,
            world_x=4.0,
            world_y=5.0,
            cluster_size=9,
        ),
    ]

    env.set_frontier_state(
        candidates=candidates,
        robot_x=0.0,
        robot_y=0.0,
    )

    assert env.candidate_for_action(0) == candidates[0]
    assert env.candidate_for_action(1) == candidates[1]

    env.close()


def test_action_rejects_unused_candidate_slot():
    from active_slam_rl.rl_frontier import FrontierCandidate

    env = ActiveSlamEnv(
        max_candidates=4,
    )

    env.set_frontier_state(
        candidates=[
            FrontierCandidate(
                cell_x=2,
                cell_y=3,
                world_x=1.0,
                world_y=2.0,
                cluster_size=5,
            ),
        ],
        robot_x=0.0,
        robot_y=0.0,
    )

    with pytest.raises(
        ValueError,
        match=(
            'Action 1 selects an unavailable '
            'frontier candidate slot'
        ),
    ):
        env.candidate_for_action(1)

    env.close()


def test_reset_clears_action_candidate_mapping():
    from active_slam_rl.rl_frontier import FrontierCandidate

    env = ActiveSlamEnv(
        max_candidates=2,
    )

    env.set_frontier_state(
        candidates=[
            FrontierCandidate(
                cell_x=2,
                cell_y=3,
                world_x=1.0,
                world_y=2.0,
                cluster_size=5,
            ),
        ],
        robot_x=0.0,
        robot_y=0.0,
    )

    env.reset()

    with pytest.raises(
        ValueError,
        match=(
            'Action 0 selects an unavailable '
            'frontier candidate slot'
        ),
    ):
        env.candidate_for_action(0)

    env.close()


def test_action_candidate_mapping_copies_input_sequence():
    from active_slam_rl.rl_frontier import FrontierCandidate

    env = ActiveSlamEnv(
        max_candidates=2,
    )

    original = FrontierCandidate(
        cell_x=2,
        cell_y=3,
        world_x=1.0,
        world_y=2.0,
        cluster_size=5,
    )

    candidates = [original]

    env.set_frontier_state(
        candidates=candidates,
        robot_x=0.0,
        robot_y=0.0,
    )

    candidates.clear()

    assert env.candidate_for_action(0) == original

    env.close()


def test_environment_exposes_sb3_action_masks():
    from sb3_contrib.common.maskable.utils import (
        get_action_masks,
        is_masking_supported,
    )

    from active_slam_rl.rl_frontier import FrontierCandidate

    env = ActiveSlamEnv(
        max_candidates=4,
    )

    env.set_frontier_state(
        candidates=[
            FrontierCandidate(
                cell_x=1,
                cell_y=1,
                world_x=1.0,
                world_y=0.0,
                cluster_size=5,
            ),
            FrontierCandidate(
                cell_x=2,
                cell_y=1,
                world_x=2.0,
                world_y=0.0,
                cluster_size=6,
            ),
        ],
        robot_x=0.0,
        robot_y=0.0,
    )

    assert is_masking_supported(env)

    masks = get_action_masks(env)

    assert masks.dtype == np.bool_

    np.testing.assert_array_equal(
        masks,
        np.array(
            [
                True,
                True,
                False,
                False,
            ],
            dtype=bool,
        ),
    )

    env.close()


def test_action_masks_returns_independent_copy():
    from active_slam_rl.rl_frontier import FrontierCandidate

    env = ActiveSlamEnv(
        max_candidates=2,
    )

    env.set_frontier_state(
        candidates=[
            FrontierCandidate(
                cell_x=1,
                cell_y=1,
                world_x=1.0,
                world_y=0.0,
                cluster_size=5,
            ),
        ],
        robot_x=0.0,
        robot_y=0.0,
    )

    masks = env.action_masks()

    masks[0] = False

    assert env.action_masks().tolist() == [
        True,
        False,
    ]

    env.close()


def test_reset_clears_sb3_action_masks():
    from active_slam_rl.rl_frontier import FrontierCandidate

    env = ActiveSlamEnv(
        max_candidates=2,
    )

    env.set_frontier_state(
        candidates=[
            FrontierCandidate(
                cell_x=1,
                cell_y=1,
                world_x=1.0,
                world_y=0.0,
                cluster_size=5,
            ),
        ],
        robot_x=0.0,
        robot_y=0.0,
    )

    env.reset()

    assert env.action_masks().tolist() == [
        False,
        False,
    ]

    env.close()


def test_live_reset_fails_closed_and_preserves_state():
    from active_slam_rl.rl_frontier import FrontierCandidate

    env = ActiveSlamEnv(
        max_candidates=2,
    )

    candidate = FrontierCandidate(
        cell_x=2,
        cell_y=3,
        world_x=1.0,
        world_y=2.0,
        cluster_size=5,
    )

    before = env.set_frontier_state(
        candidates=[candidate],
        robot_x=0.0,
        robot_y=0.0,
    )

    env.require_external_episode_reset()

    assert (
        env.external_episode_reset_required
        is True
    )

    with pytest.raises(
        RuntimeError,
        match=(
            'fresh simulator and SLAM state'
        ),
    ):
        env.reset()

    after = env._copy_observation()

    np.testing.assert_array_equal(
        after['action_mask'],
        before['action_mask'],
    )

    np.testing.assert_array_equal(
        after['candidates'],
        before['candidates'],
    )

    assert (
        env.candidate_for_action(0)
        == candidate
    )

    env.close()
