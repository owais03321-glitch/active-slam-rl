from types import SimpleNamespace

import numpy as np
import pytest

from active_slam_rl.rl_eval import (
    evaluate_frozen_policy,
    evaluation_config,
    parse_args,
)


class FakeEvalEnv:
    def __init__(self):
        self.episode = -1
        self.step_in_episode = 0
        self.reset_seeds = []
        self.actions = []

    def reset(
        self,
        *,
        seed=None,
    ):
        self.episode += 1
        self.step_in_episode = 0

        self.reset_seeds.append(
            seed
        )

        return (
            {
                'candidates': np.zeros(
                    (32, 4),
                    dtype=np.float32,
                ),
                'action_mask': (
                    self.action_masks()
                    .astype(
                        np.int8
                    )
                ),
            },
            {},
        )

    def action_masks(self):
        mask = np.zeros(
            32,
            dtype=bool,
        )

        mask[
            (
                self.episode
                + self.step_in_episode
            )
            % 3
        ] = True

        return mask

    def step(self, action):
        self.actions.append(
            int(action)
        )

        self.step_in_episode += 1

        done = (
            self.step_in_episode
            >= 2
        )

        return (
            {
                'candidates': np.zeros(
                    (32, 4),
                    dtype=np.float32,
                ),
                'action_mask': (
                    self.action_masks()
                    .astype(
                        np.int8
                    )
                ),
            },
            1.0,
            done,
            False,
            {},
        )


class FakeModel:
    def __init__(self):
        self.predict_calls = []

    def predict(
        self,
        observation,
        *,
        action_masks,
        deterministic,
    ):
        del observation

        self.predict_calls.append(
            {
                'mask': (
                    np.asarray(
                        action_masks
                    )
                    .copy()
                ),
                'deterministic': (
                    deterministic
                ),
            }
        )

        valid = np.flatnonzero(
            action_masks
        )

        return (
            np.asarray(
                valid[0]
            ),
            None,
        )


def test_parse_requires_evaluation_contract():
    args = parse_args(
        [
            '--checkpoint',
            '/tmp/model.zip',
            '--episodes',
            '3',
            '--run-id',
            'eval_probe',
            '--run-kind',
            'diagnostic',
        ]
    )

    assert args.checkpoint == (
        '/tmp/model.zip'
    )

    assert args.episodes == 3
    assert args.seed == 0
    assert args.device == 'cpu'


def test_config_explicitly_disables_learning():
    args = SimpleNamespace(
        episodes=3,
        seed=7,
        device='cpu',
    )

    config = evaluation_config(
        args,
        checkpoint_path=(
            '/tmp/model.zip'
        ),
        checkpoint_sha256=(
            'abc123'
        ),
    )

    assert (
        config[
            'mode'
        ]
        == 'frozen_evaluation'
    )

    assert (
        config[
            'learning_enabled'
        ]
        is False
    )

    assert (
        config[
            'deterministic_actions'
        ]
        is True
    )

    assert (
        config[
            'session_settle_s'
        ]
        == 3.0
    )


def test_frozen_loop_uses_masks_and_no_learning():
    model = FakeModel()
    env = FakeEvalEnv()

    evaluate_frozen_policy(
        model=model,
        env=env,
        episodes=3,
        seed=20,
    )

    assert env.reset_seeds == [
        20,
        21,
        22,
    ]

    assert len(
        model.predict_calls
    ) == 6

    assert all(
        call[
            'deterministic'
        ]
        is True
        for call
        in model.predict_calls
    )

    assert len(
        env.actions
    ) == 6


def test_all_false_mask_is_rejected():
    class EmptyMaskEnv(FakeEvalEnv):
        def action_masks(self):
            return np.zeros(
                32,
                dtype=bool,
            )

    with pytest.raises(
        RuntimeError,
        match='all-false',
    ):
        evaluate_frozen_policy(
            model=FakeModel(),
            env=EmptyMaskEnv(),
            episodes=1,
            seed=0,
        )
