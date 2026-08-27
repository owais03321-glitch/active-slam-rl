from types import SimpleNamespace

import numpy as np
import pytest

from active_slam_rl.rl_eval import (
    evaluate_frozen_policy,
    evaluation_config,
    parse_args,
    simulation_command,
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


def test_parse_visual_demo_flags():
    args = parse_args(
        [
            '--checkpoint',
            '/tmp/model.zip',
            '--episodes',
            '1',
            '--run-id',
            'visual_demo',
            '--run-kind',
            'diagnostic',
            '--visual',
            '--verbose-steps',
        ]
    )

    assert args.visual is True
    assert args.verbose_steps is True


def test_config_records_visual_demo_contract():
    args = SimpleNamespace(
        episodes=1,
        seed=9,
        device='cpu',
        visual=True,
        verbose_steps=True,
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
            'visual_simulation'
        ]
        is True
    )

    assert (
        config[
            'verbose_steps'
        ]
        is True
    )

    assert (
        'use_rviz:=True'
        in config[
            'simulation_command'
        ]
    )

    assert (
        'headless:=False'
        in config[
            'simulation_command'
        ]
    )

    assert (
        config[
            'learning_enabled'
        ]
        is False
    )


def test_verbose_frozen_loop_prints_transition(capsys):
    class VerboseEvalEnv(
        FakeEvalEnv
    ):
        def step(
            self,
            action,
        ):
            (
                observation,
                reward,
                terminated,
                truncated,
                _,
            ) = super().step(
                action
            )

            return (
                observation,
                reward,
                terminated,
                truncated,
                {
                    'area_gain_m2': 2.5,
                    'path_delta_m': 1.25,
                    'goal_x': 0.5,
                    'goal_y': -0.75,
                    'navigation_status': 4,
                    'navigation_succeeded': True,
                },
            )

    evaluate_frozen_policy(
        model=FakeModel(),
        env=VerboseEvalEnv(),
        episodes=1,
        seed=0,
        verbose_steps=True,
    )

    output = (
        capsys.readouterr()
        .out
    )

    assert 'RL_DEMO_STEP' in output
    assert 'reward=1.000000' in output
    assert 'area_gain_m2=2.500000' in output
    assert 'nav_success=True' in output



def test_parse_initial_pose_arguments():
    args = parse_args(
        [
            '--checkpoint',
            '/tmp/model.zip',
            '--episodes',
            '1',
            '--run-id',
            'pose_probe',
            '--run-kind',
            'diagnostic',
            '--x-pose',
            '-1.75',
            '--y-pose',
            '-0.25',
            '--yaw',
            '1.5708',
        ]
    )

    assert args.x_pose == -1.75
    assert args.y_pose == -0.25
    assert args.yaw == 1.5708


def test_simulation_command_appends_pose():
    command = simulation_command(
        visual=False,
        x_pose=-1.75,
        y_pose=-0.25,
        yaw=1.5708,
    )

    assert 'headless:=True' in command
    assert 'use_rviz:=False' in command
    assert 'x_pose:=-1.75' in command
    assert 'y_pose:=-0.25' in command
    assert 'yaw:=1.5708' in command


def test_config_records_initial_pose():
    args = SimpleNamespace(
        episodes=1,
        seed=4,
        device='cpu',
        visual=False,
        verbose_steps=False,
        x_pose=-2.0,
        y_pose=-0.25,
        yaw=-1.5708,
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

    assert config[
        'initial_pose'
    ] == {
        'x_pose': -2.0,
        'y_pose': -0.25,
        'yaw': -1.5708,
    }

    assert (
        'x_pose:=-2.0'
        in config[
            'simulation_command'
        ]
    )

    assert (
        config[
            'learning_enabled'
        ]
        is False
    )



def test_parse_rviz_config_file_argument():
    args = parse_args(
        [
            '--checkpoint',
            '/tmp/model.zip',
            '--episodes',
            '1',
            '--run-id',
            'presentation_probe',
            '--run-kind',
            'diagnostic',
            '--visual',
            '--rviz-config-file',
            '/tmp/demo.rviz',
        ]
    )

    assert (
        args.rviz_config_file
        == '/tmp/demo.rviz'
    )


def test_visual_simulation_command_appends_rviz_config():
    command = simulation_command(
        visual=True,
        rviz_config_file=(
            '/tmp/demo.rviz'
        ),
    )

    assert (
        'rviz_config_file:=/tmp/demo.rviz'
        in command
    )


def test_headless_simulation_command_ignores_rviz_config():
    command = simulation_command(
        visual=False,
        rviz_config_file=(
            '/tmp/demo.rviz'
        ),
    )

    assert not any(
        item.startswith(
            'rviz_config_file:='
        )
        for item in command
    )
