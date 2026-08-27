import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from active_slam_rl import rl_train
from active_slam_rl.rl_training_env import (
    FreshSessionEnv,
)


def clean_provenance():
    return {
        'captured_at_utc': (
            '2026-08-27T00:00:00+00:00'
        ),
        'git_commit': 'test-commit',
        'git_branch': 'rl-active-slam',
        'git_status_at_start': '',
        'git_worktree_clean': True,
        'python_executable': (
            '/project/.venv/bin/python3'
        ),
        'python_version': '3.12.3',
        'platform': 'test-platform',
        'package_versions': {
            'sb3-contrib': '2.9.0',
            'stable-baselines3': '2.9.0',
            'gymnasium': '1.3.0',
            'torch': '2.9.1+cpu',
        },
    }


def test_cli_requires_explicit_mode():
    with pytest.raises(
        SystemExit
    ):
        rl_train.parse_args(
            []
        )


def test_cli_uses_smoke_safe_rollout_defaults():
    args = rl_train.parse_args(
        [
            '--validate-only',
        ]
    )

    assert args.n_steps == 2
    assert args.batch_size == 2


@pytest.mark.parametrize(
    'arguments',
    [
        [
            '--validate-only',
            '--n-steps',
            '1',
        ],
        [
            '--validate-only',
            '--batch-size',
            '1',
        ],
        [
            '--validate-only',
            '--n-steps',
            '2',
            '--batch-size',
            '4',
        ],
        [
            '--validate-only',
            '--n-steps',
            '3',
            '--batch-size',
            '2',
        ],
    ],
)
def test_cli_rejects_unsafe_rollout_configuration(
    arguments,
):
    with pytest.raises(
        SystemExit
    ):
        rl_train.parse_args(
            arguments
        )


def test_cli_rejects_non_integral_training_rollout():
    with pytest.raises(
        SystemExit
    ):
        rl_train.parse_args(
            [
                '--train',
                '3',
                '--run-id',
                'diag_001',
                '--run-kind',
                'diagnostic',
                '--n-steps',
                '2',
                '--batch-size',
                '2',
            ]
        )


@pytest.mark.parametrize(
    'arguments',
    [
        [
            '--train',
            '2',
            '--run-kind',
            'diagnostic',
        ],
        [
            '--train',
            '2',
            '--run-id',
            'diag_001',
        ],
    ],
)
def test_cli_requires_training_evidence_identity(
    arguments,
):
    with pytest.raises(
        SystemExit
    ):
        rl_train.parse_args(
            arguments
        )


def test_validate_only_does_not_start_session_or_create_run(
    monkeypatch,
    capsys,
):
    factory_calls = []

    def forbidden_session_factory():
        factory_calls.append(
            'called'
        )

        raise RuntimeError(
            'physical session must not start'
        )

    env = FreshSessionEnv(
        session_factory=(
            forbidden_session_factory
        )
    )

    monkeypatch.setattr(
        rl_train,
        'build_training_env',
        lambda: env,
    )

    def forbidden_recorder(
        *args,
        **kwargs,
    ):
        raise AssertionError(
            'validation must not create evidence'
        )

    monkeypatch.setattr(
        rl_train,
        'ExperimentRecorder',
        forbidden_recorder,
    )

    result = rl_train.main(
        [
            '--validate-only',
        ]
    )

    captured = (
        capsys.readouterr()
    )

    assert result == 0
    assert factory_calls == []
    assert env.session is None

    assert (
        'n_steps: 2'
        in captured.out
    )

    assert (
        'batch_size: 2'
        in captured.out
    )

    assert (
        'rollout_transitions_per_update: 2'
        in captured.out
    )

    assert (
        'RL_TRAIN_ENTRYPOINT_VALIDATION_PASS'
        in captured.out
    )


def test_build_model_uses_requested_rollout_configuration():
    factory_calls = []

    def forbidden_session_factory():
        factory_calls.append(
            'called'
        )

        raise RuntimeError(
            'physical session must not start'
        )

    env = FreshSessionEnv(
        session_factory=(
            forbidden_session_factory
        )
    )

    model = None

    try:
        model = rl_train.build_model(
            env,
            seed=7,
            device='cpu',
            n_steps=4,
            batch_size=2,
        )

        assert model.n_steps == 4
        assert model.batch_size == 2
        assert model.n_envs == 1

        assert factory_calls == []
        assert env.session is None

    finally:
        if model is not None:
            model_env = (
                model.get_env()
            )

            if model_env is not None:
                model_env.close()

            else:
                env.close()

        else:
            env.close()


def test_formal_training_rejects_dirty_worktree_before_env(
    monkeypatch,
    tmp_path,
):
    provenance = (
        clean_provenance()
    )

    provenance[
        'git_worktree_clean'
    ] = False

    provenance[
        'git_status_at_start'
    ] = ' M dirty.py'

    monkeypatch.setattr(
        rl_train,
        'collect_runtime_provenance',
        lambda repo_root: provenance,
    )

    env_calls = []

    monkeypatch.setattr(
        rl_train,
        'build_training_env',
        lambda: env_calls.append(
            'called'
        ),
    )

    with pytest.raises(
        RuntimeError,
        match='clean Git worktree',
    ):
        rl_train.main(
            [
                '--train',
                '2',
                '--run-id',
                'formal_dirty',
                '--run-kind',
                'formal',
                '--evidence-root',
                str(
                    tmp_path
                ),
            ]
        )

    assert env_calls == []

    assert not (
        tmp_path
        / 'runs'
        / 'formal_dirty'
    ).exists()


class FakeEpisodeTimeLimit:
    def __init__(self):
        self.elapsed_s = 0.0


class FakePathTracker:
    def __init__(self):
        self.path_length_m = 0.0


class FakeNode:
    def __init__(self):
        self.map_revision = 1

        self.latest_explored_area_m2 = (
            1.0
        )

        self.latest_robot_xy = (
            0.0,
            0.0,
        )

        self.path_tracker = (
            FakePathTracker()
        )

        self.episode_time_limit = (
            FakeEpisodeTimeLimit()
        )


class FakeSession:
    def __init__(self):
        self.node = FakeNode()


class FakePhysicalEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = (
            spaces.Discrete(
                32
            )
        )

        self.observation_space = (
            spaces.Dict(
                {
                    'candidates': (
                        spaces.Box(
                            low=-1e6,
                            high=1e6,
                            shape=(
                                32,
                                4,
                            ),
                            dtype=np.float32,
                        )
                    ),
                    'action_mask': (
                        spaces.MultiBinary(
                            32
                        )
                    ),
                }
            )
        )

        self.session = None
        self.closed = False
        self.step_count = 0

        self._mask = np.zeros(
            32,
            dtype=np.int8,
        )

    @property
    def live_env(self):
        return self

    def _observation(self):
        return {
            'candidates': (
                np.zeros(
                    (
                        32,
                        4,
                    ),
                    dtype=np.float32,
                )
            ),
            'action_mask': (
                self._mask.copy()
            ),
        }

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(
            seed=seed
        )

        self.session = (
            FakeSession()
        )

        self.step_count = 0

        self._mask = np.zeros(
            32,
            dtype=np.int8,
        )

        self._mask[
            0
        ] = 1

        self._mask[
            1
        ] = 1

        return (
            self._observation(),
            {},
        )

    def action_masks(self):
        return self._mask.astype(
            bool,
            copy=True,
        )

    def step(
        self,
        action,
    ):
        action = int(
            action
        )

        assert self._mask[
            action
        ] == 1

        self.step_count += 1

        node = (
            self.session.node
        )

        node.map_revision += 2

        node.latest_explored_area_m2 += (
            1.5
        )

        node.path_tracker.path_length_m += (
            0.75
        )

        node.latest_robot_xy = (
            0.25
            * self.step_count,
            -0.1
            * self.step_count,
        )

        node.episode_time_limit.elapsed_s += (
            3.0
        )

        self._mask = np.zeros(
            32,
            dtype=np.int8,
        )

        self._mask[
            0
        ] = 1

        self._mask[
            2
        ] = 1

        info = {
            'action': action,
            'goal_x': (
                0.5
                * self.step_count
            ),
            'goal_y': (
                -0.25
                * self.step_count
            ),
            'area_gain_m2': 1.5,
            'path_delta_m': 0.75,
            'navigation_accepted': True,
            'navigation_status': 4,
            'navigation_succeeded': True,
        }

        return (
            self._observation(),
            1.425,
            False,
            False,
            info,
        )

    def close(self):
        self.closed = True


class FakeVecEnv:
    def __init__(
        self,
        env,
    ):
        self.env = env
        self.closed = False

    def close(self):
        if self.closed:
            return

        self.env.close()
        self.closed = True


class FakeModel:
    def __init__(
        self,
        env,
        *,
        n_steps,
        batch_size,
        telemetry_sink=None,
    ):
        self.env = env
        self.telemetry_sink = telemetry_sink
        self.policy = object()
        self._n_updates = 0

        self.vec_env = (
            FakeVecEnv(
                env
            )
        )

        self.n_steps = (
            n_steps
        )

        self.batch_size = (
            batch_size
        )

        self.n_envs = 1
        self.num_timesteps = 0

    def get_env(self):
        return self.vec_env

    def learn(
        self,
        *,
        total_timesteps,
    ):
        self.env.reset()

        for _ in range(
            total_timesteps
        ):
            valid = np.flatnonzero(
                self.env.action_masks()
            )

            assert len(
                valid
            ) > 0

            self.env.step(
                int(
                    valid[
                        0
                    ]
                )
            )

            self.num_timesteps += 1

        self._n_updates = 10

        if self.telemetry_sink is not None:
            self.telemetry_sink(
                {
                    'num_timesteps': (
                        self.num_timesteps
                    ),
                    'n_updates': (
                        self._n_updates
                    ),
                    'progress_remaining': 0.0,
                    'learning_rate': 0.0003,
                    'entropy_loss': -1.0,
                    'policy_gradient_loss': -0.1,
                    'value_loss': 0.2,
                    'approx_kl': 0.01,
                    'clip_fraction': 0.05,
                    'loss': 0.15,
                    'explained_variance': 0.25,
                    'clip_range': 0.2,
                    'clip_range_vf': None,
                    'policy_fingerprint': (
                        'f' * 64
                    ),
                }
            )

        return self

    def save(
        self,
        path,
    ):
        checkpoint = Path(
            str(
                path
            )
            + '.zip'
        )

        checkpoint.write_bytes(
            b'fake-maskable-ppo-model'
        )


def read_csv(
    path,
):
    with Path(
        path
    ).open(
        newline='',
    ) as file_handle:
        return list(
            csv.DictReader(
                file_handle
            )
        )


def test_diagnostic_training_writes_evidence_and_model_hash(
    monkeypatch,
    tmp_path,
    capsys,
):
    physical_env = (
        FakePhysicalEnv()
    )

    monkeypatch.setattr(
        rl_train,
        'collect_runtime_provenance',
        lambda repo_root: (
            clean_provenance()
        ),
    )

    monkeypatch.setattr(
        rl_train,
        'build_training_env',
        lambda: physical_env,
    )

    build_calls = []

    def fake_build_model(
        env,
        *,
        seed,
        device,
        n_steps,
        batch_size,
        telemetry_sink=None,
    ):
        build_calls.append(
            {
                'seed': seed,
                'device': device,
                'n_steps': n_steps,
                'batch_size': batch_size,
            }
        )

        return FakeModel(
            env,
            n_steps=n_steps,
            batch_size=batch_size,
            telemetry_sink=telemetry_sink,
        )

    monkeypatch.setattr(
        rl_train,
        'build_model',
        fake_build_model,
    )

    monkeypatch.setattr(
        rl_train,
        'validate_training_stack',
        lambda model, env: (
            model.get_env()
        ),
    )

    monkeypatch.setattr(
        rl_train,
        'resolved_maskable_ppo_config',
        lambda model: {
            'algorithm': (
                'AuditedMaskablePPO'
            ),
            'n_steps': 2,
            'batch_size': 2,
            'n_epochs': 10,
        },
    )

    fingerprints = iter(
        [
            'a' * 64,
            'b' * 64,
        ]
    )

    monkeypatch.setattr(
        rl_train,
        'policy_fingerprint',
        lambda policy: next(
            fingerprints
        ),
    )

    result = rl_train.main(
        [
            '--train',
            '2',
            '--run-id',
            'diag_001',
            '--run-kind',
            'diagnostic',
            '--seed',
            '11',
            '--n-steps',
            '2',
            '--batch-size',
            '2',
            '--evidence-root',
            str(
                tmp_path
            ),
        ]
    )

    captured = (
        capsys.readouterr()
    )

    assert result == 0

    assert build_calls == [
        {
            'seed': 11,
            'device': 'cpu',
            'n_steps': 2,
            'batch_size': 2,
        }
    ]

    run_dir = (
        tmp_path
        / 'runs'
        / 'diag_001'
    )

    assert run_dir.is_dir()

    metadata = json.loads(
        (
            run_dir
            / 'metadata.json'
        ).read_text()
    )

    config = json.loads(
        (
            run_dir
            / 'config.json'
        ).read_text()
    )

    summary = json.loads(
        (
            run_dir
            / 'summary.json'
        ).read_text()
    )

    assert metadata[
        'run_kind'
    ] == 'diagnostic'

    assert metadata[
        'git_worktree_clean'
    ] is True

    assert config[
        'algorithm'
    ] == 'MaskablePPO'

    assert config[
        'total_timesteps_requested'
    ] == 2

    assert config[
        'n_steps'
    ] == 2

    assert config[
        'batch_size'
    ] == 2

    steps = read_csv(
        run_dir
        / 'steps.csv'
    )

    assert len(
        steps
    ) == 2

    assert [
        row[
            'step_index'
        ]
        for row in steps
    ] == [
        '0',
        '1',
    ]

    assert all(
        row[
            'navigation_succeeded'
        ] == 'True'
        for row in steps
    )

    assert all(
        row[
            'action_mask_bits'
        ]
        for row in steps
    )

    episodes = read_csv(
        run_dir
        / 'episodes.csv'
    )

    assert len(
        episodes
    ) == 1

    assert episodes[
        0
    ][
        'outcome'
    ] == 'training_stop'

    assert episodes[
        0
    ][
        'steps'
    ] == '2'

    assert (
        run_dir
        / 'initial_model.zip'
    ).is_file()

    assert (
        run_dir
        / 'initial_model.sha256'
    ).is_file()

    assert (
        run_dir
        / 'resolved_model.json'
    ).is_file()

    assert (
        run_dir
        / 'model.zip'
    ).is_file()

    assert (
        run_dir
        / 'model.sha256'
    ).is_file()

    hash_text = (
        run_dir
        / 'model.sha256'
    ).read_text()

    assert (
        'model.zip'
        in hash_text
    )

    assert len(
        hash_text.split()[0]
    ) == 64

    assert summary[
        'status'
    ] == 'complete'

    assert summary[
        'requested_timesteps'
    ] == 2

    assert summary[
        'model_num_timesteps'
    ] == 2

    assert summary[
        'recorded_steps'
    ] == 2

    assert summary[
        'recorded_episodes'
    ] == 1

    assert summary[
        'recorded_updates'
    ] == 1

    assert summary[
        'model_n_updates'
    ] == 10

    assert summary[
        'initial_model_checkpoint'
    ] == 'initial_model.zip'

    assert len(
        summary[
            'initial_model_sha256'
        ]
    ) == 64

    assert summary[
        'model_checkpoint'
    ] == 'model.zip'

    assert len(
        summary[
            'model_sha256'
        ]
    ) == 64

    assert summary[
        'initial_policy_fingerprint'
    ] == (
        'a' * 64
    )

    assert summary[
        'final_policy_fingerprint'
    ] == (
        'b' * 64
    )

    assert summary[
        'policy_parameters_changed'
    ] is True

    resolved_model = json.loads(
        (
            run_dir
            / 'resolved_model.json'
        ).read_text()
    )

    assert resolved_model[
        'algorithm'
    ] == 'AuditedMaskablePPO'

    assert resolved_model[
        'initial_policy_fingerprint'
    ] == (
        'a' * 64
    )

    assert len(
        resolved_model[
            'initial_model_sha256'
        ]
    ) == 64

    updates = read_csv(
        run_dir
        / 'updates.csv'
    )

    assert len(
        updates
    ) == 1

    assert updates[
        0
    ][
        'optimization_index'
    ] == '0'

    assert updates[
        0
    ][
        'n_updates'
    ] == '10'

    assert (
        physical_env.closed
        is True
    )

    assert (
        'training_requested_timesteps: 2'
        in captured.out
    )

    assert (
        'recorded_steps: 2'
        in captured.out
    )

    assert (
        'recorded_episodes: 1'
        in captured.out
    )

    assert (
        'recorded_updates: 1'
        in captured.out
    )

    assert (
        'model_n_updates: 10'
        in captured.out
    )

    assert (
        'policy_parameters_changed: True'
        in captured.out
    )

    assert (
        'model_sha256:'
        in captured.out
    )

    assert (
        'RL_TRAIN_COMPLETE'
        in captured.out
    )



def test_build_model_uses_audited_maskableppo_without_session():
    factory_calls = []

    def forbidden_session_factory():
        factory_calls.append(
            'called'
        )

        raise RuntimeError(
            'physical session must not start'
        )

    env = FreshSessionEnv(
        session_factory=(
            forbidden_session_factory
        )
    )

    model = None

    try:
        model = rl_train.build_model(
            env,
            seed=0,
            device='cpu',
            n_steps=2,
            batch_size=2,
        )

        assert type(
            model
        ).__name__ == (
            'AuditedMaskablePPO'
        )

        assert (
            model.optimization_index
            == 0
        )

        assert factory_calls == []
        assert env.session is None

    finally:
        if model is not None:
            model_env = (
                model.get_env()
            )

            if model_env is not None:
                model_env.close()

            else:
                env.close()

        else:
            env.close()
