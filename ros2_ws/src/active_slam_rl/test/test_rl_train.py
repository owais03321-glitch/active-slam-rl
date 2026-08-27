import pytest

from active_slam_rl import rl_train
from active_slam_rl.rl_training_env import FreshSessionEnv


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
    (
        'arguments',
    ),
    [
        (
            [
                '--validate-only',
                '--n-steps',
                '1',
            ],
        ),
        (
            [
                '--validate-only',
                '--batch-size',
                '1',
            ],
        ),
        (
            [
                '--validate-only',
                '--n-steps',
                '2',
                '--batch-size',
                '4',
            ],
        ),
        (
            [
                '--validate-only',
                '--n-steps',
                '3',
                '--batch-size',
                '2',
            ],
        ),
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


def test_validate_only_does_not_start_physical_session(
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
        session_factory=forbidden_session_factory
    )

    monkeypatch.setattr(
        rl_train,
        'build_training_env',
        lambda: env,
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
        session_factory=forbidden_session_factory
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


def test_train_mode_dispatches_without_real_learning(
    monkeypatch,
    capsys,
):
    class FakeEnv:
        action_space = (
            'fake-action-space'
        )

        observation_space = (
            'fake-observation-space'
        )

        session = None

        def close(self):
            raise AssertionError(
                'vector environment '
                'should own closure'
            )

    class FakeVecEnv:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeModel:
        def __init__(self):
            self.vec_env = (
                FakeVecEnv()
            )

            self.learn_calls = []

            self.n_steps = 4
            self.batch_size = 2
            self.n_envs = 1

        def get_env(self):
            return self.vec_env

        def learn(
            self,
            *,
            total_timesteps,
        ):
            self.learn_calls.append(
                total_timesteps
            )

    env = FakeEnv()
    model = FakeModel()

    build_calls = []

    monkeypatch.setattr(
        rl_train,
        'build_training_env',
        lambda: env,
    )

    def fake_build_model(
        env,
        *,
        seed,
        device,
        n_steps,
        batch_size,
    ):
        build_calls.append(
            {
                'seed': seed,
                'device': device,
                'n_steps': n_steps,
                'batch_size': batch_size,
            }
        )

        return model

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

    result = rl_train.main(
        [
            '--train',
            '2',
            '--seed',
            '11',
            '--n-steps',
            '4',
            '--batch-size',
            '2',
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
            'n_steps': 4,
            'batch_size': 2,
        },
    ]

    assert model.learn_calls == [
        2,
    ]

    assert (
        model.vec_env.closed
        is True
    )

    assert (
        'n_steps: 4'
        in captured.out
    )

    assert (
        'batch_size: 2'
        in captured.out
    )

    assert (
        'rollout_transitions_per_update: 4'
        in captured.out
    )

    assert (
        'training_requested_timesteps: 2'
        in captured.out
    )

    assert (
        'RL_TRAIN_COMPLETE'
        in captured.out
    )
