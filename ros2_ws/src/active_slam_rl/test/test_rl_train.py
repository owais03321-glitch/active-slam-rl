import pytest

from active_slam_rl import rl_train
from active_slam_rl.rl_training_env import FreshSessionEnv


def test_cli_requires_explicit_mode():
    with pytest.raises(SystemExit):
        rl_train.parse_args([])


def test_validate_only_does_not_start_physical_session(
    monkeypatch,
    capsys,
):
    factory_calls = []

    def forbidden_session_factory():
        factory_calls.append('called')
        raise RuntimeError('physical session must not start')

    env = FreshSessionEnv(
        session_factory=forbidden_session_factory
    )
    monkeypatch.setattr(
        rl_train,
        'build_training_env',
        lambda: env,
    )

    result = rl_train.main(['--validate-only'])
    captured = capsys.readouterr()

    assert result == 0
    assert factory_calls == []
    assert env.session is None
    assert (
        'RL_TRAIN_ENTRYPOINT_VALIDATION_PASS'
        in captured.out
    )


def test_train_mode_dispatches_without_real_learning(
    monkeypatch,
    capsys,
):
    class FakeEnv:
        action_space = 'fake-action-space'
        observation_space = 'fake-observation-space'
        session = None

        def close(self):
            raise AssertionError(
                'vector environment should own closure'
            )

    class FakeVecEnv:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeModel:
        def __init__(self):
            self.vec_env = FakeVecEnv()
            self.learn_calls = []

        def get_env(self):
            return self.vec_env

        def learn(self, *, total_timesteps):
            self.learn_calls.append(total_timesteps)

    env = FakeEnv()
    model = FakeModel()

    monkeypatch.setattr(
        rl_train,
        'build_training_env',
        lambda: env,
    )
    monkeypatch.setattr(
        rl_train,
        'build_model',
        lambda env, seed, device: model,
    )
    monkeypatch.setattr(
        rl_train,
        'validate_training_stack',
        lambda model, env: model.get_env(),
    )

    result = rl_train.main(['--train', '17'])
    captured = capsys.readouterr()

    assert result == 0
    assert model.learn_calls == [17]
    assert model.vec_env.closed is True
    assert 'training_requested_timesteps: 17' in captured.out
    assert 'RL_TRAIN_COMPLETE' in captured.out
