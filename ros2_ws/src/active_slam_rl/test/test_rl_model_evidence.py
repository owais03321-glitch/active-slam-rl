from types import SimpleNamespace

import pytest
import torch
from sb3_contrib import MaskablePPO

from active_slam_rl import rl_train
from active_slam_rl.rl_model_evidence import (
    AuditedMaskablePPO,
    extract_train_telemetry,
    policy_fingerprint,
    resolved_maskable_ppo_config,
)
from active_slam_rl.rl_training_env import (
    FreshSessionEnv,
)


CORE_LOGGER_VALUES = {
    'train/entropy_loss': -1.25,
    'train/policy_gradient_loss': -0.05,
    'train/value_loss': 0.75,
    'train/approx_kl': 0.0125,
    'train/clip_fraction': 0.10,
    'train/loss': 0.20,
    'train/explained_variance': 0.30,
    'train/clip_range': 0.20,
}


def test_policy_fingerprint_is_deterministic_and_sensitive():
    torch.manual_seed(
        123
    )

    policy_a = torch.nn.Linear(
        4,
        3,
    )

    torch.manual_seed(
        123
    )

    policy_b = torch.nn.Linear(
        4,
        3,
    )

    fingerprint_a = (
        policy_fingerprint(
            policy_a
        )
    )

    fingerprint_b = (
        policy_fingerprint(
            policy_b
        )
    )

    assert len(
        fingerprint_a
    ) == 64

    assert (
        fingerprint_a
        == fingerprint_b
    )

    with torch.no_grad():
        policy_b.weight[
            0,
            0,
        ] += 1.0

    fingerprint_changed = (
        policy_fingerprint(
            policy_b
        )
    )

    assert (
        fingerprint_changed
        != fingerprint_a
    )


def test_resolved_contract_matches_real_project_model_without_session():
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
        model = (
            rl_train.build_model(
                env,
                seed=0,
                device='cpu',
                n_steps=2,
                batch_size=2,
            )
        )

        contract = (
            resolved_maskable_ppo_config(
                model
            )
        )

        assert factory_calls == []
        assert env.session is None

        assert contract[
            'algorithm'
        ] == 'AuditedMaskablePPO'

        assert contract[
            'policy_class'
        ] == (
            'MaskableMultiInputActorCriticPolicy'
        )

        assert contract[
            'n_steps'
        ] == 2

        assert contract[
            'batch_size'
        ] == 2

        assert contract[
            'n_envs'
        ] == 1

        assert contract[
            'n_epochs'
        ] == 10

        assert contract[
            'gamma'
        ] == pytest.approx(
            0.99
        )

        assert contract[
            'gae_lambda'
        ] == pytest.approx(
            0.95
        )

        assert contract[
            'learning_rate_at_progress_1'
        ] == pytest.approx(
            0.0003
        )

        assert contract[
            'clip_range_at_progress_1'
        ] == pytest.approx(
            0.2
        )

        assert contract[
            'optimizer_class'
        ] == 'Adam'

        assert contract[
            'policy_parameter_count'
        ] == 31073

        assert contract[
            'policy_net_arch'
        ] == {
            'pi': [
                64,
                64,
            ],
            'vf': [
                64,
                64,
            ],
        }

        assert contract[
            'initial_n_updates'
        ] == 0

        assert len(
            policy_fingerprint(
                model.policy
            )
        ) == 64

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


def fake_telemetry_model(
    values=None,
):
    logger_values = dict(
        CORE_LOGGER_VALUES
        if values is None
        else values
    )

    optimizer = SimpleNamespace(
        param_groups=[
            {
                'lr': 0.0003,
            },
        ]
    )

    policy = SimpleNamespace(
        optimizer=optimizer
    )

    return SimpleNamespace(
        logger=SimpleNamespace(
            name_to_value=(
                logger_values
            )
        ),
        policy=policy,
        num_timesteps=32,
        _n_updates=20,
        _current_progress_remaining=0.75,
    )


def test_extract_train_telemetry_requires_core_metrics():
    model = (
        fake_telemetry_model()
    )

    telemetry = (
        extract_train_telemetry(
            model
        )
    )

    assert telemetry[
        'num_timesteps'
    ] == 32

    assert telemetry[
        'n_updates'
    ] == 20

    assert telemetry[
        'learning_rate'
    ] == pytest.approx(
        0.0003
    )

    assert telemetry[
        'approx_kl'
    ] == pytest.approx(
        0.0125
    )

    broken_values = dict(
        CORE_LOGGER_VALUES
    )

    broken_values.pop(
        'train/approx_kl'
    )

    broken = (
        fake_telemetry_model(
            broken_values
        )
    )

    with pytest.raises(
        RuntimeError,
        match='train/approx_kl',
    ):
        extract_train_telemetry(
            broken
        )


def test_audited_maskableppo_emits_post_train_evidence(
    monkeypatch,
):
    emitted = []

    model = (
        AuditedMaskablePPO
        .__new__(
            AuditedMaskablePPO
        )
    )

    model._telemetry_sink = (
        emitted.append
    )

    model._optimization_index = 0
    model._n_updates = 0
    model.num_timesteps = 2
    model._current_progress_remaining = 0.5

    model._logger = SimpleNamespace(
        name_to_value=dict(
            CORE_LOGGER_VALUES
        )
    )

    parameter = torch.nn.Parameter(
        torch.tensor(
            [
                1.0,
                2.0,
            ]
        )
    )

    class FakePolicy:
        def __init__(self):
            self.optimizer = (
                SimpleNamespace(
                    param_groups=[
                        {
                            'lr': 0.0003,
                        },
                    ]
                )
            )

        def state_dict(self):
            return {
                'weight': (
                    parameter
                ),
            }

    model.policy = FakePolicy()

    def fake_base_train(
        self,
    ):
        self._n_updates = 10

        with torch.no_grad():
            parameter[
                0
            ] += 0.25

    monkeypatch.setattr(
        MaskablePPO,
        'train',
        fake_base_train,
    )

    model.train()

    assert (
        model.optimization_index
        == 1
    )

    assert len(
        emitted
    ) == 1

    row = emitted[
        0
    ]

    assert row[
        'optimization_index'
    ] == 0

    assert row[
        'num_timesteps'
    ] == 2

    assert row[
        'n_updates'
    ] == 10

    assert row[
        'approx_kl'
    ] == pytest.approx(
        0.0125
    )

    assert len(
        row[
            'policy_fingerprint'
        ]
    ) == 64
