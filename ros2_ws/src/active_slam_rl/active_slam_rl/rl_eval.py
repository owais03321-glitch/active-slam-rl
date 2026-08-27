import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from active_slam_rl.rl_experiment import (
    ExperimentRecorder,
    collect_runtime_provenance,
    sha256_file,
)
from active_slam_rl.rl_model_evidence import (
    policy_fingerprint,
    resolved_maskable_ppo_config,
)
from active_slam_rl.rl_recording_env import (
    RecordedTrainingEnv,
)
from active_slam_rl.rl_training_env import (
    DEFAULT_SESSION_SETTLE_S,
    FreshSessionEnv,
)


DEFAULT_SEED = 0
DEFAULT_DEVICE = 'cpu'

DEFAULT_EVIDENCE_ROOT = str(
    Path.home()
    / 'active-slam-rl-evidence'
)


def _positive_int(value):
    parsed = int(value)

    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            'value must be greater than zero'
        )

    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Run an auditable frozen MaskablePPO '
            'Active SLAM evaluation.'
        )
    )

    parser.add_argument(
        '--checkpoint',
        required=True,
    )

    parser.add_argument(
        '--episodes',
        type=_positive_int,
        required=True,
    )

    parser.add_argument(
        '--run-id',
        required=True,
    )

    parser.add_argument(
        '--run-kind',
        choices=(
            'diagnostic',
            'formal',
        ),
        required=True,
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        '--device',
        default=DEFAULT_DEVICE,
    )

    parser.add_argument(
        '--evidence-root',
        default=DEFAULT_EVIDENCE_ROOT,
    )

    return parser.parse_args(argv)


def evaluation_config(
    args,
    *,
    checkpoint_path,
    checkpoint_sha256,
):
    return {
        'mode': 'frozen_evaluation',
        'algorithm': 'MaskablePPO',
        'checkpoint': str(
            checkpoint_path
        ),
        'checkpoint_sha256': (
            checkpoint_sha256
        ),
        'episodes_requested': int(
            args.episodes
        ),
        'seed': int(
            args.seed
        ),
        'device': str(
            args.device
        ),
        'deterministic_actions': True,
        'learning_enabled': False,
        'fresh_physical_session_per_episode': (
            True
        ),
        'session_settle_s': float(
            DEFAULT_SESSION_SETTLE_S
        ),
        'action_masking_required': True,
    }


def evaluate_frozen_policy(
    *,
    model,
    env,
    episodes,
    seed,
):
    for episode_index in range(
        episodes
    ):
        observation, _ = env.reset(
            seed=(
                seed
                + episode_index
            )
        )

        while True:
            mask = np.asarray(
                env.action_masks(),
                dtype=bool,
            ).reshape(-1)

            if not np.any(mask):
                raise RuntimeError(
                    'Frozen evaluation received '
                    'an all-false decision mask.'
                )

            action, _ = model.predict(
                observation,
                action_masks=mask,
                deterministic=True,
            )

            (
                observation,
                _,
                terminated,
                truncated,
                _,
            ) = env.step(
                int(
                    np.asarray(
                        action
                    ).item()
                )
            )

            if (
                terminated
                or truncated
            ):
                break


def main(argv=None):
    args = parse_args(argv)

    provenance = (
        collect_runtime_provenance(
            '.'
        )
    )

    if (
        args.run_kind == 'formal'
        and not provenance[
            'git_worktree_clean'
        ]
    ):
        raise RuntimeError(
            'Formal evaluation requires '
            'a clean Git worktree.'
        )

    checkpoint_path = (
        Path(
            args.checkpoint
        )
        .expanduser()
        .resolve()
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint_sha256 = (
        sha256_file(
            checkpoint_path
        )
    )

    evidence_root = (
        Path(
            args.evidence_root
        )
        .expanduser()
        .resolve()
    )

    recorder = ExperimentRecorder(
        evidence_root=evidence_root,
        run_id=args.run_id,
        run_kind=args.run_kind,
        config=evaluation_config(
            args,
            checkpoint_path=(
                checkpoint_path
            ),
            checkpoint_sha256=(
                checkpoint_sha256
            ),
        ),
        provenance=provenance,
    )

    raw_env = None
    env = None

    try:
        raw_env = FreshSessionEnv()

        env = RecordedTrainingEnv(
            raw_env,
            recorder=recorder,
        )

        model = MaskablePPO.load(
            str(
                checkpoint_path
            ),
            device=args.device,
        )

        initial_fingerprint = (
            policy_fingerprint(
                model.policy
            )
        )

        resolved = (
            resolved_maskable_ppo_config(
                model
            )
        )

        resolved[
            'evaluation_checkpoint'
        ] = checkpoint_path.name

        resolved[
            'evaluation_checkpoint_sha256'
        ] = checkpoint_sha256

        resolved[
            'evaluation_policy_fingerprint'
        ] = initial_fingerprint

        resolved[
            'learning_enabled'
        ] = False

        resolved[
            'deterministic_actions'
        ] = True

        recorder.record_model_contract(
            resolved
        )

        print(
            'evaluation_checkpoint: '
            f'{checkpoint_path}'
        )

        print(
            'checkpoint_sha256: '
            f'{checkpoint_sha256}'
        )

        print(
            'policy_fingerprint: '
            f'{initial_fingerprint}'
        )

        print(
            'episodes_requested: '
            f'{args.episodes}'
        )

        print(
            'deterministic_actions: True'
        )

        print(
            'learning_enabled: False'
        )

        evaluate_frozen_policy(
            model=model,
            env=env,
            episodes=args.episodes,
            seed=args.seed,
        )

        final_fingerprint = (
            policy_fingerprint(
                model.policy
            )
        )

        if (
            final_fingerprint
            != initial_fingerprint
        ):
            raise RuntimeError(
                'Frozen evaluation changed '
                'policy parameters.'
            )

        env.close()

        summary = recorder.finish(
            {
                'status': 'complete',
                'mode': (
                    'frozen_evaluation'
                ),
                'episodes_requested': (
                    int(
                        args.episodes
                    )
                ),
                'total_recorded_reward': (
                    float(
                        env.total_recorded_reward
                    )
                ),
                'checkpoint': (
                    str(
                        checkpoint_path
                    )
                ),
                'checkpoint_sha256': (
                    checkpoint_sha256
                ),
                'initial_policy_fingerprint': (
                    initial_fingerprint
                ),
                'final_policy_fingerprint': (
                    final_fingerprint
                ),
                'policy_parameters_unchanged': (
                    True
                ),
                'learning_enabled': False,
                'deterministic_actions': True,
            }
        )

        print(
            'recorded_steps: '
            f'{summary["recorded_steps"]}'
        )

        print(
            'recorded_episodes: '
            f'{summary["recorded_episodes"]}'
        )

        print(
            'recorded_updates: '
            f'{summary["recorded_updates"]}'
        )

        print(
            'policy_parameters_unchanged: True'
        )

        print(
            'RL_EVAL_COMPLETE'
        )

        return 0

    except Exception as exc:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

        try:
            recorder.finish(
                {
                    'status': 'failed',
                    'mode': (
                        'frozen_evaluation'
                    ),
                    'error_type': (
                        type(exc).__name__
                    ),
                    'error_message': (
                        str(exc)
                    ),
                    'episodes_requested': (
                        int(
                            args.episodes
                        )
                    ),
                    'checkpoint': (
                        str(
                            checkpoint_path
                        )
                    ),
                    'checkpoint_sha256': (
                        checkpoint_sha256
                    ),
                }
            )
        except Exception:
            pass

        raise


if __name__ == '__main__':
    main()
