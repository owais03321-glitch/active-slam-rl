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
from active_slam_rl.rl_session import FreshRlSession
from active_slam_rl.rl_simulation import (
    DEFAULT_SIMULATION_COMMAND,
    SimulationLifecycle,
)
from active_slam_rl.rl_training_env import (
    DEFAULT_SESSION_SETTLE_S,
    FreshSessionEnv,
)


DEFAULT_SEED = 0
DEFAULT_DEVICE = 'cpu'

VISUAL_SIMULATION_COMMAND = (
    'ros2',
    'launch',
    'nav2_bringup',
    'tb3_simulation_launch.py',
    'slam:=True',
    'use_rviz:=True',
    'headless:=False',
)

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

    parser.add_argument(
        '--visual',
        action='store_true',
        help=(
            'Launch Gazebo and RViz GUIs for '
            'frozen-policy demonstration.'
        ),
    )

    parser.add_argument(
        '--verbose-steps',
        action='store_true',
        help=(
            'Print each frozen policy decision '
            'and physical transition result.'
        ),
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
        'visual_simulation': bool(
            getattr(
                args,
                'visual',
                False,
            )
        ),
        'verbose_steps': bool(
            getattr(
                args,
                'verbose_steps',
                False,
            )
        ),
        'simulation_command': list(
            VISUAL_SIMULATION_COMMAND
            if bool(
                getattr(
                    args,
                    'visual',
                    False,
                )
            )
            else DEFAULT_SIMULATION_COMMAND
        ),
    }


def make_visual_session():
    return FreshRlSession(
        simulation=SimulationLifecycle(
            command=VISUAL_SIMULATION_COMMAND
        )
    )


def build_evaluation_env(args):
    if bool(
        getattr(
            args,
            'visual',
            False,
        )
    ):
        return FreshSessionEnv(
            session_factory=make_visual_session
        )

    return FreshSessionEnv()


def evaluate_frozen_policy(
    *,
    model,
    env,
    episodes,
    seed,
    verbose_steps=False,
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

        step_index = 0

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

            action_index = int(
                np.asarray(
                    action
                ).item()
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(
                action_index
            )

            if verbose_steps:
                print(
                    'RL_DEMO_STEP '
                    f'episode={episode_index} '
                    f'step={step_index} '
                    f'action={action_index} '
                    f'valid_actions='
                    f'{int(np.count_nonzero(mask))} '
                    f'reward={float(reward):.6f} '
                    f'area_gain_m2='
                    f'{float(info["area_gain_m2"]):.6f} '
                    f'path_delta_m='
                    f'{float(info["path_delta_m"]):.6f} '
                    f'goal=('
                    f'{float(info["goal_x"]):.3f},'
                    f'{float(info["goal_y"]):.3f}) '
                    f'nav_status='
                    f'{info["navigation_status"]} '
                    f'nav_success='
                    f'{bool(info["navigation_succeeded"])} '
                    f'terminated={bool(terminated)} '
                    f'truncated={bool(truncated)}'
                )

            step_index += 1

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
        raw_env = build_evaluation_env(
            args
        )

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

        print(
            'visual_simulation: '
            f'{bool(args.visual)}'
        )

        print(
            'verbose_steps: '
            f'{bool(args.verbose_steps)}'
        )

        evaluate_frozen_policy(
            model=model,
            env=env,
            episodes=args.episodes,
            seed=args.seed,
            verbose_steps=(
                args.verbose_steps
            ),
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
                'visual_simulation': bool(
                    args.visual
                ),
                'verbose_steps': bool(
                    args.verbose_steps
                ),
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
