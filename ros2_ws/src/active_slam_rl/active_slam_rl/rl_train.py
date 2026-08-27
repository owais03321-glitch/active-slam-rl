import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import (
    is_masking_supported,
)

from active_slam_rl.rl_episode import (
    DEFAULT_EPISODE_HORIZON_S,
)
from active_slam_rl.rl_experiment import (
    ExperimentRecorder,
    collect_runtime_provenance,
)
from active_slam_rl.rl_recording_env import (
    RecordedTrainingEnv,
)
from active_slam_rl.rl_training_env import (
    FreshSessionEnv,
)


POLICY_NAME = 'MultiInputPolicy'

DEFAULT_SEED = 0
DEFAULT_DEVICE = 'cpu'

DEFAULT_N_STEPS = 2
DEFAULT_BATCH_SIZE = 2

DEFAULT_EVIDENCE_ROOT = str(
    Path.home()
    / 'active-slam-rl-evidence'
)


def _positive_int(value):
    parsed = int(
        value
    )

    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            'value must be greater than zero'
        )

    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Validate or explicitly start auditable '
            'MaskablePPO training for Active SLAM.'
        )
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        '--validate-only',
        action='store_true',
        help=(
            'Construct and validate the training stack '
            'without resetting the environment, creating '
            'a run, or training.'
        ),
    )

    mode.add_argument(
        '--train',
        type=_positive_int,
        metavar='TOTAL_TIMESTEPS',
        help=(
            'Explicitly train for the requested number '
            'of timesteps.'
        ),
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
        '--n-steps',
        type=_positive_int,
        default=DEFAULT_N_STEPS,
        help=(
            'Transitions collected per environment '
            'before each PPO update.'
        ),
    )

    parser.add_argument(
        '--batch-size',
        type=_positive_int,
        default=DEFAULT_BATCH_SIZE,
        help='PPO minibatch size.',
    )

    parser.add_argument(
        '--run-id',
        help=(
            'Unique auditable run identifier. '
            'Required for --train.'
        ),
    )

    parser.add_argument(
        '--run-kind',
        choices=(
            'diagnostic',
            'formal',
        ),
        help=(
            'Evidence classification. '
            'Required for --train.'
        ),
    )

    parser.add_argument(
        '--evidence-root',
        default=DEFAULT_EVIDENCE_ROOT,
        help=(
            'Directory containing immutable run '
            'evidence. Defaults outside the Git repo.'
        ),
    )

    args = parser.parse_args(
        argv
    )

    if args.n_steps < 2:
        parser.error(
            '--n-steps must be at least 2'
        )

    if args.batch_size < 2:
        parser.error(
            '--batch-size must be at least 2'
        )

    if args.batch_size > args.n_steps:
        parser.error(
            '--batch-size must not exceed --n-steps '
            'for the single-environment trainer'
        )

    if (
        args.n_steps
        % args.batch_size
        != 0
    ):
        parser.error(
            '--n-steps must be divisible by '
            '--batch-size'
        )

    if args.train is not None:
        if not args.run_id:
            parser.error(
                '--run-id is required with --train'
            )

        if not args.run_kind:
            parser.error(
                '--run-kind is required with --train'
            )

        if (
            args.train
            % args.n_steps
            != 0
        ):
            parser.error(
                '--train must be divisible by '
                '--n-steps so requested and actual '
                'rollout transitions cannot diverge'
            )

    return args


def build_training_env():
    return FreshSessionEnv()


def build_model(
    env,
    *,
    seed=DEFAULT_SEED,
    device=DEFAULT_DEVICE,
    n_steps=DEFAULT_N_STEPS,
    batch_size=DEFAULT_BATCH_SIZE,
):
    return MaskablePPO(
        POLICY_NAME,
        env,
        seed=seed,
        device=device,
        n_steps=n_steps,
        batch_size=batch_size,
        verbose=0,
    )


def validate_training_stack(
    model,
    env,
):
    if env.session is not None:
        raise RuntimeError(
            'Physical session started during '
            'model construction.'
        )

    if getattr(
        env.action_space,
        'n',
        None,
    ) != 32:
        raise RuntimeError(
            'Training action space must be '
            'Discrete(32).'
        )

    if (
        env.observation_space[
            'candidates'
        ].shape
        != (
            32,
            4,
        )
    ):
        raise RuntimeError(
            'Candidate observation shape '
            'must be (32, 4).'
        )

    if (
        env.observation_space[
            'action_mask'
        ].shape
        != (
            32,
        )
    ):
        raise RuntimeError(
            'Action-mask observation shape '
            'must be (32,).'
        )

    vec_env = model.get_env()

    if vec_env is None:
        raise RuntimeError(
            'MaskablePPO did not retain '
            'an environment.'
        )

    if not is_masking_supported(
        vec_env
    ):
        raise RuntimeError(
            'MaskablePPO environment does not '
            'expose action masking.'
        )

    return vec_env


def training_config(
    args,
):
    return {
        'algorithm': (
            'MaskablePPO'
        ),
        'policy': (
            POLICY_NAME
        ),
        'total_timesteps_requested': (
            int(
                args.train
            )
        ),
        'seed': int(
            args.seed
        ),
        'device': str(
            args.device
        ),
        'n_steps': int(
            args.n_steps
        ),
        'batch_size': int(
            args.batch_size
        ),
        'n_envs': 1,
        'rollout_transitions_per_update': (
            int(
                args.n_steps
            )
        ),
        'episode_horizon_s': float(
            DEFAULT_EPISODE_HORIZON_S
        ),
        'fresh_physical_session_per_episode': (
            True
        ),
        'action_masking_required': (
            True
        ),
    }


def _print_stack(
    *,
    model,
    vec_env,
    env,
):
    rollout_transitions = (
        model.n_steps
        * model.n_envs
    )

    print(
        f'policy: {POLICY_NAME}'
    )

    print(
        'vec_env_type: '
        f'{type(vec_env).__name__}'
    )

    print(
        'masking_supported: True'
    )

    print(
        'session_after_model: '
        f'{env.session}'
    )

    print(
        f'n_steps: {model.n_steps}'
    )

    print(
        f'batch_size: '
        f'{model.batch_size}'
    )

    print(
        f'n_envs: {model.n_envs}'
    )

    print(
        'rollout_transitions_per_update: '
        f'{rollout_transitions}'
    )


def _close_training_stack(
    *,
    model,
    env,
):
    if model is not None:
        model_env = (
            model.get_env()
        )

        if model_env is not None:
            model_env.close()
            return

    if env is not None:
        env.close()


def _run_validation(
    args,
):
    env = None
    model = None

    try:
        env = build_training_env()

        model = build_model(
            env,
            seed=args.seed,
            device=args.device,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
        )

        vec_env = (
            validate_training_stack(
                model,
                env,
            )
        )

        _print_stack(
            model=model,
            vec_env=vec_env,
            env=env,
        )

        print(
            'RL_TRAIN_ENTRYPOINT_VALIDATION_PASS'
        )

        return 0

    finally:
        _close_training_stack(
            model=model,
            env=env,
        )


def _run_training(
    args,
):
    provenance = (
        collect_runtime_provenance(
            '.'
        )
    )

    if (
        args.run_kind
        == 'formal'
        and not provenance[
            'git_worktree_clean'
        ]
    ):
        raise RuntimeError(
            'Formal training requires a clean '
            'Git worktree at run start.'
        )

    evidence_root = (
        Path(
            args.evidence_root
        )
        .expanduser()
        .resolve()
    )

    recorder = (
        ExperimentRecorder(
            evidence_root=evidence_root,
            run_id=args.run_id,
            run_kind=args.run_kind,
            config=training_config(
                args
            ),
            provenance=provenance,
        )
    )

    raw_env = None
    env = None
    model = None
    stack_closed = False

    try:
        raw_env = (
            build_training_env()
        )

        env = RecordedTrainingEnv(
            raw_env,
            recorder=recorder,
        )

        model = build_model(
            env,
            seed=args.seed,
            device=args.device,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
        )

        vec_env = (
            validate_training_stack(
                model,
                env,
            )
        )

        _print_stack(
            model=model,
            vec_env=vec_env,
            env=env,
        )

        print(
            f'run_id: {args.run_id}'
        )

        print(
            f'run_kind: {args.run_kind}'
        )

        print(
            'evidence_run_dir: '
            f'{recorder.run_dir}'
        )

        print(
            'training_requested_timesteps: '
            f'{args.train}'
        )

        model.learn(
            total_timesteps=args.train
        )

        actual_timesteps = int(
            getattr(
                model,
                'num_timesteps',
                args.train,
            )
        )

        if actual_timesteps != args.train:
            raise RuntimeError(
                'Actual model timesteps do not '
                'match the requested auditable '
                'training horizon: '
                f'{actual_timesteps} != '
                f'{args.train}'
            )

        checkpoint_base = (
            recorder.run_dir
            / 'model'
        )

        model.save(
            str(
                checkpoint_base
            )
        )

        checkpoint_path = (
            recorder.run_dir
            / 'model.zip'
        )

        if not checkpoint_path.is_file():
            raise RuntimeError(
                'MaskablePPO did not create '
                'model.zip in the evidence run.'
            )

        model_sha256 = (
            recorder
            .record_checkpoint_hash(
                checkpoint_path,
                label='model',
            )
        )

        _close_training_stack(
            model=model,
            env=env,
        )

        stack_closed = True

        summary = (
            recorder.finish(
                {
                    'status': (
                        'complete'
                    ),
                    'requested_timesteps': (
                        int(
                            args.train
                        )
                    ),
                    'model_num_timesteps': (
                        actual_timesteps
                    ),
                    'total_recorded_reward': (
                        float(
                            env.total_recorded_reward
                        )
                    ),
                    'model_checkpoint': (
                        'model.zip'
                    ),
                    'model_sha256': (
                        model_sha256
                    ),
                }
            )
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
            'model_sha256: '
            f'{model_sha256}'
        )

        print(
            'RL_TRAIN_COMPLETE'
        )

        return 0

    except Exception as exc:
        if not stack_closed:
            try:
                _close_training_stack(
                    model=model,
                    env=(
                        env
                        if env is not None
                        else raw_env
                    ),
                )

                stack_closed = True

            except Exception:
                pass

        try:
            recorder.finish(
                {
                    'status': (
                        'failed'
                    ),
                    'requested_timesteps': (
                        int(
                            args.train
                        )
                    ),
                    'error_type': (
                        type(
                            exc
                        ).__name__
                    ),
                    'error_message': (
                        str(
                            exc
                        )
                    ),
                }
            )

        except Exception:
            pass

        raise

    finally:
        if not stack_closed:
            try:
                _close_training_stack(
                    model=model,
                    env=(
                        env
                        if env is not None
                        else raw_env
                    ),
                )

            except Exception:
                pass


def main(argv=None):
    args = parse_args(
        argv
    )

    if args.validate_only:
        return _run_validation(
            args
        )

    return _run_training(
        args
    )


if __name__ == '__main__':
    raise SystemExit(
        main()
    )
