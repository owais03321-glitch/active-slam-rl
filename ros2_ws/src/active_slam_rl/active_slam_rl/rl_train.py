import argparse

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import is_masking_supported

from active_slam_rl.rl_training_env import FreshSessionEnv


POLICY_NAME = 'MultiInputPolicy'
DEFAULT_SEED = 0
DEFAULT_DEVICE = 'cpu'


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
            'Validate or explicitly start MaskablePPO training '
            'for Active SLAM.'
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        '--validate-only',
        action='store_true',
        help=(
            'Construct and validate the training stack without '
            'resetting the environment or training.'
        ),
    )
    mode.add_argument(
        '--train',
        type=_positive_int,
        metavar='TOTAL_TIMESTEPS',
        help='Explicitly train for the requested number of timesteps.',
    )
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--device', default=DEFAULT_DEVICE)
    return parser.parse_args(argv)


def build_training_env():
    return FreshSessionEnv()


def build_model(env, *, seed=DEFAULT_SEED, device=DEFAULT_DEVICE):
    return MaskablePPO(
        POLICY_NAME,
        env,
        seed=seed,
        device=device,
        verbose=0,
    )


def validate_training_stack(model, env):
    if env.session is not None:
        raise RuntimeError(
            'Physical session started during model construction.'
        )
    if getattr(env.action_space, 'n', None) != 32:
        raise RuntimeError(
            'Training action space must be Discrete(32).'
        )
    if env.observation_space['candidates'].shape != (32, 4):
        raise RuntimeError(
            'Candidate observation shape must be (32, 4).'
        )
    if env.observation_space['action_mask'].shape != (32,):
        raise RuntimeError(
            'Action-mask observation shape must be (32,).'
        )

    vec_env = model.get_env()
    if vec_env is None:
        raise RuntimeError(
            'MaskablePPO did not retain an environment.'
        )
    if not is_masking_supported(vec_env):
        raise RuntimeError(
            'MaskablePPO environment does not expose action masking.'
        )
    return vec_env


def main(argv=None):
    args = parse_args(argv)
    env = None
    model = None

    try:
        env = build_training_env()
        model = build_model(
            env,
            seed=args.seed,
            device=args.device,
        )
        vec_env = validate_training_stack(model, env)

        print(f'policy: {POLICY_NAME}')
        print(f'vec_env_type: {type(vec_env).__name__}')
        print('masking_supported: True')
        print(f'session_after_model: {env.session}')

        if args.validate_only:
            print('RL_TRAIN_ENTRYPOINT_VALIDATION_PASS')
            return 0

        print(f'training_requested_timesteps: {args.train}')
        model.learn(total_timesteps=args.train)
        print('RL_TRAIN_COMPLETE')
        return 0

    finally:
        if model is not None:
            model_env = model.get_env()
            if model_env is not None:
                model_env.close()
            elif env is not None:
                env.close()
        elif env is not None:
            env.close()


if __name__ == '__main__':
    raise SystemExit(main())
