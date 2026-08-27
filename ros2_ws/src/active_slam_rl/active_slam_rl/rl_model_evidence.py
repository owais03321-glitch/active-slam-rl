import hashlib

from sb3_contrib import MaskablePPO


CORE_TRAIN_TELEMETRY = {
    'entropy_loss': (
        'train/entropy_loss'
    ),
    'policy_gradient_loss': (
        'train/policy_gradient_loss'
    ),
    'value_loss': (
        'train/value_loss'
    ),
    'approx_kl': (
        'train/approx_kl'
    ),
    'clip_fraction': (
        'train/clip_fraction'
    ),
    'loss': (
        'train/loss'
    ),
    'explained_variance': (
        'train/explained_variance'
    ),
    'clip_range': (
        'train/clip_range'
    ),
}


def _python_scalar(
    value,
):
    if hasattr(
        value,
        'item',
    ):
        try:
            return value.item()

        except (
            ValueError,
            TypeError,
        ):
            pass

    return value


def _json_safe(
    value,
):
    value = _python_scalar(
        value
    )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _json_safe(
                nested
            )
            for key, nested
            in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            _json_safe(
                nested
            )
            for nested in value
        ]

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    return str(
        value
    )


def policy_fingerprint(
    policy,
):
    """Return deterministic SHA-256 over named policy tensors."""

    state = policy.state_dict()

    digest = hashlib.sha256()

    for name in sorted(
        state
    ):
        tensor = (
            state[
                name
            ]
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            name.encode(
                'utf-8'
            )
        )

        digest.update(
            b'\0'
        )

        digest.update(
            str(
                tensor.dtype
            ).encode(
                'utf-8'
            )
        )

        digest.update(
            b'\0'
        )

        digest.update(
            str(
                tuple(
                    tensor.shape
                )
            ).encode(
                'utf-8'
            )
        )

        digest.update(
            b'\0'
        )

        digest.update(
            tensor.numpy()
            .tobytes()
        )

        digest.update(
            b'\0'
        )

    return digest.hexdigest()


def resolved_maskable_ppo_config(
    model,
):
    """Capture the actual resolved optimization configuration."""

    optimizer = (
        model.policy.optimizer
    )

    clip_range_vf = (
        None
        if model.clip_range_vf
        is None
        else float(
            model.clip_range_vf(
                1.0
            )
        )
    )

    return {
        'algorithm': (
            type(
                model
            ).__name__
        ),
        'policy_class': (
            type(
                model.policy
            ).__name__
        ),
        'seed': _json_safe(
            getattr(
                model,
                'seed',
                None,
            )
        ),
        'device': str(
            model.device
        ),
        'n_steps': int(
            model.n_steps
        ),
        'batch_size': int(
            model.batch_size
        ),
        'n_envs': int(
            model.n_envs
        ),
        'n_epochs': int(
            model.n_epochs
        ),
        'gamma': float(
            model.gamma
        ),
        'gae_lambda': float(
            model.gae_lambda
        ),
        'normalize_advantage': bool(
            model.normalize_advantage
        ),
        'ent_coef': float(
            model.ent_coef
        ),
        'vf_coef': float(
            model.vf_coef
        ),
        'max_grad_norm': float(
            model.max_grad_norm
        ),
        'target_kl': (
            None
            if model.target_kl
            is None
            else float(
                model.target_kl
            )
        ),
        'learning_rate_at_progress_1': float(
            model.lr_schedule(
                1.0
            )
        ),
        'clip_range_at_progress_1': float(
            model.clip_range(
                1.0
            )
        ),
        'clip_range_vf_at_progress_1': (
            clip_range_vf
        ),
        'optimizer_class': (
            type(
                optimizer
            ).__name__
        ),
        'optimizer_defaults': (
            _json_safe(
                optimizer.defaults
            )
        ),
        'policy_parameter_count': int(
            sum(
                parameter.numel()
                for parameter
                in model.policy.parameters()
            )
        ),
        'policy_net_arch': (
            _json_safe(
                getattr(
                    model.policy,
                    'net_arch',
                    None,
                )
            )
        ),
        'initial_n_updates': int(
            model._n_updates
        ),
    }


def extract_train_telemetry(
    model,
):
    """Extract required MaskablePPO telemetry after train()."""

    values = (
        model.logger.name_to_value
    )

    missing = [
        logger_key
        for logger_key
        in CORE_TRAIN_TELEMETRY.values()
        if logger_key not in values
    ]

    if missing:
        raise RuntimeError(
            'MaskablePPO telemetry is missing: '
            + ', '.join(
                sorted(
                    missing
                )
            )
        )

    payload = {
        output_key: float(
            _python_scalar(
                values[
                    logger_key
                ]
            )
        )
        for output_key, logger_key
        in CORE_TRAIN_TELEMETRY.items()
    }

    optimizer = (
        model.policy.optimizer
    )

    param_groups = (
        optimizer.param_groups
    )

    if not param_groups:
        raise RuntimeError(
            'MaskablePPO optimizer has no parameter groups.'
        )

    learning_rates = {
        float(
            group[
                'lr'
            ]
        )
        for group
        in param_groups
    }

    if len(
        learning_rates
    ) != 1:
        raise RuntimeError(
            'MaskablePPO optimizer parameter groups '
            'do not share one auditable learning rate.'
        )

    payload[
        'learning_rate'
    ] = learning_rates.pop()

    payload[
        'clip_range_vf'
    ] = (
        None
        if 'train/clip_range_vf'
        not in values
        else float(
            _python_scalar(
                values[
                    'train/clip_range_vf'
                ]
            )
        )
    )

    payload[
        'num_timesteps'
    ] = int(
        model.num_timesteps
    )

    payload[
        'n_updates'
    ] = int(
        model._n_updates
    )

    payload[
        'progress_remaining'
    ] = float(
        model._current_progress_remaining
    )

    return payload


class AuditedMaskablePPO(
    MaskablePPO
):
    """MaskablePPO that emits evidence after every train() call."""

    def __init__(
        self,
        *args,
        telemetry_sink=None,
        **kwargs,
    ):
        self._telemetry_sink = (
            telemetry_sink
        )

        self._optimization_index = 0

        super().__init__(
            *args,
            **kwargs,
        )

    @property
    def optimization_index(self):
        return (
            self._optimization_index
        )

    def train(self):
        super().train()

        if self._telemetry_sink is None:
            self._optimization_index += 1
            return

        payload = (
            extract_train_telemetry(
                self
            )
        )

        payload[
            'optimization_index'
        ] = (
            self._optimization_index
        )

        payload[
            'policy_fingerprint'
        ] = (
            policy_fingerprint(
                self.policy
            )
        )

        self._telemetry_sink(
            payload
        )

        self._optimization_index += 1
