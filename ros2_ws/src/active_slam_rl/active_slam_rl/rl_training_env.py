import gymnasium as gym

from active_slam_rl.rl_env import ActiveSlamEnv
from active_slam_rl.rl_session import FreshRlSession


class FreshSessionEnv(gym.Env):
    """Gym environment that replaces the physical RL session on reset."""

    metadata = ActiveSlamEnv.metadata

    def __init__(
        self,
        *,
        session_factory=FreshRlSession,
        max_start_attempts=3,
    ):
        super().__init__()

        if not callable(
            session_factory
        ):
            raise TypeError(
                'session_factory must be callable.'
            )

        self._session_factory = (
            session_factory
        )

        if (
            not isinstance(
                max_start_attempts,
                int,
            )
            or isinstance(
                max_start_attempts,
                bool,
            )
            or max_start_attempts <= 0
        ):
            raise ValueError(
                'max_start_attempts must be a '
                'positive integer.'
            )

        self.max_start_attempts = (
            max_start_attempts
        )

        # Stable Gym spaces must exist before the first physical
        # reset because SB3 inspects them while constructing a model.
        template_env = ActiveSlamEnv()

        self.action_space = (
            template_env.action_space
        )

        self.observation_space = (
            template_env.observation_space
        )

        self._session = None

    @property
    def session(self):
        """Return the currently active physical session, if any."""

        return self._session

    @property
    def live_env(self):
        """Return the current session's live ActiveSlamEnv."""

        if self._session is None:
            return None

        return self._session.env

    def _require_live_env(self):
        live_env = self.live_env

        if live_env is None:
            raise RuntimeError(
                'FreshSessionEnv requires reset() '
                'before live interaction.'
            )

        return live_env

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        """Replace the complete physical simulator/SLAM episode."""

        super().reset(
            seed=seed
        )

        previous_session = (
            self._session
        )

        # Fail closed. Once reset begins, the old episode must never
        # remain usable even if its cleanup or replacement fails.
        self._session = None

        if previous_session is not None:
            previous_session.close()

        for attempt in range(
            1,
            self.max_start_attempts + 1,
        ):
            new_session = (
                self._session_factory()
            )

            try:
                new_session.start()

                observation = (
                    new_session.initial_observation
                )

                if observation is None:
                    raise RuntimeError(
                        'Fresh RL session did not provide '
                        'an initial observation.'
                    )

            except TimeoutError:
                new_session.close()

                print(
                    'FRESH_SESSION_START_TIMEOUT '
                    f'attempt={attempt}/'
                    f'{self.max_start_attempts}'
                )

                if (
                    attempt
                    >= self.max_start_attempts
                ):
                    raise

                continue

            except Exception:
                new_session.close()
                raise

            self._session = (
                new_session
            )

            if attempt > 1:
                print(
                    'FRESH_SESSION_START_RECOVERED '
                    f'attempt={attempt}/'
                    f'{self.max_start_attempts}'
                )

            return (
                observation,
                {},
            )

        raise RuntimeError(
            'Fresh session retry loop exited '
            'without a result.'
        )

    def step(
        self,
        action,
    ):
        """Delegate one synchronous RL transition to the live session."""

        return (
            self._require_live_env()
            .step(
                action
            )
        )

    def action_masks(self):
        """Expose the live MaskablePPO valid-action mask."""

        return (
            self._require_live_env()
            .action_masks()
        )

    def close(self):
        """Close the current physical episode idempotently."""

        session = (
            self._session
        )

        self._session = None

        if session is not None:
            session.close()
