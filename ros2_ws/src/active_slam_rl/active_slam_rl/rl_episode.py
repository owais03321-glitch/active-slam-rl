import time


DEFAULT_EPISODE_HORIZON_S = 300.0


class EpisodeTimeLimit:
    """Track the wall-clock truncation horizon for one RL episode."""

    def __init__(
        self,
        *,
        horizon_s=DEFAULT_EPISODE_HORIZON_S,
        clock=None,
    ):
        if horizon_s <= 0.0:
            raise ValueError(
                'horizon_s must be greater than zero.'
            )

        self.horizon_s = float(
            horizon_s
        )

        self._clock = (
            clock
            if clock is not None
            else time.monotonic
        )

        self._start_time = None

    @property
    def active(self):
        """Return whether the episode clock has started."""

        return self._start_time is not None

    @property
    def start_time(self):
        """Return the captured episode start time, if active."""

        return self._start_time

    def start(self):
        """Start the episode clock once without restarting it."""

        if self._start_time is None:
            self._start_time = float(
                self._clock()
            )

        return self._start_time

    @property
    def elapsed_s(self):
        """Return elapsed episode time without mutating state."""

        if self._start_time is None:
            return 0.0

        now = float(
            self._clock()
        )

        if now < self._start_time:
            raise RuntimeError(
                'Episode clock moved backwards.'
            )

        return (
            now
            - self._start_time
        )

    @property
    def truncated(self):
        """Return whether the time limit has been reached."""

        return (
            self.active
            and self.elapsed_s
            >= self.horizon_s
        )

    def reset(self):
        """Re-arm the time limit for a genuinely fresh episode."""

        self._start_time = None
