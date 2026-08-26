import math


DEFAULT_MAX_ODOM_STEP_M = 1.0


class PathLengthTracker:
    """Accumulate physical path length from consecutive XY positions."""

    def __init__(
        self,
        *,
        max_step_m=DEFAULT_MAX_ODOM_STEP_M,
    ):
        if max_step_m <= 0.0:
            raise ValueError(
                'max_step_m must be greater than zero.'
            )

        self.max_step_m = float(
            max_step_m
        )

        self.reset()

    def reset(self):
        """Clear accumulated path and require a new origin sample."""

        self.last_xy = None
        self.path_length_m = 0.0

    def update(
        self,
        *,
        x,
        y,
    ):
        """Process one position and return the distance counted."""

        current_xy = (
            float(x),
            float(y),
        )

        if self.last_xy is None:
            self.last_xy = current_xy
            return 0.0

        last_x, last_y = self.last_xy

        step_distance = math.hypot(
            current_xy[0] - last_x,
            current_xy[1] - last_y,
        )

        self.last_xy = current_xy

        if step_distance > self.max_step_m:
            return 0.0

        self.path_length_m += step_distance

        return step_distance
