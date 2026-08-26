from dataclasses import dataclass

from active_slam_rl.rl_reward import (
    DEFAULT_PATH_COST_PER_M,
    compute_exploration_reward,
)


@dataclass(frozen=True)
class TransitionMetrics:
    """Goal-aligned physical measurements for one RL transition."""

    area_gain_m2: float
    path_delta_m: float
    reward: float


def compute_transition_metrics(
    *,
    start_area_m2,
    end_area_m2,
    start_path_m,
    end_path_m,
    path_cost_per_m=DEFAULT_PATH_COST_PER_M,
):
    """Compute one goal-aligned exploration transition."""

    values = {
        'start_area_m2': start_area_m2,
        'end_area_m2': end_area_m2,
        'start_path_m': start_path_m,
        'end_path_m': end_path_m,
    }

    for name, value in values.items():
        if value < 0.0:
            raise ValueError(
                f'{name} must be nonnegative.'
            )

    if end_path_m < start_path_m:
        raise ValueError(
            'end_path_m must not be less than start_path_m.'
        )

    area_gain_m2 = max(
        0.0,
        float(end_area_m2)
        - float(start_area_m2),
    )

    path_delta_m = (
        float(end_path_m)
        - float(start_path_m)
    )

    reward = compute_exploration_reward(
        area_gain_m2=area_gain_m2,
        path_delta_m=path_delta_m,
        path_cost_per_m=path_cost_per_m,
    )

    return TransitionMetrics(
        area_gain_m2=area_gain_m2,
        path_delta_m=path_delta_m,
        reward=reward,
    )


@dataclass(frozen=True)
class TransitionStart:
    """Immutable snapshot taken when an RL navigation action begins."""

    action: int
    goal_x: float
    goal_y: float
    area_m2: float
    path_m: float


@dataclass(frozen=True)
class CompletedTransition:
    """Completed transition with its frozen start and physical outcome."""

    start: TransitionStart
    metrics: TransitionMetrics


class GoalTransitionTracker:
    """Track at most one goal-aligned RL transition at a time."""

    def __init__(self):
        self._start = None

    @property
    def active(self):
        return self._start is not None

    @property
    def start(self):
        return self._start

    def begin(
        self,
        *,
        action,
        goal_x,
        goal_y,
        area_m2,
        path_m,
    ):
        """Freeze action and measurements at navigation start."""

        if self.active:
            raise RuntimeError(
                'A transition is already active.'
            )

        if action < 0:
            raise ValueError(
                'action must be nonnegative.'
            )

        if area_m2 < 0.0:
            raise ValueError(
                'area_m2 must be nonnegative.'
            )

        if path_m < 0.0:
            raise ValueError(
                'path_m must be nonnegative.'
            )

        self._start = TransitionStart(
            action=int(action),
            goal_x=float(goal_x),
            goal_y=float(goal_y),
            area_m2=float(area_m2),
            path_m=float(path_m),
        )

        return self._start

    def complete(
        self,
        *,
        end_area_m2,
        end_path_m,
    ):
        """Finish the active transition from cumulative measurements."""

        if self._start is None:
            raise RuntimeError(
                'No transition is active.'
            )

        metrics = compute_transition_metrics(
            start_area_m2=self._start.area_m2,
            end_area_m2=end_area_m2,
            start_path_m=self._start.path_m,
            end_path_m=end_path_m,
        )

        completed = CompletedTransition(
            start=self._start,
            metrics=metrics,
        )

        self._start = None

        return completed

    def reset(self):
        """Discard any active transition."""

        self._start = None
