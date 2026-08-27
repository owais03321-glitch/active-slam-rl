import numpy as np


class RlGymStepBridge:
    """Execute one synchronous Gym action over live ROS navigation."""

    def __init__(
        self,
        *,
        start_action,
        complete_action,
        observation_sync,
        truncation_check=None,
    ):
        self.start_action = start_action
        self.complete_action = complete_action
        self.observation_sync = observation_sync

        if (
            truncation_check is not None
            and not callable(
                truncation_check
            )
        ):
            raise TypeError(
                'truncation_check must be callable.'
            )

        self.truncation_check = (
            truncation_check
        )

    def step(
        self,
        action,
    ):
        """Execute one goal-aligned physical RL transition."""

        self.start_action(
            action
        )

        outcome = self.complete_action(
            timeout=None
        )

        if outcome is None:
            raise RuntimeError(
                'Blocking RL action completion returned no '
                'terminal outcome.'
            )

        observation = (
            self.observation_sync()
        )

        navigation = outcome.navigation
        transition = outcome.transition
        metrics = transition.metrics

        reward = float(
            metrics.reward
        )

        action_mask = np.asarray(
            observation[
                'action_mask'
            ],
            dtype=bool,
        ).reshape(-1)

        frontier_exhausted = (
            np.count_nonzero(
                action_mask
            )
            == 0
        )

        horizon_due = False

        if self.truncation_check is not None:
            horizon_due = bool(
                self.truncation_check()
            )

        truncated = (
            bool(
                outcome.truncated
            )
            or horizon_due
        )

        terminated = (
            frontier_exhausted
            and not truncated
        )

        info = {
            'action': int(
                transition.start.action
            ),
            'goal_x': float(
                transition.start.goal_x
            ),
            'goal_y': float(
                transition.start.goal_y
            ),
            'area_gain_m2': float(
                metrics.area_gain_m2
            ),
            'path_delta_m': float(
                metrics.path_delta_m
            ),
            'navigation_accepted': bool(
                navigation.accepted
            ),
            'navigation_status': (
                navigation.status
            ),
            'navigation_succeeded': bool(
                navigation.succeeded
            ),
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )
