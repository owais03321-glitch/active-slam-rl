class RlGymStepBridge:
    """Execute one synchronous Gym action over live ROS navigation."""

    def __init__(
        self,
        *,
        start_action,
        complete_action,
        observation_sync,
    ):
        self.start_action = start_action
        self.complete_action = complete_action
        self.observation_sync = observation_sync

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

        terminated = False
        truncated = False

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
