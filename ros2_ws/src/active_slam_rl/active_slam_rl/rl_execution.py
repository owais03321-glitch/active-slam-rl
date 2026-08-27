class RlActionCoordinator:
    """Coordinate one RL frontier action with transition accounting."""

    def __init__(
        self,
        *,
        env,
        transition_tracker,
        nav_executor,
    ):
        self.env = env
        self.transition_tracker = transition_tracker
        self.nav_executor = nav_executor

    def start_action(
        self,
        *,
        action,
        area_m2,
        path_m,
        stamp,
    ):
        """Freeze one selected frontier and begin its Nav2 lifecycle."""

        candidate = self.env.candidate_for_action(
            action
        )

        start = self.transition_tracker.begin(
            action=action,
            goal_x=candidate.world_x,
            goal_y=candidate.world_y,
            area_m2=area_m2,
            path_m=path_m,
        )

        try:
            self.nav_executor.start(
                x=start.goal_x,
                y=start.goal_y,
                stamp=stamp,
            )

        except Exception:
            self.transition_tracker.reset()
            raise

        return start
