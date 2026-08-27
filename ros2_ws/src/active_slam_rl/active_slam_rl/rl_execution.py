from dataclasses import dataclass


@dataclass(frozen=True)
class RlActionOutcome:
    """Completed RL action with Nav2 and physical transition outcomes."""

    navigation: object
    transition: object


class RlActionCoordinator:
    """Coordinate one RL frontier action with transition accounting."""

    def __init__(
        self,
        *,
        env,
        transition_tracker,
        nav_executor,
        visited_goals,
    ):
        self.env = env
        self.transition_tracker = transition_tracker
        self.nav_executor = nav_executor
        self.visited_goals = visited_goals

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

    def complete_action(
        self,
        *,
        measurement_provider,
        timeout=None,
    ):
        """Wait for Nav2, then sample and complete the RL transition."""

        if not self.transition_tracker.active:
            raise RuntimeError(
                'No RL action transition is active.'
            )

        navigation = (
            self.nav_executor.wait_for_completion(
                timeout=timeout
            )
        )

        if navigation is None:
            return None

        end_area_m2, end_path_m = (
            measurement_provider()
        )

        transition = (
            self.transition_tracker.complete(
                end_area_m2=end_area_m2,
                end_path_m=end_path_m,
            )
        )

        if navigation.succeeded:
            self.visited_goals.append(
                (
                    transition.start.goal_x,
                    transition.start.goal_y,
                )
            )

        return RlActionOutcome(
            navigation=navigation,
            transition=transition,
        )
