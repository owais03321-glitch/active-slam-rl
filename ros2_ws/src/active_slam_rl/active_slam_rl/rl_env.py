import gymnasium as gym
import numpy as np
from gymnasium import spaces

from active_slam_rl.rl_observation import (
    CANDIDATE_FEATURE_COUNT,
    DEFAULT_MAX_CANDIDATES,
    encode_frontier_observation,
)


class ActiveSlamEnv(gym.Env):
    """Gymnasium environment scaffold for RL-based Active SLAM."""

    metadata = {
        'render_modes': [],
        'render_fps': 0,
    }

    def __init__(
        self,
        *,
        max_candidates=DEFAULT_MAX_CANDIDATES,
    ):
        super().__init__()

        if max_candidates <= 0:
            raise ValueError(
                'max_candidates must be greater than zero.'
            )

        self.max_candidates = max_candidates

        float32_max = np.finfo(np.float32).max

        candidate_low = np.tile(
            np.array(
                [
                    -float32_max,  # relative x
                    -float32_max,  # relative y
                    0.0,           # distance
                    0.0,           # cluster size
                ],
                dtype=np.float32,
            ),
            (
                self.max_candidates,
                1,
            ),
        )

        candidate_high = np.full(
            (
                self.max_candidates,
                CANDIDATE_FEATURE_COUNT,
            ),
            float32_max,
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(
            {
                'candidates': spaces.Box(
                    low=candidate_low,
                    high=candidate_high,
                    dtype=np.float32,
                ),
                'action_mask': spaces.MultiBinary(
                    self.max_candidates
                ),
            }
        )

        self.action_space = spaces.Discrete(
            self.max_candidates
        )

        self._observation = (
            self._make_empty_observation()
        )

        self._candidates = []
        self._step_bridge = None

    def _make_empty_observation(self):
        return {
            'candidates': np.zeros(
                (
                    self.max_candidates,
                    CANDIDATE_FEATURE_COUNT,
                ),
                dtype=np.float32,
            ),
            'action_mask': np.zeros(
                self.max_candidates,
                dtype=np.int8,
            ),
        }

    def _copy_observation(self):
        return {
            key: value.copy()
            for key, value
            in self._observation.items()
        }

    def set_frontier_state(
        self,
        *,
        candidates,
        robot_x,
        robot_y,
    ):
        """Update the environment observation from frontier state."""

        candidate_list = list(candidates)

        self._observation = encode_frontier_observation(
            candidates=candidate_list,
            robot_x=robot_x,
            robot_y=robot_y,
            max_candidates=self.max_candidates,
        )

        self._candidates = candidate_list

        return self._copy_observation()

    def candidate_for_action(
        self,
        action,
    ):
        """Return the frontier candidate referenced by an RL action."""

        if not self.action_space.contains(action):
            raise ValueError(
                f'Invalid action {action!r} for '
                f'{self.action_space}.'
            )

        if self._observation['action_mask'][action] == 0:
            raise ValueError(
                f'Action {action} selects an unavailable '
                'frontier candidate slot.'
            )

        return self._candidates[action]

    def action_masks(self):
        """Return the current valid-action mask for MaskablePPO."""

        return self._observation[
            'action_mask'
        ].astype(
            bool,
            copy=True,
        )

    @property
    def step_bridge(self):
        """Return the synchronous live execution bridge, if bound."""

        return self._step_bridge

    def bind_step_bridge(
        self,
        step_bridge,
    ):
        """Bind synchronous live execution semantics to Gym step()."""

        if not callable(
            getattr(
                step_bridge,
                'step',
                None,
            )
        ):
            raise TypeError(
                'step_bridge must provide a callable step(action).'
            )

        self._step_bridge = step_bridge

        return step_bridge

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        self._observation = (
            self._make_empty_observation()
        )

        self._candidates = []

        info = {}

        return (
            self._copy_observation(),
            info,
        )

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(
                f'Invalid action {action!r} for '
                f'{self.action_space}.'
            )

        if self._step_bridge is None:
            raise RuntimeError(
                'ActiveSlamEnv has no synchronous step bridge.'
            )

        return self._step_bridge.step(
            action
        )


def main():
    env = ActiveSlamEnv()

    observation, info = env.reset(seed=0)

    print(
        f'candidate_shape='
        f'{observation["candidates"].shape}'
    )
    print(
        f'action_mask_shape='
        f'{observation["action_mask"].shape}'
    )
    print(f'info={info}')
    print(f'action_space={env.action_space}')
    print(
        f'observation_space='
        f'{env.observation_space}'
    )

    env.close()


if __name__ == '__main__':
    main()
