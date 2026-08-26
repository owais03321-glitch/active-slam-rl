import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ActiveSlamEnv(gym.Env):
    """Gymnasium environment scaffold for RL-based Active SLAM."""

    metadata = {
        'render_modes': [],
        'render_fps': 0,
    }

    def __init__(self):
        super().__init__()

        # Temporary API-level spaces.
        #
        # These are intentionally minimal. They will be replaced only after
        # the ROS observation and exploration-action contracts are defined
        # and tested.
        self.observation_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(1)

        self._observation = np.array(
            [0.0],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._observation = np.array(
            [0.0],
            dtype=np.float32,
        )

        info = {}

        return self._observation.copy(), info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(
                f'Invalid action {action!r} for '
                f'{self.action_space}.'
            )

        observation = self._observation.copy()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )


def main():
    env = ActiveSlamEnv()

    observation, info = env.reset(seed=0)

    print(f'observation={observation}')
    print(f'info={info}')
    print(f'action_space={env.action_space}')
    print(f'observation_space={env.observation_space}')

    env.close()


if __name__ == '__main__':
    main()
