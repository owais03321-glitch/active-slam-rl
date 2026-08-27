import time

import gymnasium as gym
import numpy as np

from active_slam_rl.rl_experiment import (
    utc_now_iso,
)


REQUIRED_TRANSITION_INFO = (
    'action',
    'goal_x',
    'goal_y',
    'area_gain_m2',
    'path_delta_m',
    'navigation_accepted',
    'navigation_status',
    'navigation_succeeded',
)


def mask_to_bits(mask):
    values = np.asarray(
        mask,
        dtype=bool,
    ).reshape(-1)

    return ''.join(
        '1' if value else '0'
        for value in values
    )


class RecordedTrainingEnv(gym.Wrapper):
    """Record exact decision-level evidence around a live training env."""

    def __init__(
        self,
        env,
        *,
        recorder,
        clock=None,
    ):
        super().__init__(
            env
        )

        if not callable(
            getattr(
                recorder,
                'record_step',
                None,
            )
        ):
            raise TypeError(
                'recorder must provide record_step().'
            )

        if not callable(
            getattr(
                recorder,
                'record_episode',
                None,
            )
        ):
            raise TypeError(
                'recorder must provide record_episode().'
            )

        self.recorder = recorder

        self._clock = (
            clock
            if clock is not None
            else time.monotonic
        )

        self._episode_index = -1
        self._episode_open = False
        self._episode_steps = 0
        self._episode_return = 0.0

        self._episode_reset_at_utc = None
        self._initial_area_m2 = None
        self._initial_path_m = None

        self._navigation_successes = 0
        self._navigation_failures = 0

        self._total_steps = 0
        self._total_reward = 0.0

        self._closed = False
        self._decision_mask = None

    @property
    def session(self):
        return getattr(
            self.env,
            'session',
            None,
        )

    @property
    def live_env(self):
        return getattr(
            self.env,
            'live_env',
            None,
        )

    @property
    def total_recorded_steps(self):
        return self._total_steps

    @property
    def total_recorded_reward(self):
        return self._total_reward

    @property
    def episode_index(self):
        return self._episode_index

    @staticmethod
    def _observation_mask(
        observation,
    ):
        if 'action_mask' not in observation:
            raise RuntimeError(
                'Agent observation has no action_mask.'
            )

        return np.asarray(
            observation[
                'action_mask'
            ],
            dtype=bool,
        ).reshape(-1).copy()

    def _latch_decision_mask(
        self,
        observation,
    ):
        mask = self._observation_mask(
            observation
        )

        expected_size = getattr(
            self.action_space,
            'n',
            mask.size,
        )

        if mask.size != expected_size:
            raise RuntimeError(
                'Observation action mask has '
                'unexpected size.'
            )

        self._decision_mask = (
            mask.copy()
        )

        return mask

    def action_masks(self):
        if self._decision_mask is None:
            raise RuntimeError(
                'No agent decision mask is latched; '
                'reset() is required.'
            )

        return (
            self._decision_mask
            .copy()
        )

    def _require_node(self):
        session = self.session

        if session is None:
            raise RuntimeError(
                'Recorded live evidence requires an '
                'active physical session.'
            )

        node = getattr(
            session,
            'node',
            None,
        )

        if node is None:
            raise RuntimeError(
                'Active physical session has no '
                'observation node.'
            )

        return node

    def _physical_snapshot(self):
        node = self._require_node()

        area_m2 = getattr(
            node,
            'latest_explored_area_m2',
            None,
        )

        if area_m2 is None:
            raise RuntimeError(
                'Explored-area evidence is unavailable.'
            )

        path_tracker = getattr(
            node,
            'path_tracker',
            None,
        )

        if path_tracker is None:
            raise RuntimeError(
                'Path evidence tracker is unavailable.'
            )

        path_m = getattr(
            path_tracker,
            'path_length_m',
            None,
        )

        if path_m is None:
            raise RuntimeError(
                'Cumulative path evidence is unavailable.'
            )

        robot_xy = getattr(
            node,
            'latest_robot_xy',
            None,
        )

        if robot_xy is None:
            raise RuntimeError(
                'Robot-position evidence is unavailable.'
            )

        episode_time_limit = getattr(
            node,
            'episode_time_limit',
            None,
        )

        if episode_time_limit is None:
            raise RuntimeError(
                'Episode-time evidence is unavailable.'
            )

        map_revision = getattr(
            node,
            'map_revision',
            None,
        )

        if map_revision is None:
            raise RuntimeError(
                'Map-revision evidence is unavailable.'
            )

        robot_x, robot_y = robot_xy

        return {
            'episode_elapsed_s': float(
                episode_time_limit.elapsed_s
            ),
            'explored_area_m2': float(
                area_m2
            ),
            'cumulative_path_m': float(
                path_m
            ),
            'robot_x': float(
                robot_x
            ),
            'robot_y': float(
                robot_y
            ),
            'map_revision': int(
                map_revision
            ),
        }

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        if self._closed:
            raise RuntimeError(
                'RecordedTrainingEnv is closed.'
            )

        if self._episode_open:
            raise RuntimeError(
                'Cannot reset while the recorded '
                'episode is still open.'
            )

        observation, info = (
            self.env.reset(
                seed=seed,
                options=options,
            )
        )

        self._latch_decision_mask(
            observation
        )

        snapshot = (
            self._physical_snapshot()
        )

        self._episode_index += 1
        self._episode_open = True
        self._episode_steps = 0
        self._episode_return = 0.0

        self._episode_reset_at_utc = (
            utc_now_iso()
        )

        self._initial_area_m2 = (
            snapshot[
                'explored_area_m2'
            ]
        )

        self._initial_path_m = (
            snapshot[
                'cumulative_path_m'
            ]
        )

        self._navigation_successes = 0
        self._navigation_failures = 0

        return (
            observation,
            info,
        )

    def _record_episode(
        self,
        *,
        outcome,
        terminated,
        truncated,
    ):
        if not self._episode_open:
            return

        snapshot = (
            self._physical_snapshot()
        )

        self.recorder.record_episode(
            {
                'episode_index': (
                    self._episode_index
                ),
                'episode_reset_at_utc': (
                    self._episode_reset_at_utc
                ),
                'steps': (
                    self._episode_steps
                ),
                'episode_return': (
                    self._episode_return
                ),
                'initial_explored_area_m2': (
                    self._initial_area_m2
                ),
                'final_explored_area_m2': (
                    snapshot[
                        'explored_area_m2'
                    ]
                ),
                'initial_path_m': (
                    self._initial_path_m
                ),
                'final_path_m': (
                    snapshot[
                        'cumulative_path_m'
                    ]
                ),
                'final_episode_elapsed_s': (
                    snapshot[
                        'episode_elapsed_s'
                    ]
                ),
                'navigation_successes': (
                    self._navigation_successes
                ),
                'navigation_failures': (
                    self._navigation_failures
                ),
                'terminated': bool(
                    terminated
                ),
                'truncated': bool(
                    truncated
                ),
                'outcome': outcome,
            }
        )

        self._episode_open = False

    def step(
        self,
        action,
    ):
        if self._closed:
            raise RuntimeError(
                'RecordedTrainingEnv is closed.'
            )

        if not self._episode_open:
            raise RuntimeError(
                'reset() is required before step().'
            )

        action_index = int(
            action
        )

        pre_mask = (
            self.action_masks()
        )

        if (
            action_index < 0
            or action_index
            >= pre_mask.size
            or not pre_mask[
                action_index
            ]
        ):
            raise RuntimeError(
                'Training attempted an action that '
                'was invalid in the recorded mask.'
            )

        valid_action_count = int(
            np.count_nonzero(
                pre_mask
            )
        )

        step_started_at_utc = (
            utc_now_iso()
        )

        start_clock = float(
            self._clock()
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(
            action_index
        )

        end_clock = float(
            self._clock()
        )

        step_duration_s = (
            end_clock
            - start_clock
        )

        if step_duration_s < 0.0:
            raise RuntimeError(
                'Step clock moved backwards.'
            )

        missing = [
            field
            for field
            in REQUIRED_TRANSITION_INFO
            if field not in info
        ]

        if missing:
            raise RuntimeError(
                'Transition evidence is missing: '
                + ', '.join(
                    missing
                )
            )

        if int(
            info['action']
        ) != action_index:
            raise RuntimeError(
                'Executed action does not match '
                'recorded requested action.'
            )

        next_mask = (
            self._latch_decision_mask(
                observation
            )
        )

        snapshot = (
            self._physical_snapshot()
        )

        reward_value = float(
            reward
        )

        self._episode_steps += 1
        self._episode_return += (
            reward_value
        )

        self._total_steps += 1
        self._total_reward += (
            reward_value
        )

        navigation_succeeded = bool(
            info[
                'navigation_succeeded'
            ]
        )

        if navigation_succeeded:
            self._navigation_successes += 1
        else:
            self._navigation_failures += 1

        self.recorder.record_step(
            {
                'episode_index': (
                    self._episode_index
                ),
                'step_started_at_utc': (
                    step_started_at_utc
                ),
                'step_duration_s': (
                    step_duration_s
                ),
                'episode_elapsed_s': (
                    snapshot[
                        'episode_elapsed_s'
                    ]
                ),
                'action': (
                    action_index
                ),
                'action_mask_bits': (
                    mask_to_bits(
                        pre_mask
                    )
                ),
                'valid_action_count': (
                    valid_action_count
                ),
                'goal_x': float(
                    info[
                        'goal_x'
                    ]
                ),
                'goal_y': float(
                    info[
                        'goal_y'
                    ]
                ),
                'reward': (
                    reward_value
                ),
                'cumulative_episode_return': (
                    self._episode_return
                ),
                'area_gain_m2': float(
                    info[
                        'area_gain_m2'
                    ]
                ),
                'explored_area_m2': (
                    snapshot[
                        'explored_area_m2'
                    ]
                ),
                'path_delta_m': float(
                    info[
                        'path_delta_m'
                    ]
                ),
                'cumulative_path_m': (
                    snapshot[
                        'cumulative_path_m'
                    ]
                ),
                'robot_x': (
                    snapshot[
                        'robot_x'
                    ]
                ),
                'robot_y': (
                    snapshot[
                        'robot_y'
                    ]
                ),
                'navigation_accepted': bool(
                    info[
                        'navigation_accepted'
                    ]
                ),
                'navigation_status': (
                    info[
                        'navigation_status'
                    ]
                ),
                'navigation_succeeded': (
                    navigation_succeeded
                ),
                'map_revision': (
                    snapshot[
                        'map_revision'
                    ]
                ),
                'next_action_mask_bits': (
                    mask_to_bits(
                        next_mask
                    )
                ),
                'next_valid_action_count': int(
                    np.count_nonzero(
                        next_mask
                    )
                ),
                'terminated': bool(
                    terminated
                ),
                'truncated': bool(
                    truncated
                ),
            }
        )

        if terminated or truncated:
            if (
                terminated
                and truncated
            ):
                outcome = (
                    'terminated_and_truncated'
                )

            elif terminated:
                outcome = (
                    'terminated'
                )

            else:
                outcome = (
                    'truncated'
                )

            self._record_episode(
                outcome=outcome,
                terminated=terminated,
                truncated=truncated,
            )

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        if self._closed:
            return

        try:
            if (
                self._episode_open
                and self._episode_steps > 0
            ):
                self._record_episode(
                    outcome='training_stop',
                    terminated=False,
                    truncated=False,
                )

            else:
                self._episode_open = False

        finally:
            try:
                self.env.close()

            finally:
                self._decision_mask = None
                self._closed = True
