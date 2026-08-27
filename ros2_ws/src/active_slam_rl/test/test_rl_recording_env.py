import csv

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from active_slam_rl.rl_experiment import (
    ExperimentRecorder,
)
from active_slam_rl.rl_recording_env import (
    RecordedTrainingEnv,
)


def fixed_provenance():
    return {
        'captured_at_utc': (
            '2026-08-27T00:00:00+00:00'
        ),
        'git_commit': 'test-commit',
        'git_branch': 'rl-active-slam',
        'git_status_at_start': '',
        'git_worktree_clean': True,
        'python_executable': (
            '/project/.venv/bin/python3'
        ),
        'python_version': '3.12.3',
        'platform': 'test-platform',
        'package_versions': {
            'sb3-contrib': '2.9.0',
            'stable-baselines3': '2.9.0',
            'gymnasium': '1.3.0',
            'torch': '2.9.1+cpu',
        },
    }


class FakeEpisodeTimeLimit:
    def __init__(self):
        self.elapsed_s = 0.0


class FakePathTracker:
    def __init__(self):
        self.path_length_m = 0.0


class FakeNode:
    def __init__(self):
        self.map_revision = 1

        self.latest_explored_area_m2 = (
            1.0
        )

        self.latest_robot_xy = (
            0.0,
            0.0,
        )

        self.path_tracker = (
            FakePathTracker()
        )

        self.episode_time_limit = (
            FakeEpisodeTimeLimit()
        )


class FakeSession:
    def __init__(self):
        self.node = FakeNode()


class FakeLiveEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = (
            spaces.Discrete(
                4
            )
        )

        self.observation_space = (
            spaces.Dict(
                {
                    'action_mask': (
                        spaces.MultiBinary(
                            4
                        )
                    ),
                }
            )
        )

        self.session = (
            FakeSession()
        )

        self.step_calls = []
        self.closed = False

        self.terminated = False
        self.truncated = False

        self.omit_info_field = None

        self._mask = np.array(
            [
                1,
                0,
                1,
                0,
            ],
            dtype=np.int8,
        )

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(
            seed=seed
        )

        self.session = (
            FakeSession()
        )

        self._mask = np.array(
            [
                1,
                0,
                1,
                0,
            ],
            dtype=np.int8,
        )

        return (
            {
                'action_mask': (
                    self._mask.copy()
                ),
            },
            {},
        )

    def action_masks(self):
        return self._mask.astype(
            bool,
            copy=True,
        )

    def step(
        self,
        action,
    ):
        self.step_calls.append(
            int(action)
        )

        node = (
            self.session.node
        )

        node.map_revision = 4

        node.latest_explored_area_m2 = (
            3.5
        )

        node.latest_robot_xy = (
            0.5,
            -0.2,
        )

        node.path_tracker.path_length_m = (
            1.25
        )

        node.episode_time_limit.elapsed_s = (
            7.0
        )

        self._mask = np.array(
            [
                1,
                1,
                0,
                0,
            ],
            dtype=np.int8,
        )

        info = {
            'action': int(
                action
            ),
            'goal_x': 0.8,
            'goal_y': -0.4,
            'area_gain_m2': 2.5,
            'path_delta_m': 1.25,
            'navigation_accepted': True,
            'navigation_status': 4,
            'navigation_succeeded': True,
        }

        if (
            self.omit_info_field
            is not None
        ):
            info.pop(
                self.omit_info_field
            )

        return (
            {
                'action_mask': (
                    self._mask.copy()
                ),
            },
            2.375,
            self.terminated,
            self.truncated,
            info,
        )

    def close(self):
        self.closed = True


def make_recorder(
    tmp_path,
    run_id='diag_recording',
):
    return ExperimentRecorder(
        evidence_root=tmp_path,
        run_id=run_id,
        run_kind='diagnostic',
        config={
            'seed': 0,
            'n_steps': 2,
            'batch_size': 2,
        },
        provenance=fixed_provenance(),
    )


def read_csv(
    path,
):
    with path.open(
        newline='',
    ) as file_handle:
        return list(
            csv.DictReader(
                file_handle
            )
        )


def test_records_complete_decision_evidence(
    tmp_path,
):
    recorder = make_recorder(
        tmp_path
    )

    physical_env = (
        FakeLiveEnv()
    )

    times = iter(
        [
            10.0,
            12.5,
        ]
    )

    env = RecordedTrainingEnv(
        physical_env,
        recorder=recorder,
        clock=lambda: next(
            times
        ),
    )

    env.reset()

    observation, reward, terminated, truncated, info = (
        env.step(
            0
        )
    )

    assert reward == 2.375
    assert terminated is False
    assert truncated is False

    rows = read_csv(
        recorder.steps_path
    )

    assert len(rows) == 1

    row = rows[0]

    assert row[
        'step_index'
    ] == '0'

    assert row[
        'episode_index'
    ] == '0'

    assert row[
        'action'
    ] == '0'

    assert row[
        'action_mask_bits'
    ] == '1010'

    assert row[
        'valid_action_count'
    ] == '2'

    assert row[
        'next_action_mask_bits'
    ] == '1100'

    assert row[
        'next_valid_action_count'
    ] == '2'

    assert float(
        row[
            'step_duration_s'
        ]
    ) == pytest.approx(
        2.5
    )

    assert float(
        row[
            'episode_elapsed_s'
        ]
    ) == pytest.approx(
        7.0
    )

    assert float(
        row[
            'explored_area_m2'
        ]
    ) == pytest.approx(
        3.5
    )

    assert float(
        row[
            'area_gain_m2'
        ]
    ) == pytest.approx(
        2.5
    )

    assert float(
        row[
            'cumulative_path_m'
        ]
    ) == pytest.approx(
        1.25
    )

    assert float(
        row[
            'path_delta_m'
        ]
    ) == pytest.approx(
        1.25
    )

    assert float(
        row[
            'robot_x'
        ]
    ) == pytest.approx(
        0.5
    )

    assert float(
        row[
            'robot_y'
        ]
    ) == pytest.approx(
        -0.2
    )

    assert row[
        'navigation_accepted'
    ] == 'True'

    assert row[
        'navigation_status'
    ] == '4'

    assert row[
        'navigation_succeeded'
    ] == 'True'

    assert row[
        'map_revision'
    ] == '4'

    assert row[
        'terminated'
    ] == 'False'

    assert row[
        'truncated'
    ] == 'False'

    assert (
        row[
            'step_started_at_utc'
        ]
    )

    env.close()


def test_terminal_step_records_episode_summary_once(
    tmp_path,
):
    recorder = make_recorder(
        tmp_path
    )

    physical_env = (
        FakeLiveEnv()
    )

    physical_env.terminated = (
        True
    )

    times = iter(
        [
            1.0,
            2.0,
        ]
    )

    env = RecordedTrainingEnv(
        physical_env,
        recorder=recorder,
        clock=lambda: next(
            times
        ),
    )

    env.reset()

    env.step(
        0
    )

    rows = read_csv(
        recorder.episodes_path
    )

    assert len(rows) == 1

    row = rows[0]

    assert row[
        'episode_index'
    ] == '0'

    assert row[
        'steps'
    ] == '1'

    assert float(
        row[
            'episode_return'
        ]
    ) == pytest.approx(
        2.375
    )

    assert float(
        row[
            'initial_explored_area_m2'
        ]
    ) == pytest.approx(
        1.0
    )

    assert float(
        row[
            'final_explored_area_m2'
        ]
    ) == pytest.approx(
        3.5
    )

    assert float(
        row[
            'initial_path_m'
        ]
    ) == pytest.approx(
        0.0
    )

    assert float(
        row[
            'final_path_m'
        ]
    ) == pytest.approx(
        1.25
    )

    assert float(
        row[
            'final_episode_elapsed_s'
        ]
    ) == pytest.approx(
        7.0
    )

    assert row[
        'navigation_successes'
    ] == '1'

    assert row[
        'navigation_failures'
    ] == '0'

    assert row[
        'terminated'
    ] == 'True'

    assert row[
        'truncated'
    ] == 'False'

    assert row[
        'outcome'
    ] == 'terminated'

    env.close()

    rows_after_close = read_csv(
        recorder.episodes_path
    )

    assert len(
        rows_after_close
    ) == 1


def test_close_records_partial_training_stop(
    tmp_path,
):
    recorder = make_recorder(
        tmp_path
    )

    physical_env = (
        FakeLiveEnv()
    )

    times = iter(
        [
            1.0,
            2.0,
        ]
    )

    env = RecordedTrainingEnv(
        physical_env,
        recorder=recorder,
        clock=lambda: next(
            times
        ),
    )

    env.reset()

    env.step(
        0
    )

    env.close()

    rows = read_csv(
        recorder.episodes_path
    )

    assert len(rows) == 1

    assert rows[0][
        'outcome'
    ] == 'training_stop'

    assert rows[0][
        'terminated'
    ] == 'False'

    assert rows[0][
        'truncated'
    ] == 'False'

    assert (
        physical_env.closed
        is True
    )


def test_masked_action_is_blocked_before_physical_step(
    tmp_path,
):
    recorder = make_recorder(
        tmp_path
    )

    physical_env = (
        FakeLiveEnv()
    )

    env = RecordedTrainingEnv(
        physical_env,
        recorder=recorder,
    )

    env.reset()

    with pytest.raises(
        RuntimeError,
        match='invalid',
    ):
        env.step(
            1
        )

    assert (
        physical_env.step_calls
        == []
    )

    assert read_csv(
        recorder.steps_path
    ) == []

    env.close()


def test_missing_live_info_fails_closed(
    tmp_path,
):
    recorder = make_recorder(
        tmp_path
    )

    physical_env = (
        FakeLiveEnv()
    )

    physical_env.omit_info_field = (
        'goal_x'
    )

    times = iter(
        [
            1.0,
            2.0,
        ]
    )

    env = RecordedTrainingEnv(
        physical_env,
        recorder=recorder,
        clock=lambda: next(
            times
        ),
    )

    env.reset()

    with pytest.raises(
        RuntimeError,
        match='goal_x',
    ):
        env.step(
            0
        )

    assert (
        physical_env.step_calls
        == [
            0,
        ]
    )

    assert read_csv(
        recorder.steps_path
    ) == []

    env.close()


def test_agent_decision_mask_is_latched_across_live_mask_drift(
    tmp_path,
):
    recorder = make_recorder(
        tmp_path,
        run_id='diag_mask_latch',
    )

    physical_env = FakeLiveEnv()

    env = RecordedTrainingEnv(
        physical_env,
        recorder=recorder,
    )

    env.reset()

    assert env.action_masks().tolist() == [
        True,
        False,
        True,
        False,
    ]

    # Simulate asynchronous physical/live state drift
    # after the observation was handed to the policy.
    physical_env._mask = np.array(
        [
            0,
            1,
            0,
            0,
        ],
        dtype=np.int8,
    )

    # PPO must still receive and execute against the
    # mask paired with its actual observation.
    assert env.action_masks().tolist() == [
        True,
        False,
        True,
        False,
    ]

    env.step(0)

    rows = read_csv(
        recorder.steps_path
    )

    assert len(rows) == 1

    assert rows[0][
        'action_mask_bits'
    ] == '1010'

    assert rows[0][
        'action'
    ] == '0'

    env.close()
