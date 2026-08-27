from types import SimpleNamespace

import numpy as np
import pytest

from active_slam_rl.rl_gym_bridge import (
    RlGymStepBridge,
)


def make_outcome(
    *,
    action=1,
    goal_x=2.0,
    goal_y=3.0,
    reward=1.4,
    area_gain_m2=1.5,
    path_delta_m=1.0,
    accepted=True,
    status=4,
    succeeded=True,
    truncated=False,
):
    return SimpleNamespace(
        truncated=truncated,
        navigation=SimpleNamespace(
            accepted=accepted,
            status=status,
            succeeded=succeeded,
        ),
        transition=SimpleNamespace(
            start=SimpleNamespace(
                action=action,
                goal_x=goal_x,
                goal_y=goal_y,
            ),
            metrics=SimpleNamespace(
                reward=reward,
                area_gain_m2=area_gain_m2,
                path_delta_m=path_delta_m,
            ),
        ),
    )


def make_observation(
    *,
    active_slots=1,
):
    mask = np.zeros(
        4,
        dtype=np.int8,
    )

    mask[:active_slots] = 1

    return {
        'candidates': np.zeros(
            (4, 4),
            dtype=np.float32,
        ),
        'action_mask': mask,
    }


def test_step_executes_start_complete_and_sync_in_order():
    calls = []

    outcome = make_outcome()
    observation = make_observation()

    def start_action(action):
        calls.append(
            ('start', action)
        )

    def complete_action(*, timeout):
        calls.append(
            ('complete', timeout)
        )
        return outcome

    def observation_sync():
        calls.append(
            ('sync', None)
        )
        return observation

    bridge = RlGymStepBridge(
        start_action=start_action,
        complete_action=complete_action,
        observation_sync=observation_sync,
    )

    result = bridge.step(1)

    assert calls == [
        ('start', 1),
        ('complete', None),
        ('sync', None),
    ]

    assert result[0] is observation
    assert result[1] == pytest.approx(
        1.4
    )
    assert result[2] is False
    assert result[3] is False

    info = result[4]

    assert info == {
        'action': 1,
        'goal_x': 2.0,
        'goal_y': 3.0,
        'area_gain_m2': 1.5,
        'path_delta_m': 1.0,
        'navigation_accepted': True,
        'navigation_status': 4,
        'navigation_succeeded': True,
    }


def test_failed_navigation_still_returns_physical_transition():
    outcome = make_outcome(
        reward=-0.2,
        area_gain_m2=0.0,
        path_delta_m=2.0,
        accepted=True,
        status=6,
        succeeded=False,
    )

    bridge = RlGymStepBridge(
        start_action=lambda action: None,
        complete_action=(
            lambda *, timeout: outcome
        ),
        observation_sync=make_observation,
    )

    (
        _,
        reward,
        terminated,
        truncated,
        info,
    ) = bridge.step(0)

    assert reward == pytest.approx(
        -0.2
    )
    assert terminated is False
    assert truncated is False

    assert (
        info['navigation_succeeded']
        is False
    )

    assert info[
        'path_delta_m'
    ] == pytest.approx(
        2.0
    )


def test_empty_next_frontier_does_not_terminate_transition():
    outcome = make_outcome()

    observation = make_observation(
        active_slots=0,
    )

    bridge = RlGymStepBridge(
        start_action=lambda action: None,
        complete_action=(
            lambda *, timeout: outcome
        ),
        observation_sync=(
            lambda: observation
        ),
    )

    (
        returned_observation,
        _,
        terminated,
        truncated,
        _,
    ) = bridge.step(1)

    assert returned_observation[
        'action_mask'
    ].tolist() == [
        0,
        0,
        0,
        0,
    ]

    assert terminated is False
    assert truncated is False


def test_blocking_step_requires_terminal_completion():
    sync_calls = []

    bridge = RlGymStepBridge(
        start_action=lambda action: None,
        complete_action=(
            lambda *, timeout: None
        ),
        observation_sync=(
            lambda: sync_calls.append(
                True
            )
        ),
    )

    with pytest.raises(
        RuntimeError,
        match='no terminal outcome',
    ):
        bridge.step(0)

    assert sync_calls == []


def test_horizon_outcome_sets_truncated_without_termination():
    outcome = make_outcome(
        reward=0.75,
        area_gain_m2=1.0,
        path_delta_m=2.5,
        accepted=True,
        status=5,
        succeeded=False,
        truncated=True,
    )

    observation = make_observation(
        active_slots=2,
    )

    bridge = RlGymStepBridge(
        start_action=lambda action: None,
        complete_action=(
            lambda *, timeout: outcome
        ),
        observation_sync=(
            lambda: observation
        ),
    )

    (
        returned_observation,
        reward,
        terminated,
        truncated,
        info,
    ) = bridge.step(0)

    assert returned_observation is observation

    assert reward == pytest.approx(
        0.75
    )

    assert terminated is False
    assert truncated is True

    assert (
        info['navigation_succeeded']
        is False
    )

    assert info[
        'navigation_status'
    ] == 5

    assert info[
        'area_gain_m2'
    ] == pytest.approx(
        1.0
    )

    assert info[
        'path_delta_m'
    ] == pytest.approx(
        2.5
    )
