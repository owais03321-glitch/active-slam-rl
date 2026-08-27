import numpy as np
import pytest

from active_slam_rl.rl_env import ActiveSlamEnv
from active_slam_rl.rl_training_env import FreshSessionEnv


def make_observation(
    *,
    active_action,
):
    observation = {
        'candidates': np.zeros(
            (
                32,
                4,
            ),
            dtype=np.float32,
        ),
        'action_mask': np.zeros(
            32,
            dtype=np.int8,
        ),
    }

    observation[
        'action_mask'
    ][active_action] = 1

    observation[
        'candidates'
    ][active_action, 2] = float(
        active_action + 1
    )

    return observation


class FakeLiveEnv:
    def __init__(
        self,
        observation,
    ):
        template = (
            ActiveSlamEnv()
        )

        self.action_space = (
            template.action_space
        )

        self.observation_space = (
            template.observation_space
        )

        self.observation = {
            key: value.copy()
            for key, value
            in observation.items()
        }

        self.step_calls = []

    def step(
        self,
        action,
    ):
        self.step_calls.append(
            action
        )

        return (
            {
                key: value.copy()
                for key, value
                in self.observation.items()
            },
            1.25,
            False,
            True,
            {
                'delegated_action': action,
            },
        )

    def action_masks(self):
        return (
            self.observation[
                'action_mask'
            ]
            .astype(
                bool,
                copy=True,
            )
        )


class FakeSession:
    def __init__(
        self,
        *,
        name,
        observation,
        events,
        fail_start=False,
    ):
        self.name = name
        self.events = events
        self.fail_start = fail_start

        self.env = FakeLiveEnv(
            observation
        )

        self.initial_observation = {
            key: value.copy()
            for key, value
            in observation.items()
        }

        self.started = False
        self.closed = False

    def start(self):
        self.events.append(
            f'{self.name}:start'
        )

        if self.fail_start:
            raise RuntimeError(
                f'{self.name} start failed'
            )

        self.started = True

        return self.env

    def close(self):
        if self.closed:
            return

        self.events.append(
            f'{self.name}:close'
        )

        self.closed = True


class SessionFactory:
    def __init__(
        self,
        sessions,
        events,
    ):
        self.sessions = list(
            sessions
        )

        self.events = events

    def __call__(self):
        if not self.sessions:
            raise RuntimeError(
                'No fake session available.'
            )

        session = (
            self.sessions.pop(0)
        )

        self.events.append(
            f'{session.name}:construct'
        )

        return session


def test_spaces_exist_before_first_physical_reset():
    env = FreshSessionEnv(
        session_settle_s=0.0,
        session_factory=lambda: None
    )

    assert env.action_space.n == 32

    assert (
        env.observation_space[
            'candidates'
        ].shape
        == (
            32,
            4,
        )
    )

    assert (
        env.observation_space[
            'action_mask'
        ].shape
        == (
            32,
        )
    )

    assert env.session is None
    assert env.live_env is None


def test_first_reset_starts_one_fresh_session():
    events = []

    session = FakeSession(
        name='a',
        observation=make_observation(
            active_action=3
        ),
        events=events,
    )

    env = FreshSessionEnv(
        session_settle_s=0.0,
        session_factory=SessionFactory(
            [
                session,
            ],
            events,
        )
    )

    observation, info = (
        env.reset(
            seed=7
        )
    )

    assert events == [
        'a:construct',
        'a:start',
    ]

    assert env.session is session
    assert env.live_env is session.env

    assert (
        observation[
            'action_mask'
        ].tolist()
        == session.initial_observation[
            'action_mask'
        ].tolist()
    )

    assert info == {}

    env.close()


def test_second_reset_closes_old_session_before_constructing_new():
    events = []

    first = FakeSession(
        name='first',
        observation=make_observation(
            active_action=1
        ),
        events=events,
    )

    second = FakeSession(
        name='second',
        observation=make_observation(
            active_action=9
        ),
        events=events,
    )

    env = FreshSessionEnv(
        session_settle_s=0.0,
        session_factory=SessionFactory(
            [
                first,
                second,
            ],
            events,
        )
    )

    env.reset()
    observation, _ = env.reset()

    assert events == [
        'first:construct',
        'first:start',
        'first:close',
        'second:construct',
        'second:start',
    ]

    assert first.closed is True
    assert env.session is second

    assert (
        observation[
            'action_mask'
        ][9]
        == 1
    )

    env.close()


def test_step_and_action_masks_delegate_to_live_environment():
    events = []

    session = FakeSession(
        name='live',
        observation=make_observation(
            active_action=5
        ),
        events=events,
    )

    env = FreshSessionEnv(
        session_settle_s=0.0,
        session_factory=SessionFactory(
            [
                session,
            ],
            events,
        )
    )

    with pytest.raises(
        RuntimeError,
        match='requires reset',
    ):
        env.step(
            5
        )

    with pytest.raises(
        RuntimeError,
        match='requires reset',
    ):
        env.action_masks()

    env.reset()

    (
        observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(
        5
    )

    assert (
        session.env.step_calls
        == [
            5,
        ]
    )

    assert reward == pytest.approx(
        1.25
    )

    assert terminated is False
    assert truncated is True

    assert info == {
        'delegated_action': 5,
    }

    assert (
        observation[
            'action_mask'
        ][5]
        == 1
    )

    assert env.action_masks().tolist() == [
        index == 5
        for index in range(
            32
        )
    ]

    env.close()


def test_close_is_idempotent_and_disables_live_interaction():
    events = []

    session = FakeSession(
        name='a',
        observation=make_observation(
            active_action=2
        ),
        events=events,
    )

    env = FreshSessionEnv(
        session_settle_s=0.0,
        session_factory=SessionFactory(
            [
                session,
            ],
            events,
        )
    )

    env.reset()

    env.close()
    env.close()

    assert events == [
        'a:construct',
        'a:start',
        'a:close',
    ]

    assert env.session is None
    assert env.live_env is None

    with pytest.raises(
        RuntimeError,
        match='requires reset',
    ):
        env.step(
            2
        )


def test_failed_replacement_reset_fails_closed_and_cleans_new_session():
    events = []

    first = FakeSession(
        name='first',
        observation=make_observation(
            active_action=1
        ),
        events=events,
    )

    failing = FakeSession(
        name='failing',
        observation=make_observation(
            active_action=4
        ),
        events=events,
        fail_start=True,
    )

    env = FreshSessionEnv(
        session_settle_s=0.0,
        session_factory=SessionFactory(
            [
                first,
                failing,
            ],
            events,
        )
    )

    env.reset()

    with pytest.raises(
        RuntimeError,
        match='failing start failed',
    ):
        env.reset()

    assert events == [
        'first:construct',
        'first:start',
        'first:close',
        'failing:construct',
        'failing:start',
        'failing:close',
    ]

    assert first.closed is True
    assert failing.closed is True

    assert env.session is None
    assert env.live_env is None

    with pytest.raises(
        RuntimeError,
        match='requires reset',
    ):
        env.action_masks()


class TimeoutStartSession(FakeSession):
    def start(self):
        self.events.append(
            f'{self.name}:start'
        )

        raise TimeoutError(
            f'{self.name} readiness timeout'
        )


def test_reset_retries_timeout_with_completely_fresh_session():
    events = []

    timed_out = TimeoutStartSession(
        name='timed-out',
        observation=make_observation(
            active_action=1
        ),
        events=events,
    )

    recovered = FakeSession(
        name='recovered',
        observation=make_observation(
            active_action=7
        ),
        events=events,
    )

    env = FreshSessionEnv(
        session_settle_s=0.0,
        session_factory=SessionFactory(
            [
                timed_out,
                recovered,
            ],
            events,
        ),
        max_start_attempts=3,
    )

    observation, info = env.reset()

    assert events == [
        'timed-out:construct',
        'timed-out:start',
        'timed-out:close',
        'recovered:construct',
        'recovered:start',
    ]

    assert timed_out.closed is True
    assert env.session is recovered
    assert env.live_env is recovered.env

    assert (
        observation[
            'action_mask'
        ][7]
        == 1
    )

    assert info == {}

    env.close()


def test_reset_stops_after_bounded_timeout_attempts():
    events = []

    first = TimeoutStartSession(
        name='first-timeout',
        observation=make_observation(
            active_action=1
        ),
        events=events,
    )

    second = TimeoutStartSession(
        name='second-timeout',
        observation=make_observation(
            active_action=2
        ),
        events=events,
    )

    env = FreshSessionEnv(
        session_settle_s=0.0,
        session_factory=SessionFactory(
            [
                first,
                second,
            ],
            events,
        ),
        max_start_attempts=2,
    )

    with pytest.raises(
        TimeoutError,
        match='second-timeout',
    ):
        env.reset()

    assert first.closed is True
    assert second.closed is True
    assert env.session is None
    assert env.live_env is None

    assert events == [
        'first-timeout:construct',
        'first-timeout:start',
        'first-timeout:close',
        'second-timeout:construct',
        'second-timeout:start',
        'second-timeout:close',
    ]


def test_max_start_attempts_must_be_positive_integer():
    with pytest.raises(
        ValueError,
        match='positive integer',
    ):
        FreshSessionEnv(
        session_settle_s=0.0,
            max_start_attempts=0
        )

    with pytest.raises(
        ValueError,
        match='positive integer',
    ):
        FreshSessionEnv(
        session_settle_s=0.0,
            max_start_attempts=True
        )


def test_physical_episode_reset_settles_before_relaunch():
    events = []
    sleeps = []

    first = FakeSession(
        name='first',
        observation=make_observation(
            active_action=1
        ),
        events=events,
    )

    second = FakeSession(
        name='second',
        observation=make_observation(
            active_action=2
        ),
        events=events,
    )

    env = FreshSessionEnv(
        session_factory=SessionFactory(
            [
                first,
                second,
            ],
            events,
        ),
        max_start_attempts=1,
        session_settle_s=3.0,
        sleep=sleeps.append,
    )

    env.reset()

    assert sleeps == []

    env.reset()

    assert sleeps == [
        3.0,
    ]

    assert events == [
        'first:construct',
        'first:start',
        'first:close',
        'second:construct',
        'second:start',
    ]

    env.close()


def test_timeout_retry_settles_before_fresh_attempt():
    events = []
    sleeps = []

    timed_out = TimeoutStartSession(
        name='timeout',
        observation=make_observation(
            active_action=1
        ),
        events=events,
    )

    recovered = FakeSession(
        name='recovered',
        observation=make_observation(
            active_action=4
        ),
        events=events,
    )

    env = FreshSessionEnv(
        session_factory=SessionFactory(
            [
                timed_out,
                recovered,
            ],
            events,
        ),
        max_start_attempts=2,
        session_settle_s=3.0,
        sleep=sleeps.append,
    )

    observation, _ = env.reset()

    assert sleeps == [
        3.0,
    ]

    assert (
        observation[
            'action_mask'
        ][4]
        == 1
    )

    assert events == [
        'timeout:construct',
        'timeout:start',
        'timeout:close',
        'recovered:construct',
        'recovered:start',
    ]

    env.close()


def test_session_settle_configuration_validation():
    with pytest.raises(
        ValueError,
        match='session_settle_s',
    ):
        FreshSessionEnv(
            session_settle_s=-0.1,
        )

    with pytest.raises(
        TypeError,
        match='sleep must be callable',
    ):
        FreshSessionEnv(
            session_settle_s=0.0,
            sleep=None,
        )
