import rclpy
from nav_msgs.msg import OccupancyGrid
import pytest

from active_slam_rl.rl_execution import (
    RlActionOutcome,
)
from active_slam_rl.rl_observation_node import (
    RlObservationNode,
)


@pytest.fixture
def node():
    rclpy.init()

    test_node = RlObservationNode(
        max_candidates=4,
    )

    try:
        yield test_node

    finally:
        test_node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


def make_two_frontier_map():
    msg = OccupancyGrid()

    msg.info.width = 15
    msg.info.height = 10
    msg.info.resolution = 1.0

    msg.data = [
        100
        for _ in range(
            msg.info.width
            * msg.info.height
        )
    ]

    for y in range(2, 7):
        msg.data[
            y * msg.info.width + 3
        ] = 0

        msg.data[
            y * msg.info.width + 4
        ] = -1

        msg.data[
            y * msg.info.width + 10
        ] = 0

        msg.data[
            y * msg.info.width + 11
        ] = -1

    return msg


def test_node_starts_with_empty_rl_observation(node):
    assert node.env.action_space.n == 4

    assert (
        node.latest_observation[
            'action_mask'
        ].tolist()
        == [0, 0, 0, 0]
    )

    assert node.latest_candidates == []


def test_node_updates_observation_from_map(node):
    msg = make_two_frontier_map()

    observation = node.update_from_map(
        msg,
        robot_x=0.0,
        robot_y=0.0,
    )

    assert len(node.latest_candidates) == 2

    assert observation[
        'action_mask'
    ].tolist() == [
        1,
        1,
        0,
        0,
    ]

    node.sync_env_to_latest_frontier()

    first = node.env.candidate_for_action(
        0
    )

    second = node.env.candidate_for_action(
        1
    )

    assert first.world_x == pytest.approx(
        3.5
    )

    assert second.world_x == pytest.approx(
        10.5
    )


def test_map_callback_uses_robot_pose_and_visited_filter(
    node,
    monkeypatch,
):
    msg = make_two_frontier_map()

    node.visited_goals.append(
        (3.5, 4.5)
    )

    monkeypatch.setattr(
        node,
        '_lookup_robot_position',
        lambda: (0.0, 0.0),
    )

    node.map_callback(msg)

    assert len(node.latest_candidates) == 1

    assert node.latest_observation[
        'action_mask'
    ].tolist() == [
        1,
        0,
        0,
        0,
    ]

    node.sync_env_to_latest_frontier()

    candidate = (
        node.env.candidate_for_action(0)
    )

    assert candidate.world_x == pytest.approx(
        10.5
    )

    assert candidate.world_y == pytest.approx(
        4.5
    )


def test_node_starts_with_empty_reward_measurements(node):
    assert node.latest_explored_area_m2 is None

    assert (
        node.path_tracker.path_length_m
        == pytest.approx(0.0)
    )


def test_map_update_records_explored_area(node):
    msg = make_two_frontier_map()

    node.update_from_map(
        msg,
        robot_x=0.0,
        robot_y=0.0,
    )

    assert (
        node.latest_explored_area_m2
        == pytest.approx(140.0)
    )


def test_odom_callback_updates_path_tracker(node):
    from nav_msgs.msg import Odometry

    first = Odometry()

    first.pose.pose.position.x = 0.0
    first.pose.pose.position.y = 0.0

    node.odom_callback(first)

    second = Odometry()

    second.pose.pose.position.x = 0.3
    second.pose.pose.position.y = 0.4

    node.odom_callback(second)

    assert (
        node.path_tracker.path_length_m
        == pytest.approx(0.5)
    )


def test_transition_measurements_require_map(node):
    from nav_msgs.msg import Odometry

    odom = Odometry()
    odom.pose.pose.position.x = 0.0
    odom.pose.pose.position.y = 0.0

    node.odom_callback(odom)

    with pytest.raises(
        RuntimeError,
        match='Explored-area measurement is not available yet',
    ):
        node.current_transition_measurements()


def test_transition_measurements_require_odom(node):
    msg = make_two_frontier_map()

    node.update_from_map(
        msg,
        robot_x=0.0,
        robot_y=0.0,
    )

    with pytest.raises(
        RuntimeError,
        match='Odometry measurement is not available yet',
    ):
        node.current_transition_measurements()


def test_transition_measurements_return_live_cumulative_state(
    node,
):
    from nav_msgs.msg import Odometry

    msg = make_two_frontier_map()

    node.update_from_map(
        msg,
        robot_x=0.0,
        robot_y=0.0,
    )

    first = Odometry()
    first.pose.pose.position.x = 0.0
    first.pose.pose.position.y = 0.0

    node.odom_callback(first)

    second = Odometry()
    second.pose.pose.position.x = 0.3
    second.pose.pose.position.y = 0.4

    node.odom_callback(second)

    area_m2, path_m = (
        node.current_transition_measurements()
    )

    assert area_m2 == pytest.approx(140.0)
    assert path_m == pytest.approx(0.5)


def test_node_wires_rl_nav2_execution_stack(node):
    assert (
        node.nav_executor.action_client
        is node.nav_client
    )

    assert (
        node.action_coordinator.env
        is node.env
    )

    assert (
        node.action_coordinator.transition_tracker
        is node.transition_tracker
    )

    assert (
        node.action_coordinator.nav_executor
        is node.nav_executor
    )

    assert (
        node.action_coordinator.visited_goals
        is node.visited_goals
    )

    assert (
        node.env.step_bridge
        is node.gym_step_bridge
    )

    assert callable(
        node.gym_step_bridge.start_action
    )

    assert callable(
        node.gym_step_bridge.complete_action
    )

    assert callable(
        node.gym_step_bridge.observation_sync
    )


def test_start_rl_action_uses_live_measurements(
    node,
    monkeypatch,
):
    from nav_msgs.msg import Odometry

    msg = make_two_frontier_map()

    node.update_from_map(
        msg,
        robot_x=0.0,
        robot_y=0.0,
    )

    first = Odometry()
    first.pose.pose.position.x = 0.0
    first.pose.pose.position.y = 0.0
    node.odom_callback(first)

    second = Odometry()
    second.pose.pose.position.x = 0.3
    second.pose.pose.position.y = 0.4
    node.odom_callback(second)

    captured = {}

    def fake_start_action(**kwargs):
        captured.update(kwargs)
        return 'started'

    monkeypatch.setattr(
        node.action_coordinator,
        'start_action',
        fake_start_action,
    )

    node.sync_env_to_latest_frontier()

    result = node.start_rl_action(1)

    assert result == 'started'
    assert captured['action'] == 1
    assert captured['area_m2'] == pytest.approx(
        140.0
    )
    assert captured['path_m'] == pytest.approx(
        0.5
    )

    assert captured['stamp'] is not None

    assert (
        node.episode_time_limit.active
        is True
    )


def test_complete_rl_action_passes_live_measurement_provider(
    node,
    monkeypatch,
):
    captured = {}

    def fake_complete_action(
        *,
        measurement_provider,
        timeout,
        cancel_on_timeout,
        cutoff_provider,
    ):
        captured['provider'] = (
            measurement_provider
        )
        captured['timeout'] = timeout
        captured['cancel_on_timeout'] = (
            cancel_on_timeout
        )
        captured['cutoff_provider'] = (
            cutoff_provider
        )
        return 'completed'

    monkeypatch.setattr(
        node.action_coordinator,
        'complete_action',
        fake_complete_action,
    )

    result = node.complete_rl_action(
        timeout=2.5
    )

    assert result == 'completed'
    assert callable(captured['provider'])
    assert captured['timeout'] == pytest.approx(
        2.5
    )

    assert (
        captured['cancel_on_timeout']
        is False
    )

    assert callable(
        captured['cutoff_provider']
    )


def test_sync_env_requires_live_frontier_state(node):
    with pytest.raises(
        RuntimeError,
        match='Frontier state is not available yet',
    ):
        node.sync_env_to_latest_frontier()


def test_live_map_update_does_not_replace_frozen_gym_snapshot(
    node,
):
    msg = make_two_frontier_map()

    node.update_from_map(
        msg,
        robot_x=0.0,
        robot_y=0.0,
    )

    node.sync_env_to_latest_frontier()

    frozen_candidate = (
        node.env.candidate_for_action(0)
    )

    assert frozen_candidate.world_x == pytest.approx(
        3.5
    )

    with node._state_lock:
        node.visited_goals.append(
            (3.5, 4.5)
        )

    node.update_from_map(
        msg,
        robot_x=0.0,
        robot_y=0.0,
    )

    still_frozen = (
        node.env.candidate_for_action(0)
    )

    assert still_frozen.world_x == pytest.approx(
        3.5
    )

    assert node.latest_candidates[
        0
    ].world_x == pytest.approx(
        10.5
    )

    node.sync_env_to_latest_frontier()

    refreshed_candidate = (
        node.env.candidate_for_action(0)
    )

    assert refreshed_candidate.world_x == pytest.approx(
        10.5
    )


def test_sync_applies_new_visited_goal_without_new_map_update(
    node,
):
    msg = make_two_frontier_map()

    node.update_from_map(
        msg,
        robot_x=0.0,
        robot_y=0.0,
    )

    assert len(
        node.latest_candidates
    ) == 2

    node.sync_env_to_latest_frontier()

    assert (
        node.env.candidate_for_action(
            0
        ).world_x
        == pytest.approx(3.5)
    )

    with node._state_lock:
        node.visited_goals.append(
            (3.5, 4.5)
        )

    # Deliberately do NOT call update_from_map().
    # latest_candidates is therefore stale with respect
    # to the newly completed visited goal.
    assert len(
        node.latest_candidates
    ) == 2

    observation = (
        node.sync_env_to_latest_frontier()
    )

    assert observation[
        'action_mask'
    ].tolist() == [
        1,
        0,
        0,
        0,
    ]

    remaining = (
        node.env.candidate_for_action(0)
    )

    assert remaining.world_x == pytest.approx(
        10.5
    )

    assert remaining.world_y == pytest.approx(
        4.5
    )


def test_live_node_requires_external_episode_reset(
    node,
):
    assert (
        node.env.external_episode_reset_required
        is True
    )

    with pytest.raises(
        RuntimeError,
        match=(
            'fresh simulator and SLAM state'
        ),
    ):
        node.env.reset()


def test_node_episode_clock_starts_inactive(node):
    assert (
        node.episode_time_limit.horizon_s
        == pytest.approx(300.0)
    )

    assert (
        node.episode_time_limit.active
        is False
    )

    assert (
        node.episode_time_limit.remaining_s
        == pytest.approx(300.0)
    )


def test_failed_action_does_not_start_episode_clock(
    node,
    monkeypatch,
):
    from nav_msgs.msg import Odometry

    msg = make_two_frontier_map()

    node.update_from_map(
        msg,
        robot_x=0.0,
        robot_y=0.0,
    )

    odom = Odometry()
    odom.pose.pose.position.x = 0.0
    odom.pose.pose.position.y = 0.0

    node.odom_callback(odom)

    node.sync_env_to_latest_frontier()

    def fail_start_action(**kwargs):
        raise RuntimeError(
            'synthetic Nav2 start failure'
        )

    monkeypatch.setattr(
        node.action_coordinator,
        'start_action',
        fail_start_action,
    )

    with pytest.raises(
        RuntimeError,
        match='synthetic Nav2 start failure',
    ):
        node.start_rl_action(0)

    assert (
        node.episode_time_limit.active
        is False
    )


def test_default_completion_uses_ros_watchdog_cutoff_provider(
    node,
    monkeypatch,
):
    captured = {}
    events = []

    class FakeEpisodeLimit:

        active = True

    node.episode_time_limit = (
        FakeEpisodeLimit()
    )

    outcome = RlActionOutcome(
        navigation=object(),
        transition=object(),
        truncated=False,
    )

    def fake_complete_action(
        *,
        measurement_provider,
        timeout,
        cancel_on_timeout,
        cutoff_provider,
    ):
        events.append(
            'complete'
        )

        captured['provider'] = (
            measurement_provider
        )
        captured['timeout'] = timeout
        captured['cancel_on_timeout'] = (
            cancel_on_timeout
        )
        captured['cutoff_provider'] = (
            cutoff_provider
        )

        return outcome

    def fake_wait_for_fresh_map_or_horizon(
        *,
        after_revision,
    ):
        events.append(
            (
                'wait_for_map',
                after_revision,
            )
        )

        return True

    monkeypatch.setattr(
        node.action_coordinator,
        'complete_action',
        fake_complete_action,
    )

    monkeypatch.setattr(
        node,
        'wait_for_fresh_map_or_horizon',
        fake_wait_for_fresh_map_or_horizon,
    )

    result = node.complete_rl_action()

    assert result is outcome

    assert callable(
        captured['provider']
    )

    assert captured['timeout'] is None

    assert (
        captured['cancel_on_timeout']
        is False
    )

    assert callable(
        captured['cutoff_provider']
    )

    assert events == [
        'complete',
        (
            'wait_for_map',
            node.map_revision,
        ),
    ]


def test_map_revision_starts_at_zero(node):
    assert node.map_revision == 0


def test_processed_map_advances_revision_once(node):
    msg = make_two_frontier_map()

    assert node.map_revision == 0

    node.update_from_map(
        msg,
        robot_x=0.0,
        robot_y=0.0,
    )

    assert node.map_revision == 1

    node.update_from_map(
        msg,
        robot_x=0.0,
        robot_y=0.0,
    )

    assert node.map_revision == 2


def test_wait_for_map_revision_returns_if_map_is_already_newer(
    node,
):
    msg = make_two_frontier_map()

    node.update_from_map(
        msg,
        robot_x=0.0,
        robot_y=0.0,
    )

    assert node.map_revision == 1

    result = node.wait_for_map_revision(
        after_revision=0,
        timeout=0.0,
    )

    assert result == 1


def test_wait_for_map_revision_unblocks_on_new_map(
    node,
):
    from threading import Thread

    msg = make_two_frontier_map()

    starting_revision = (
        node.map_revision
    )

    results = []

    thread = Thread(
        target=lambda: results.append(
            node.wait_for_map_revision(
                after_revision=(
                    starting_revision
                ),
                timeout=1.0,
            )
        )
    )

    thread.start()

    node.update_from_map(
        msg,
        robot_x=0.0,
        robot_y=0.0,
    )

    thread.join(
        timeout=1.0
    )

    assert thread.is_alive() is False

    assert results == [
        starting_revision + 1
    ]

    assert (
        node.wait_for_map_revision(
            after_revision=node.map_revision,
            timeout=0.0,
        )
        is None
    )


def test_first_action_arms_episode_watchdog_only_once(
    node,
    monkeypatch,
):
    from nav_msgs.msg import Odometry

    msg = make_two_frontier_map()

    node.update_from_map(
        msg,
        robot_x=0.0,
        robot_y=0.0,
    )

    odom = Odometry()
    odom.pose.pose.position.x = 0.0
    odom.pose.pose.position.y = 0.0
    node.odom_callback(odom)

    node.sync_env_to_latest_frontier()

    class FakeWatchdog:

        def __init__(self):
            self.reset_calls = 0
            self.cancel_calls = 0

        def reset(self):
            self.reset_calls += 1

        def cancel(self):
            self.cancel_calls += 1

    watchdog = FakeWatchdog()

    node.episode_watchdog = watchdog

    monkeypatch.setattr(
        node.action_coordinator,
        'start_action',
        lambda **kwargs: 'started',
    )

    assert node.start_rl_action(0) == 'started'
    assert node.start_rl_action(0) == 'started'

    assert watchdog.reset_calls == 1

    assert (
        node.episode_time_limit.active
        is True
    )


def test_horizon_callback_freezes_measurements_and_cancels_once(
    node,
    monkeypatch,
):
    from nav_msgs.msg import Odometry

    msg = make_two_frontier_map()

    node.update_from_map(
        msg,
        robot_x=0.0,
        robot_y=0.0,
    )

    first = Odometry()
    first.pose.pose.position.x = 0.0
    first.pose.pose.position.y = 0.0
    node.odom_callback(first)

    second = Odometry()
    second.pose.pose.position.x = 0.3
    second.pose.pose.position.y = 0.4
    node.odom_callback(second)

    class FakeEpisodeLimit:

        active = True
        truncated = True

    class FakeWatchdog:

        def __init__(self):
            self.cancel_calls = 0

        def cancel(self):
            self.cancel_calls += 1

    watchdog = FakeWatchdog()
    cancel_calls = []

    node.episode_time_limit = FakeEpisodeLimit()
    node.episode_watchdog = watchdog

    monkeypatch.setattr(
        node.nav_executor,
        'request_cancel',
        lambda: cancel_calls.append(True) or True,
    )

    node._episode_horizon_callback()
    node._episode_horizon_callback()

    assert (
        node.episode_cutoff_measurements()
        == pytest.approx(
            (
                140.0,
                0.5,
            )
        )
    )

    assert watchdog.cancel_calls == 1
    assert cancel_calls == [True]


def test_post_horizon_odom_is_excluded_from_cutoff(
    node,
    monkeypatch,
):
    from nav_msgs.msg import Odometry

    msg = make_two_frontier_map()

    node.update_from_map(
        msg,
        robot_x=0.0,
        robot_y=0.0,
    )

    first = Odometry()
    first.pose.pose.position.x = 0.0
    first.pose.pose.position.y = 0.0
    node.odom_callback(first)

    second = Odometry()
    second.pose.pose.position.x = 0.3
    second.pose.pose.position.y = 0.4
    node.odom_callback(second)

    class FakeEpisodeLimit:

        active = True
        truncated = True

    class FakeWatchdog:

        def cancel(self):
            pass

    node.episode_time_limit = FakeEpisodeLimit()
    node.episode_watchdog = FakeWatchdog()

    monkeypatch.setattr(
        node.nav_executor,
        'request_cancel',
        lambda: True,
    )

    third = Odometry()
    third.pose.pose.position.x = 0.6
    third.pose.pose.position.y = 0.8

    node.odom_callback(third)

    cutoff_area_m2, cutoff_path_m = (
        node.episode_cutoff_measurements()
    )

    assert cutoff_area_m2 == pytest.approx(
        140.0
    )

    assert cutoff_path_m == pytest.approx(
        0.5
    )

    assert (
        node.path_tracker.path_length_m
        == pytest.approx(1.0)
    )


def test_post_horizon_map_is_excluded_from_cutoff(
    node,
    monkeypatch,
):
    from nav_msgs.msg import Odometry

    initial = make_two_frontier_map()

    node.update_from_map(
        initial,
        robot_x=0.0,
        robot_y=0.0,
    )

    odom = Odometry()
    odom.pose.pose.position.x = 0.0
    odom.pose.pose.position.y = 0.0
    node.odom_callback(odom)

    changed = make_two_frontier_map()

    unknown_index = next(
        index
        for index, value
        in enumerate(changed.data)
        if value == -1
    )

    changed.data[
        unknown_index
    ] = 0

    class FakeEpisodeLimit:

        active = True
        truncated = True

    class FakeWatchdog:

        def cancel(self):
            pass

    node.episode_time_limit = FakeEpisodeLimit()
    node.episode_watchdog = FakeWatchdog()

    monkeypatch.setattr(
        node.nav_executor,
        'request_cancel',
        lambda: True,
    )

    monkeypatch.setattr(
        node,
        '_lookup_robot_position',
        lambda: (
            0.0,
            0.0,
        ),
    )

    node.map_callback(
        changed
    )

    cutoff_area_m2, cutoff_path_m = (
        node.episode_cutoff_measurements()
    )

    assert cutoff_area_m2 == pytest.approx(
        140.0
    )

    assert cutoff_path_m == pytest.approx(
        0.0
    )

    assert (
        node.latest_explored_area_m2
        == pytest.approx(141.0)
    )


def test_post_action_map_wait_unblocks_at_episode_horizon(
    node,
    monkeypatch,
):
    from nav_msgs.msg import Odometry
    from threading import Thread

    msg = make_two_frontier_map()

    node.update_from_map(
        msg,
        robot_x=0.0,
        robot_y=0.0,
    )

    odom = Odometry()
    odom.pose.pose.position.x = 0.0
    odom.pose.pose.position.y = 0.0
    node.odom_callback(odom)

    after_revision = (
        node.map_revision
    )

    results = []

    thread = Thread(
        target=lambda: results.append(
            node.wait_for_fresh_map_or_horizon(
                after_revision=(
                    after_revision
                )
            )
        )
    )

    thread.start()

    class FakeEpisodeLimit:

        active = True
        truncated = True

    class FakeWatchdog:

        def cancel(self):
            pass

    node.episode_time_limit = (
        FakeEpisodeLimit()
    )

    node.episode_watchdog = (
        FakeWatchdog()
    )

    monkeypatch.setattr(
        node.nav_executor,
        'request_cancel',
        lambda: True,
    )

    node._episode_horizon_callback()

    thread.join(
        timeout=1.0
    )

    assert thread.is_alive() is False
    assert results == [False]


def test_horizon_during_post_action_wait_truncates_outcome(
    node,
    monkeypatch,
):
    class FakeEpisodeLimit:

        active = True

    node.episode_time_limit = (
        FakeEpisodeLimit()
    )

    original = RlActionOutcome(
        navigation=object(),
        transition=object(),
        truncated=False,
    )

    monkeypatch.setattr(
        node.action_coordinator,
        'complete_action',
        lambda **kwargs: original,
    )

    monkeypatch.setattr(
        node,
        'wait_for_fresh_map_or_horizon',
        lambda **kwargs: False,
    )

    result = node.complete_rl_action()

    assert result is not original

    assert result.navigation is (
        original.navigation
    )

    assert result.transition is (
        original.transition
    )

    assert result.truncated is True
    assert original.truncated is False


def test_destroy_node_explicitly_destroys_nav2_action_client(
    monkeypatch,
):
    rclpy.init()

    test_node = RlObservationNode(
        max_candidates=4,
    )

    destroy_calls = []

    original_destroy = (
        test_node.nav_client.destroy
    )

    def tracked_destroy():
        destroy_calls.append(
            True
        )

        return original_destroy()

    monkeypatch.setattr(
        test_node.nav_client,
        'destroy',
        tracked_destroy,
    )

    try:
        test_node.destroy_node()

        assert destroy_calls == [
            True,
        ]

        assert (
            test_node._nav_client_destroyed
            is True
        )

    finally:
        if rclpy.ok():
            rclpy.shutdown()


def test_actionable_sync_waits_through_transient_zero_frontier(
    node,
    monkeypatch,
):
    class ActiveEpisodeLimit:
        active = True

    node.episode_time_limit = (
        ActiveEpisodeLimit()
    )

    zero = (
        node.env
        ._make_empty_observation()
    )

    actionable = (
        node.env
        ._make_empty_observation()
    )

    actionable[
        'action_mask'
    ][2] = 1

    observations = iter(
        [
            zero,
            actionable,
        ]
    )

    monkeypatch.setattr(
        node,
        'sync_env_to_latest_frontier',
        lambda: next(
            observations
        ),
    )

    waits = []

    def fake_wait(
        *,
        after_revision,
    ):
        waits.append(
            after_revision
        )

        return True

    monkeypatch.setattr(
        node,
        'wait_for_fresh_map_or_horizon',
        fake_wait,
    )

    result = (
        node.sync_env_to_actionable_frontier()
    )

    assert result[
        'action_mask'
    ].tolist() == [
        0,
        0,
        1,
        0,
    ]

    assert len(waits) == 1
