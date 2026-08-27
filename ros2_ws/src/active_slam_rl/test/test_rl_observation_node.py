import rclpy
from nav_msgs.msg import OccupancyGrid
import pytest

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


def test_complete_rl_action_passes_live_measurement_provider(
    node,
    monkeypatch,
):
    captured = {}

    def fake_complete_action(
        *,
        measurement_provider,
        timeout,
    ):
        captured['provider'] = (
            measurement_provider
        )
        captured['timeout'] = timeout
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
