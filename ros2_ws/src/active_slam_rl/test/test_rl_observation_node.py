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
