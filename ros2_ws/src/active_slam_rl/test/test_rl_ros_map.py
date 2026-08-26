import pytest
from nav_msgs.msg import OccupancyGrid

from active_slam_rl.rl_env import ActiveSlamEnv
from active_slam_rl.rl_ros_map import (
    eligible_frontier_candidates_from_occupancy_grid,
    frontier_candidates_from_occupancy_grid,
)


def make_map(
    *,
    width,
    height,
    resolution=1.0,
    origin_x=0.0,
    origin_y=0.0,
    default_value=100,
):
    msg = OccupancyGrid()

    msg.info.width = width
    msg.info.height = height
    msg.info.resolution = resolution
    msg.info.origin.position.x = origin_x
    msg.info.origin.position.y = origin_y

    msg.data = [
        default_value
        for _ in range(width * height)
    ]

    return msg


def test_ros_map_converts_to_expected_frontier():
    msg = make_map(
        width=9,
        height=9,
        resolution=0.5,
        origin_x=-2.0,
        origin_y=-1.0,
    )

    for y in range(2, 7):
        msg.data[y * msg.info.width + 4] = 0
        msg.data[y * msg.info.width + 5] = -1

    candidates = (
        frontier_candidates_from_occupancy_grid(
            msg
        )
    )

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.cell_x == 4
    assert candidate.cell_y == 4
    assert candidate.cluster_size == 5

    assert candidate.world_x == pytest.approx(
        0.25
    )

    assert candidate.world_y == pytest.approx(
        1.25
    )


def test_ros_map_respects_minimum_cluster_size():
    msg = make_map(
        width=8,
        height=8,
        resolution=0.05,
    )

    for y in range(2, 6):
        msg.data[y * msg.info.width + 3] = 0
        msg.data[y * msg.info.width + 4] = -1

    default_candidates = (
        frontier_candidates_from_occupancy_grid(
            msg
        )
    )

    relaxed_candidates = (
        frontier_candidates_from_occupancy_grid(
            msg,
            min_cluster_size=4,
        )
    )

    assert default_candidates == []
    assert len(relaxed_candidates) == 1
    assert relaxed_candidates[0].cluster_size == 4


def test_ros_map_propagates_invalid_data_size():
    msg = make_map(
        width=3,
        height=3,
    )

    msg.data = [0] * 8

    with pytest.raises(
        ValueError,
        match='Occupancy data has 8 cells; expected 9',
    ):
        frontier_candidates_from_occupancy_grid(
            msg
        )


def make_two_frontier_map():
    msg = make_map(
        width=15,
        height=10,
        resolution=1.0,
    )

    for y in range(2, 7):
        msg.data[y * msg.info.width + 3] = 0
        msg.data[y * msg.info.width + 4] = -1

        msg.data[y * msg.info.width + 10] = 0
        msg.data[y * msg.info.width + 11] = -1

    return msg


def test_ros_map_eligibility_rejects_near_robot_candidate():
    msg = make_two_frontier_map()

    candidates = (
        eligible_frontier_candidates_from_occupancy_grid(
            msg,
            robot_x=3.5,
            robot_y=4.5,
        )
    )

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.cell_x == 10
    assert candidate.cell_y == 4
    assert candidate.world_x == pytest.approx(10.5)
    assert candidate.world_y == pytest.approx(4.5)


def test_ros_map_eligibility_rejects_visited_candidate():
    msg = make_two_frontier_map()

    candidates = (
        eligible_frontier_candidates_from_occupancy_grid(
            msg,
            robot_x=0.0,
            robot_y=0.0,
            visited_goals=[
                (3.5, 4.5),
            ],
        )
    )

    assert len(candidates) == 1
    assert candidates[0].cell_x == 10
    assert candidates[0].cell_y == 4


def test_ros_map_eligible_candidates_align_with_env_action_slots():
    msg = make_two_frontier_map()

    candidates = (
        eligible_frontier_candidates_from_occupancy_grid(
            msg,
            robot_x=0.0,
            robot_y=0.0,
            visited_goals=[
                (3.5, 4.5),
            ],
        )
    )

    env = ActiveSlamEnv(
        max_candidates=4,
    )

    observation = env.set_frontier_state(
        candidates=candidates,
        robot_x=0.0,
        robot_y=0.0,
    )

    assert len(candidates) == 1
    assert candidates[0].world_x == pytest.approx(10.5)
    assert candidates[0].world_y == pytest.approx(4.5)

    assert observation['action_mask'].tolist() == [
        1,
        0,
        0,
        0,
    ]

    assert observation['candidates'][0, 0] == pytest.approx(
        10.5
    )
    assert observation['candidates'][0, 1] == pytest.approx(
        4.5
    )

    env.close()
