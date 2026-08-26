import pytest
from nav_msgs.msg import OccupancyGrid

from active_slam_rl.rl_ros_map import (
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
