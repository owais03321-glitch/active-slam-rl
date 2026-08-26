import pytest

from active_slam_rl.rl_frontier import extract_frontier_candidates


def make_grid(width, height, value=100):
    return [value] * (width * height)


def test_extracts_expected_frontier_candidate():
    width = 9
    height = 9
    grid = make_grid(width, height)

    for y in range(2, 7):
        grid[y * width + 4] = 0
        grid[y * width + 5] = -1

    candidates = extract_frontier_candidates(
        width=width,
        height=height,
        data=grid,
        resolution=0.5,
        origin_x=-2.0,
        origin_y=-1.0,
    )

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.cell_x == 4
    assert candidate.cell_y == 4
    assert candidate.cluster_size == 5
    assert candidate.world_x == pytest.approx(0.25)
    assert candidate.world_y == pytest.approx(1.25)


def test_rejects_cluster_below_minimum_size():
    width = 8
    height = 8
    grid = make_grid(width, height)

    for y in range(2, 6):
        grid[y * width + 3] = 0
        grid[y * width + 4] = -1

    candidates = extract_frontier_candidates(
        width=width,
        height=height,
        data=grid,
        resolution=0.05,
        origin_x=0.0,
        origin_y=0.0,
    )

    assert candidates == []


def test_candidate_order_is_deterministic():
    width = 12
    height = 10
    grid = make_grid(width, height)

    # Lower frontier cluster.
    for x in range(2, 7):
        grid[2 * width + x] = 0
        grid[3 * width + x] = -1

    # Upper frontier cluster.
    for x in range(5, 10):
        grid[7 * width + x] = 0
        grid[8 * width + x] = -1

    candidates = extract_frontier_candidates(
        width=width,
        height=height,
        data=grid,
        resolution=1.0,
        origin_x=0.0,
        origin_y=0.0,
    )

    assert len(candidates) == 2

    cells = [
        (candidate.cell_x, candidate.cell_y)
        for candidate in candidates
    ]

    assert cells == sorted(
        cells,
        key=lambda point: (
            point[1],
            point[0],
        ),
    )


def test_rejects_wrong_occupancy_data_size():
    with pytest.raises(
        ValueError,
        match='Occupancy data has 8 cells; expected 9',
    ):
        extract_frontier_candidates(
            width=3,
            height=3,
            data=[0] * 8,
            resolution=1.0,
            origin_x=0.0,
            origin_y=0.0,
        )
