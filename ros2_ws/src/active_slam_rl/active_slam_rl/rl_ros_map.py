from active_slam_rl.rl_frontier import (
    extract_frontier_candidates,
    filter_eligible_frontier_candidates,
)


def frontier_candidates_from_occupancy_grid(
    msg,
    *,
    min_cluster_size=5,
):
    """Convert a ROS OccupancyGrid into deterministic frontier candidates."""

    return extract_frontier_candidates(
        width=msg.info.width,
        height=msg.info.height,
        data=msg.data,
        resolution=msg.info.resolution,
        origin_x=msg.info.origin.position.x,
        origin_y=msg.info.origin.position.y,
        min_cluster_size=min_cluster_size,
    )


def eligible_frontier_candidates_from_occupancy_grid(
    msg,
    *,
    robot_x,
    robot_y,
    visited_goals=(),
    min_cluster_size=5,
    min_robot_distance=0.35,
    visited_goal_radius=0.50,
):
    """Extract and filter frontier candidates for an RL decision."""

    candidates = frontier_candidates_from_occupancy_grid(
        msg,
        min_cluster_size=min_cluster_size,
    )

    return filter_eligible_frontier_candidates(
        candidates,
        robot_x=robot_x,
        robot_y=robot_y,
        visited_goals=visited_goals,
        min_robot_distance=min_robot_distance,
        visited_goal_radius=visited_goal_radius,
    )


def explored_area_m2_from_occupancy_grid(msg):
    """Return known occupancy-grid area in square meters."""

    expected_size = (
        msg.info.width
        * msg.info.height
    )

    if len(msg.data) != expected_size:
        raise ValueError(
            f'Occupancy data has {len(msg.data)} cells; '
            f'expected {expected_size}.'
        )

    resolution = float(
        msg.info.resolution
    )

    if resolution <= 0.0:
        raise ValueError(
            'Occupancy grid resolution must be positive.'
        )

    known_cells = sum(
        value != -1
        for value in msg.data
    )

    return (
        known_cells
        * resolution
        * resolution
    )
