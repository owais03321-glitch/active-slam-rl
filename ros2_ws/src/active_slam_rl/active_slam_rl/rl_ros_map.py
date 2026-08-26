from active_slam_rl.rl_frontier import (
    extract_frontier_candidates,
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
