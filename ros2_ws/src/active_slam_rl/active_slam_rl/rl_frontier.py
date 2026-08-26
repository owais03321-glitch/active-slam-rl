import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class FrontierCandidate:
    """One frontier target available to the RL exploration policy."""

    cell_x: int
    cell_y: int
    world_x: float
    world_y: float
    cluster_size: int


def extract_frontier_candidates(
    *,
    width,
    height,
    data,
    resolution,
    origin_x,
    origin_y,
    min_cluster_size=5,
):
    """Extract frontier candidates using the frozen baseline definition."""

    expected_size = width * height

    if len(data) != expected_size:
        raise ValueError(
            f'Occupancy data has {len(data)} cells; '
            f'expected {expected_size}.'
        )

    frontier_cells = set()

    # Match the frozen classical baseline:
    # a frontier is a free cell touching unknown space
    # in the 4-connected neighborhood.
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            index = y * width + x

            if data[index] != 0:
                continue

            neighbors = (
                data[index - 1],
                data[index + 1],
                data[index - width],
                data[index + width],
            )

            if -1 in neighbors:
                frontier_cells.add((x, y))

    clusters = _cluster_frontiers(
        frontier_cells,
        min_cluster_size=min_cluster_size,
    )

    candidates = []

    for cluster in clusters:
        avg_x = sum(point[0] for point in cluster) / len(cluster)
        avg_y = sum(point[1] for point in cluster) / len(cluster)

        cell_x, cell_y = min(
            cluster,
            key=lambda point: (
                (point[0] - avg_x) ** 2
                + (point[1] - avg_y) ** 2
            ),
        )

        world_x = (
            origin_x
            + (cell_x + 0.5) * resolution
        )

        world_y = (
            origin_y
            + (cell_y + 0.5) * resolution
        )

        candidates.append(
            FrontierCandidate(
                cell_x=cell_x,
                cell_y=cell_y,
                world_x=world_x,
                world_y=world_y,
                cluster_size=len(cluster),
            )
        )

    # The frozen baseline only needed the nearest candidate, so cluster
    # iteration order was irrelevant. RL actions use integer indices,
    # therefore the RL-side representation must be deterministic.
    candidates.sort(
        key=lambda candidate: (
            candidate.cell_y,
            candidate.cell_x,
        )
    )

    return candidates



def filter_eligible_frontier_candidates(
    candidates,
    *,
    robot_x,
    robot_y,
    visited_goals=(),
    min_robot_distance=0.35,
    visited_goal_radius=0.50,
):
    """Apply the frozen baseline frontier eligibility rules."""

    eligible = []

    for candidate in candidates:
        robot_distance = math.hypot(
            candidate.world_x - robot_x,
            candidate.world_y - robot_y,
        )

        if robot_distance <= min_robot_distance:
            continue

        already_visited = any(
            math.hypot(
                candidate.world_x - visited_x,
                candidate.world_y - visited_y,
            ) <= visited_goal_radius
            for visited_x, visited_y
            in visited_goals
        )

        if already_visited:
            continue

        eligible.append(candidate)

    return eligible


def _cluster_frontiers(
    frontier_cells,
    *,
    min_cluster_size,
):
    """Group frontier cells using 8-connected components."""

    remaining = set(frontier_cells)
    clusters = []

    while remaining:
        start = min(
            remaining,
            key=lambda point: (
                point[1],
                point[0],
            ),
        )
        remaining.remove(start)

        queue = deque([start])
        cluster = [start]

        while queue:
            x, y = queue.popleft()

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue

                    neighbor = (
                        x + dx,
                        y + dy,
                    )

                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        queue.append(neighbor)
                        cluster.append(neighbor)

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    return clusters
