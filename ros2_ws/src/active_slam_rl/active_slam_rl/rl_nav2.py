from nav2_msgs.action import NavigateToPose


def make_navigation_goal(
    *,
    x,
    y,
    stamp,
):
    """Build a Nav2 goal for one selected RL frontier."""

    goal = NavigateToPose.Goal()

    goal.pose.header.frame_id = 'map'
    goal.pose.header.stamp = stamp

    goal.pose.pose.position.x = float(x)
    goal.pose.pose.position.y = float(y)

    goal.pose.pose.orientation.w = 1.0

    return goal
