import pytest
from builtin_interfaces.msg import Time

from active_slam_rl.rl_nav2 import (
    make_navigation_goal,
)


def test_navigation_goal_matches_frozen_baseline_contract():
    stamp = Time(
        sec=123,
        nanosec=456,
    )

    goal = make_navigation_goal(
        x=1.25,
        y=-0.75,
        stamp=stamp,
    )

    assert goal.pose.header.frame_id == 'map'
    assert goal.pose.header.stamp == stamp

    assert (
        goal.pose.pose.position.x
        == pytest.approx(1.25)
    )

    assert (
        goal.pose.pose.position.y
        == pytest.approx(-0.75)
    )

    assert (
        goal.pose.pose.orientation.x
        == pytest.approx(0.0)
    )

    assert (
        goal.pose.pose.orientation.y
        == pytest.approx(0.0)
    )

    assert (
        goal.pose.pose.orientation.z
        == pytest.approx(0.0)
    )

    assert (
        goal.pose.pose.orientation.w
        == pytest.approx(1.0)
    )


def test_navigation_goal_converts_coordinates_to_float():
    stamp = Time()

    goal = make_navigation_goal(
        x=2,
        y=3,
        stamp=stamp,
    )

    assert goal.pose.pose.position.x == 2.0
    assert goal.pose.pose.position.y == 3.0
