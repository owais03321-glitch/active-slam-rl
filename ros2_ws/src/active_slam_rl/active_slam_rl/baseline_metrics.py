#!/usr/bin/env python3

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.node import Node

from action_msgs.msg import GoalStatus, GoalStatusArray
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


class BaselineMetrics(Node):
    """
    Passive metrics collector for the autonomous frontier baseline.

    Subscribes to:
      - /map
      - /odom
      - /navigate_to_pose/_action/status

    Saves:
      - metrics.csv       time-series measurements
      - summary.json      final run summary
    """

    def __init__(self):
        super().__init__("baseline_metrics")

        # --------------------------------------------------------------
        # Parameters
        # --------------------------------------------------------------
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter(
            "nav_status_topic",
            "/navigate_to_pose/_action/status",
        )

        self.declare_parameter("output_dir", "metrics")
        self.declare_parameter("run_id", "")

        # Set later when we know the true explorable area of a world.
        self.declare_parameter("target_area_m2", 0.0)

        # Prevent simulator resets / teleportation from being counted as
        # physical robot travel.
        self.declare_parameter("max_odom_step_m", 1.0)

        map_topic = self.get_parameter(
            "map_topic"
        ).get_parameter_value().string_value

        odom_topic = self.get_parameter(
            "odom_topic"
        ).get_parameter_value().string_value

        nav_status_topic = self.get_parameter(
            "nav_status_topic"
        ).get_parameter_value().string_value

        output_dir = self.get_parameter(
            "output_dir"
        ).get_parameter_value().string_value

        run_id = self.get_parameter(
            "run_id"
        ).get_parameter_value().string_value

        self.target_area_m2 = self.get_parameter(
            "target_area_m2"
        ).get_parameter_value().double_value

        self.max_odom_step_m = self.get_parameter(
            "max_odom_step_m"
        ).get_parameter_value().double_value

        # --------------------------------------------------------------
        # Run identity
        # --------------------------------------------------------------
        if not run_id:
            timestamp = datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%SZ"
            )
            run_id = f"frontier_baseline_{timestamp}"

        self.run_id = run_id

        # --------------------------------------------------------------
        # Output files
        # --------------------------------------------------------------
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.csv_path = self.output_dir / "metrics.csv"
        self.summary_path = self.output_dir / "summary.json"

        self.csv_file = self.csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )

        self.fieldnames = [
            "ros_time_s",
            "collector_elapsed_s",
            "exploration_elapsed_s",
            "map_width_cells",
            "map_height_cells",
            "map_resolution_m",
            "total_cells",
            "known_cells",
            "unknown_cells",
            "explored_area_m2",
            "initial_explored_area_m2",
            "newly_explored_area_m2",
            "known_fraction",
            "coverage_pct",
            "path_length_m",
            "area_per_second",
            "area_per_meter",
            "goals_seen",
            "goals_succeeded",
            "goals_aborted",
            "goals_canceled",
            "goal_success_rate",
        ]

        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=self.fieldnames,
        )

        self.writer.writeheader()
        self.csv_file.flush()

        # --------------------------------------------------------------
        # Runtime state
        # --------------------------------------------------------------
        self.start_time_s = None
        self.exploration_start_time_s = None

        self.last_odom_xy = None
        self.path_length_m = 0.0

        self.initial_explored_area_m2 = None
        self.latest_explored_area_m2 = 0.0
        self.latest_newly_explored_area_m2 = 0.0
        self.latest_known_fraction = 0.0
        self.latest_coverage_pct = None

        self.map_updates = 0

        self.goals_seen = set()
        self.terminal_goal_status = {}

        self.t90_s = None
        self.t99_s = None

        # --------------------------------------------------------------
        # QoS
        #
        # /map is normally transient-local in Nav2 / SLAM systems.
        # Matching that QoS makes the metrics node more robust when it
        # starts after the map publisher.
        # --------------------------------------------------------------
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # --------------------------------------------------------------
        # Subscribers
        # --------------------------------------------------------------
        self.create_subscription(
            OccupancyGrid,
            map_topic,
            self.map_callback,
            map_qos,
        )

        self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            50,
        )

        self.create_subscription(
            GoalStatusArray,
            nav_status_topic,
            self.goal_status_callback,
            10,
        )

        self.get_logger().info(
            f"Baseline run ID: {self.run_id}"
        )

        self.get_logger().info(
            f"Metrics CSV: {self.csv_path}"
        )

        self.get_logger().info(
            f"Summary JSON: {self.summary_path}"
        )

        if self.target_area_m2 > 0.0:
            self.get_logger().info(
                "Target explorable area: "
                f"{self.target_area_m2:.3f} m^2"
            )
        else:
            self.get_logger().warn(
                "target_area_m2 is not set. "
                "Raw explored area will be recorded, "
                "but T90/T99 will remain disabled."
            )

    def ros_time_s(self):
        """Return current ROS clock time in seconds."""
        return self.get_clock().now().nanoseconds / 1e9

    # ------------------------------------------------------------------
    # Odometry
    # ------------------------------------------------------------------
    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if (
            self.exploration_start_time_s is not None
            and self.last_odom_xy is not None
        ):
            last_x, last_y = self.last_odom_xy

            step_distance = math.hypot(
                x - last_x,
                y - last_y,
            )

            if step_distance <= self.max_odom_step_m:
                self.path_length_m += step_distance
            else:
                self.get_logger().warn(
                    "Ignoring odometry jump: "
                    f"{step_distance:.3f} m"
                )

        self.last_odom_xy = (x, y)

    # ------------------------------------------------------------------
    # Nav2 action status
    # ------------------------------------------------------------------
    def goal_status_callback(self, msg: GoalStatusArray):
        terminal_states = {
            GoalStatus.STATUS_SUCCEEDED,
            GoalStatus.STATUS_ABORTED,
            GoalStatus.STATUS_CANCELED,
        }

        for status in msg.status_list:
            goal_uuid = bytes(
                status.goal_info.goal_id.uuid
            ).hex()

            if not goal_uuid:
                continue

            if goal_uuid not in self.goals_seen:
                self.goals_seen.add(goal_uuid)

                if self.exploration_start_time_s is None:
                    self.exploration_start_time_s = self.ros_time_s()

                    # Start distance measurement from the same event as
                    # exploration timing. The next odometry sample becomes
                    # the path-length origin.
                    self.last_odom_xy = None
                    self.path_length_m = 0.0

                    self.get_logger().info(
                        "Exploration timer and path measurement "
                        "started on first Nav2 goal."
                    )

            if (
                status.status in terminal_states
                and goal_uuid not in self.terminal_goal_status
            ):
                self.terminal_goal_status[goal_uuid] = status.status

    def get_goal_counts(self):
        succeeded = sum(
            status == GoalStatus.STATUS_SUCCEEDED
            for status in self.terminal_goal_status.values()
        )

        aborted = sum(
            status == GoalStatus.STATUS_ABORTED
            for status in self.terminal_goal_status.values()
        )

        canceled = sum(
            status == GoalStatus.STATUS_CANCELED
            for status in self.terminal_goal_status.values()
        )

        completed = succeeded + aborted + canceled

        if completed > 0:
            success_rate = succeeded / completed
        else:
            success_rate = 0.0

        return (
            succeeded,
            aborted,
            canceled,
            success_rate,
        )

    # ------------------------------------------------------------------
    # Occupancy map
    # ------------------------------------------------------------------
    def map_callback(self, msg: OccupancyGrid):
        now_s = self.ros_time_s()

        if self.start_time_s is None:
            self.start_time_s = now_s

        collector_elapsed_s = max(
            0.0,
            now_s - self.start_time_s,
        )

        if self.exploration_start_time_s is not None:
            exploration_elapsed_s = max(
                0.0,
                now_s - self.exploration_start_time_s,
            )
        else:
            exploration_elapsed_s = 0.0

        resolution = float(msg.info.resolution)
        total_cells = len(msg.data)

        known_cells = sum(
            1
            for value in msg.data
            if value != -1
        )

        unknown_cells = total_cells - known_cells

        explored_area_m2 = (
            known_cells
            * resolution
            * resolution
        )

        # The first received map establishes the amount of area that was
        # already known when this metrics run began. Exploration
        # efficiency must measure only additional area discovered after
        # that point.
        if self.initial_explored_area_m2 is None:
            self.initial_explored_area_m2 = explored_area_m2

            self.get_logger().info(
                "Initial known map area: "
                f"{self.initial_explored_area_m2:.3f} m^2"
            )

        newly_explored_area_m2 = max(
            0.0,
            explored_area_m2
            - self.initial_explored_area_m2,
        )

        if total_cells > 0:
            known_fraction = (
                known_cells / total_cells
            )
        else:
            known_fraction = 0.0

        coverage_pct = None

        # --------------------------------------------------------------
        # Coverage relative to a fixed known world area.
        # Do not enable this until target_area_m2 is established.
        # --------------------------------------------------------------
        if self.target_area_m2 > 0.0:
            coverage_pct = min(
                100.0,
                100.0
                * explored_area_m2
                / self.target_area_m2,
            )

            if (
                self.exploration_start_time_s is not None
                and self.t90_s is None
                and coverage_pct >= 90.0
            ):
                self.t90_s = exploration_elapsed_s

                self.get_logger().info(
                    f"T90 reached at {exploration_elapsed_s:.2f} s"
                )

            if (
                self.exploration_start_time_s is not None
                and self.t99_s is None
                and coverage_pct >= 99.0
            ):
                self.t99_s = exploration_elapsed_s

                self.get_logger().info(
                    f"T99 reached at {exploration_elapsed_s:.2f} s"
                )

        # --------------------------------------------------------------
        # Efficiency metrics
        # --------------------------------------------------------------
        if exploration_elapsed_s > 0.0:
            area_per_second = (
                newly_explored_area_m2
                / exploration_elapsed_s
            )
        else:
            area_per_second = 0.0

        if self.path_length_m > 0.0:
            area_per_meter = (
                newly_explored_area_m2
                / self.path_length_m
            )
        else:
            area_per_meter = 0.0

        (
            succeeded,
            aborted,
            canceled,
            success_rate,
        ) = self.get_goal_counts()

        # --------------------------------------------------------------
        # Remember latest state for summary.json
        # --------------------------------------------------------------
        self.latest_explored_area_m2 = explored_area_m2
        self.latest_newly_explored_area_m2 = newly_explored_area_m2
        self.latest_known_fraction = known_fraction
        self.latest_coverage_pct = coverage_pct

        self.map_updates += 1

        # --------------------------------------------------------------
        # Save time-series sample immediately
        # --------------------------------------------------------------
        self.writer.writerow(
            {
                "ros_time_s": now_s,
                "collector_elapsed_s": collector_elapsed_s,
                "exploration_elapsed_s": exploration_elapsed_s,
                "map_width_cells": msg.info.width,
                "map_height_cells": msg.info.height,
                "map_resolution_m": resolution,
                "total_cells": total_cells,
                "known_cells": known_cells,
                "unknown_cells": unknown_cells,
                "explored_area_m2": explored_area_m2,
                "initial_explored_area_m2": (
                    self.initial_explored_area_m2
                ),
                "newly_explored_area_m2": (
                    newly_explored_area_m2
                ),
                "known_fraction": known_fraction,
                "coverage_pct": (
                    coverage_pct
                    if coverage_pct is not None
                    else ""
                ),
                "path_length_m": self.path_length_m,
                "area_per_second": area_per_second,
                "area_per_meter": area_per_meter,
                "goals_seen": len(self.goals_seen),
                "goals_succeeded": succeeded,
                "goals_aborted": aborted,
                "goals_canceled": canceled,
                "goal_success_rate": success_rate,
            }
        )

        # Flush every map update so results survive an unexpected stop.
        self.csv_file.flush()

    # ------------------------------------------------------------------
    # Final run summary
    # ------------------------------------------------------------------
    def finalize(self):
        if self.csv_file.closed:
            return

        now_s = self.ros_time_s()

        if self.start_time_s is not None:
            collector_elapsed_s = max(
                0.0,
                now_s - self.start_time_s,
            )
        else:
            collector_elapsed_s = 0.0

        if self.exploration_start_time_s is not None:
            exploration_elapsed_s = max(
                0.0,
                now_s - self.exploration_start_time_s,
            )
        else:
            exploration_elapsed_s = 0.0

        (
            succeeded,
            aborted,
            canceled,
            success_rate,
        ) = self.get_goal_counts()

        if exploration_elapsed_s > 0.0:
            area_per_second = (
                self.latest_newly_explored_area_m2
                / exploration_elapsed_s
            )
        else:
            area_per_second = 0.0

        if self.path_length_m > 0.0:
            area_per_meter = (
                self.latest_newly_explored_area_m2
                / self.path_length_m
            )
        else:
            area_per_meter = 0.0

        summary = {
            "run_id": self.run_id,
            "algorithm": "frontier_baseline",
            "collector_elapsed_s": collector_elapsed_s,
            "exploration_elapsed_s": exploration_elapsed_s,
            "exploration_started": (
                self.exploration_start_time_s is not None
            ),
            "explored_area_m2": self.latest_explored_area_m2,
            "initial_explored_area_m2": (
                self.initial_explored_area_m2
            ),
            "newly_explored_area_m2": (
                self.latest_newly_explored_area_m2
            ),
            "known_fraction": self.latest_known_fraction,
            "coverage_pct": self.latest_coverage_pct,
            "path_length_m": self.path_length_m,
            "area_per_second": area_per_second,
            "area_per_meter": area_per_meter,
            "target_area_m2": (
                self.target_area_m2
                if self.target_area_m2 > 0.0
                else None
            ),
            "t90_s": self.t90_s,
            "t99_s": self.t99_s,
            "map_updates": self.map_updates,
            "navigation": {
                "goals_seen": len(self.goals_seen),
                "succeeded": succeeded,
                "aborted": aborted,
                "canceled": canceled,
                "success_rate": success_rate,
            },
        }

        with self.summary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=2,
            )

        self.csv_file.flush()
        self.csv_file.close()

        self.get_logger().info(
            f"Saved final summary: {self.summary_path}"
        )


def main(args=None):
    rclpy.init(args=args)

    node = BaselineMetrics()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info(
            "Metrics collection stopped by user."
        )

    finally:
        node.finalize()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
