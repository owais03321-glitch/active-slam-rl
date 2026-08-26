# Frontier Baseline Experimental Protocol

## Purpose

This protocol defines the reproducible frontier-exploration baseline used for
later comparison with the reinforcement-learning Active SLAM policy.

The baseline implementation is frozen from Git commit:

`40093e4 - fix: prevent frontier livelock and clean shutdown`

All formal baseline runs must use the same simulation configuration, SLAM/Nav2
stack, metrics definitions, reset procedure, and evaluation horizon.

## Simulation Configuration

ROS distribution:

- ROS 2 Jazzy

Simulation launch:

```bash
ros2 launch nav2_bringup tb3_simulation_launch.py \
  slam:=True \
  use_rviz:=False \
  headless:=True
```

World label recorded by the experiment runner:

`nav2_tb3_default_simulation`

The default launch spawn pose and navigation configuration must not be changed
between baseline repetitions.

## Exploration Policy

Algorithm:

`frontier_baseline`

The baseline:

1. detects free cells adjacent to unknown space,
2. clusters frontier cells,
3. rejects frontier clusters smaller than 5 cells,
4. chooses a representative frontier cell near each cluster centroid,
5. rejects goals closer than 0.35 m to the robot,
6. rejects goals within 0.50 m of previously completed frontier goals,
7. selects the nearest remaining valid frontier,
8. navigates using Nav2 NavigateToPose.

Completed frontier memory is bounded to the most recent 100 goals.

## Number of Runs

Perform 5 independent baseline repetitions:

- frontier_baseline_01
- frontier_baseline_02
- frontier_baseline_03
- frontier_baseline_04
- frontier_baseline_05

Each repetition must begin from a completely fresh simulator and SLAM process.

Sanity/debug runs are not part of the scientific dataset.

## Reset Procedure

Before every formal run:

1. Stop the complete Gazebo/Nav2/SLAM simulation.
2. Verify no simulation, baseline_metrics, or frontier_explorer processes remain.
3. Restart the ROS 2 daemon.
4. Verify the ROS node graph is empty.
5. Start the simulation using the fixed launch command.
6. Wait until `/slam_toolbox` and `/bt_navigator` are available.
7. Verify `/baseline_metrics` and `/frontier_explorer` are absent.
8. Start the experiment runner.

A previously accumulated SLAM map must never be reused for another repetition.

## Evaluation Horizon

The primary evaluation horizon is:

`exploration_elapsed_s = 300.0 seconds`

The exploration timer begins when the metrics collector observes the first
unique Nav2 navigation goal.

The experiment process must run long enough to contain at least 300 seconds of
exploration time.

For primary analysis, metrics after 300 seconds are ignored.

Because the runner includes startup time before exploration begins, formal runs
should be allowed approximately 330 seconds of wall-clock execution before
shutdown.

## Recorded Raw Data

Each run must preserve:

- `metrics.csv`
- `summary.json`
- `baseline_metrics.log`
- `frontier_explorer.log`
- `metadata.txt`
- `git_commit.txt`
- `git_status_at_start.txt`
- `ros_nodes_at_start.txt`
- `ros_nodes_at_end.txt`
- `ros_topics_at_start.txt`
- `ros_parameters_at_start.txt`

Raw run directories must not be manually edited after collection.

## Primary Metrics

Metrics will be evaluated at or immediately before
`exploration_elapsed_s = 300.0 s`.

Primary comparison metrics:

1. Newly explored area (`m^2`)
2. Path length (`m`)
3. Exploration efficiency (`newly explored m^2 / m`)
4. Navigation goal success rate
5. Newly explored area versus exploration time

Area per second may also be reported, but at a fixed 300-second horizon it is
derived directly from newly explored area and time.

## Coverage and T90/T99

`target_area_m2` is currently intentionally unset.

Therefore:

- `coverage_pct`
- `t90_s`
- `t99_s`

must not be used in formal conclusions yet.

A fixed world-specific reference explorable area must be established before
T90/T99 are enabled.

The reference area must be defined once and then used unchanged for both the
frontier baseline and RL evaluation.

## Valid Run Criteria

A formal run is valid when:

- the simulator starts from a fresh state,
- the Git working tree is clean at run start,
- exploration starts successfully,
- at least 300 seconds of exploration data are recorded,
- metrics and logs are successfully saved,
- no metrics/explorer traceback or RCLError occurs,
- the simulator does not crash during the evaluation horizon.

Navigation failures such as aborted goals are algorithm outcomes and must be
recorded; they are not grounds for discarding a run.

## Reporting

For the five baseline repetitions, report at minimum:

- individual run values,
- arithmetic mean,
- standard deviation,
- minimum,
- maximum.

The same evaluation protocol and metric definitions must later be applied to
the trained RL policy.

No sanity or debugging run may be mixed with the five formal repetitions.
