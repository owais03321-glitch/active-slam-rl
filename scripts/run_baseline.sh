#!/usr/bin/env bash

set -euo pipefail


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage:
  ./scripts/run_baseline.sh RUN_ID [WORLD_LABEL]

Example:
  ./scripts/run_baseline.sh frontier_run01 nav2_tb3_default_simulation

Requirements:
  - A fresh Gazebo + SLAM + Nav2 stack must already be running.
  - frontier_explorer and baseline_metrics must NOT already be running.
  - Repository should be clean before collecting scientific baseline data.

Stop a run with Ctrl+C.
The script will stop the explorer and metrics collector cleanly and preserve
the run files.
EOF
}


if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    usage
    exit 1
fi


RUN_ID="$1"
WORLD_LABEL="${2:-nav2_tb3_default_simulation}"

PROJECT_DIR="$HOME/projects/active-slam-rl"
ROS_WS="$PROJECT_DIR/ros2_ws"
RUN_DIR="$PROJECT_DIR/experiments/baseline/runs/$RUN_ID"

METRICS_LOG="$RUN_DIR/baseline_metrics.log"
EXPLORER_LOG="$RUN_DIR/frontier_explorer.log"

METRICS_PID=""
EXPLORER_PID=""

CLEANUP_DONE=0


# ---------------------------------------------------------------------------
# ROS environment
# ---------------------------------------------------------------------------

cd "$ROS_WS"

source /opt/ros/jazzy/setup.bash
source install/setup.bash


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

echo "===== BASELINE RUN PRE-FLIGHT ====="
echo "run_id      : $RUN_ID"
echo "world       : $WORLD_LABEL"
echo "project     : $PROJECT_DIR"
echo


if [ -e "$RUN_DIR" ]; then
    echo "ERROR: Run directory already exists:"
    echo "  $RUN_DIR"
    echo
    echo "Refusing to overwrite previous experimental data."
    exit 1
fi


if ros2 node list 2>/dev/null | grep -qx '/baseline_metrics'; then
    echo "ERROR: /baseline_metrics is already running."
    exit 1
fi


if ros2 node list 2>/dev/null | grep -qx '/frontier_explorer'; then
    echo "ERROR: /frontier_explorer is already running."
    exit 1
fi


if ! ros2 node list 2>/dev/null | grep -qx '/slam_toolbox'; then
    echo "ERROR: /slam_toolbox is not running."
    echo "Start a fresh SLAM/Nav2 simulation first."
    exit 1
fi


if ! ros2 node list 2>/dev/null | grep -qx '/bt_navigator'; then
    echo "ERROR: /bt_navigator is not running."
    echo "Start Nav2 before collecting a baseline run."
    exit 1
fi


cd "$PROJECT_DIR"

GIT_COMMIT="$(git rev-parse HEAD)"
GIT_BRANCH="$(git branch --show-current)"
GIT_STATUS="$(git status --porcelain)"

if [ -n "$GIT_STATUS" ]; then
    echo "ERROR: Git working tree is not clean."
    echo
    git status --short
    echo
    echo "Commit or intentionally resolve changes before a real baseline run."
    exit 1
fi


# ---------------------------------------------------------------------------
# Create permanent run directory
# ---------------------------------------------------------------------------

mkdir -p "$RUN_DIR"


# ---------------------------------------------------------------------------
# Metadata captured before exploration begins
# ---------------------------------------------------------------------------

START_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

cat > "$RUN_DIR/metadata.txt" <<EOF
run_id=$RUN_ID
run_type=baseline
algorithm=frontier_baseline
world=$WORLD_LABEL
git_commit=$GIT_COMMIT
git_branch=$GIT_BRANCH
ros_distro=${ROS_DISTRO:-unknown}
start_timestamp_utc=$START_UTC
simulation_launch_command=ros2 launch nav2_bringup tb3_simulation_launch.py slam:=True use_rviz:=False headless:=True
metrics_command=ros2 run active_slam_rl baseline_metrics --ros-args -p use_sim_time:=true
explorer_command=ros2 run active_slam_rl frontier_explorer --ros-args -p use_sim_time:=true
EOF


git status --short > "$RUN_DIR/git_status_at_start.txt"
git log -1 --format=fuller > "$RUN_DIR/git_commit.txt"

ros2 node list | sort > "$RUN_DIR/ros_nodes_at_start.txt"
ros2 topic list -t | sort > "$RUN_DIR/ros_topics_at_start.txt"


{
    echo "===== slam_toolbox ====="
    ros2 param dump /slam_toolbox 2>&1 || true

    echo
    echo "===== bt_navigator ====="
    ros2 param dump /bt_navigator 2>&1 || true

    echo
    echo "===== controller_server ====="
    ros2 param dump /controller_server 2>&1 || true

    echo
    echo "===== planner_server ====="
    ros2 param dump /planner_server 2>&1 || true
} > "$RUN_DIR/ros_parameters_at_start.txt"


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

cleanup() {
    if [ "$CLEANUP_DONE" -eq 1 ]; then
        return
    fi

    CLEANUP_DONE=1

    echo
    echo "===== STOPPING BASELINE RUN ====="

    if [ -n "$EXPLORER_PID" ] && kill -0 "$EXPLORER_PID" 2>/dev/null; then
        echo "Stopping frontier_explorer..."
        kill -INT -- "-$EXPLORER_PID" 2>/dev/null || true
    fi

    sleep 1

    if [ -n "$METRICS_PID" ] && kill -0 "$METRICS_PID" 2>/dev/null; then
        echo "Stopping baseline_metrics..."
        kill -INT -- "-$METRICS_PID" 2>/dev/null || true
    fi

    # Allow baseline_metrics time to execute finalize() and write summary.json.
    sleep 2

    wait "$EXPLORER_PID" 2>/dev/null || true
    wait "$METRICS_PID" 2>/dev/null || true

    END_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

    {
        echo
        echo "end_timestamp_utc=$END_UTC"
    } >> "$RUN_DIR/metadata.txt"

    ros2 node list 2>/dev/null | sort \
        > "$RUN_DIR/ros_nodes_at_end.txt" || true

    echo
    echo "===== RUN FILES ====="
    ls -lh "$RUN_DIR"

    echo

    if [ -f "$RUN_DIR/summary.json" ]; then
        echo "Run summary saved:"
        echo "  $RUN_DIR/summary.json"
    else
        echo "WARNING: summary.json was not created."
        echo "Inspect:"
        echo "  $METRICS_LOG"
    fi

    echo
    echo "Run directory:"
    echo "  $RUN_DIR"
}


trap cleanup INT TERM EXIT


# ---------------------------------------------------------------------------
# Start passive metrics collector first.
# This records the initial SLAM map before exploration begins.
# ---------------------------------------------------------------------------

echo
echo "===== STARTING METRICS COLLECTOR ====="

cd "$ROS_WS"

setsid ros2 run active_slam_rl baseline_metrics \
    --ros-args \
    -p use_sim_time:=true \
    -p output_dir:="$RUN_DIR" \
    -p run_id:="$RUN_ID" \
    > "$METRICS_LOG" 2>&1 &

METRICS_PID=$!

echo "baseline_metrics PID: $METRICS_PID"


# Give the node enough time to subscribe to /map and record initial state.
sleep 6


if ! kill -0 "$METRICS_PID" 2>/dev/null; then
    echo "ERROR: baseline_metrics exited unexpectedly."
    echo
    cat "$METRICS_LOG"
    exit 1
fi


if [ ! -f "$RUN_DIR/metrics.csv" ]; then
    echo "ERROR: metrics.csv was not created."
    echo
    cat "$METRICS_LOG"
    exit 1
fi


echo "Metrics collector is running."
echo "Initial map data is being saved."


# ---------------------------------------------------------------------------
# Start autonomous frontier baseline
# ---------------------------------------------------------------------------

echo
echo "===== STARTING FRONTIER EXPLORER ====="

setsid ros2 run active_slam_rl frontier_explorer \
    --ros-args \
    -p use_sim_time:=true \
    > "$EXPLORER_LOG" 2>&1 &

EXPLORER_PID=$!

echo "frontier_explorer PID: $EXPLORER_PID"

sleep 3


if ! kill -0 "$EXPLORER_PID" 2>/dev/null; then
    echo "ERROR: frontier_explorer exited unexpectedly."
    echo
    cat "$EXPLORER_LOG"
    exit 1
fi


echo
echo "===== BASELINE RUN ACTIVE ====="
echo "Run ID:"
echo "  $RUN_ID"
echo
echo "Data directory:"
echo "  $RUN_DIR"
echo
echo "Metrics log:"
echo "  $METRICS_LOG"
echo
echo "Explorer log:"
echo "  $EXPLORER_LOG"
echo
echo "Press Ctrl+C when the run should end."


# ---------------------------------------------------------------------------
# Keep wrapper alive until stopped by the user or a child exits.
# ---------------------------------------------------------------------------

while true; do

    if ! kill -0 "$METRICS_PID" 2>/dev/null; then
        echo
        echo "ERROR: baseline_metrics stopped unexpectedly."
        break
    fi

    if ! kill -0 "$EXPLORER_PID" 2>/dev/null; then
        echo
        echo "frontier_explorer has stopped."
        break
    fi

    sleep 1
done
