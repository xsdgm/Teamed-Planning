#!/bin/bash
# main.sh — Unified operator interface for Teamed Planning system.
# Run this script on the host machine (sim) or directly on the ground robot (real).
#
# Usage:
#   ./run.sh [--sim|--real] <command> [args]
#
# Commands:
#   start                  Boot infrastructure + planning nodes, block until ready
#   takeoff                Sequential takeoff of all drones
#   land                   Land all drones and stop planning nodes
#   stop                   Full shutdown: land + kill Gazebo, MAVROS, ROS master
#   navigate <x> <y>       Send goal to ground robot and start drone mission
#   rviz                   Open RViz GCS view
#   status                 Show current robot states
#
# Flags:
#   --sim    (default) Use Docker containers — for XTDrone/Gazebo simulation
#   --real   Run commands directly — for real hardware deployment on ground robot

set -e

# ── Config ────────────────────────────────────────────────────────────────────
XTDRONE_CTR="xtdrone-ad"
PLANNING_CTR="teamedplanning-ad"
# PLANNING_WS: path inside the planning container (used for docker exec commands)
PLANNING_WS="/home/dcist/Teamed-Planning"
# HOST_WS: path on the host machine (used for docker cp and host-side operations)
# Derived from this script's location: deploy/ → teamed-planning/ → src/ → workspace root
HOST_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NUM_DRONES=3
MODE="sim"

# ── Parse flags & command ─────────────────────────────────────────────────────
COMMAND=""
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --sim)  MODE="sim" ;;
        --real) MODE="real" ;;
        *)
            if [ -z "$COMMAND" ]; then
                COMMAND="$arg"
            else
                ARGS+=("$arg")
            fi
            ;;
    esac
done

if [ -z "$COMMAND" ]; then
    echo "Usage: $0 [--sim|--real] <start|takeoff|land|stop|navigate <x> <y>|rviz|status>"
    exit 1
fi

# ── ROS environment helper ────────────────────────────────────────────────────
# run_ros "ros_cmd" — Executes a ROS command in the right environment.
# In sim mode, runs inside the planning container via docker exec.
# In real mode, runs directly on the current host.
run_ros() {
    local ros_cmd="$1"
    local env_setup="source /opt/ros/noetic/setup.bash && \
                     source ${PLANNING_WS}/devel/setup.bash && \
                     export ROS_MASTER_URI=http://localhost:11311 && \
                     export ROS_IP=127.0.0.1"
    if [ "$MODE" = "sim" ]; then
        docker exec "$PLANNING_CTR" bash -c "${env_setup} && ${ros_cmd}"
    else
        bash -c "${env_setup} && ${ros_cmd}"
    fi
}

# ── Helper: wait for MAVROS FCU connection (sim only) ────────────────────────
# Checks connected: True, not just topic existence — MAVROS publishes the topic
# even when it hasn't established a link to the PX4 FCU yet.
wait_for_mavros() {
    local drone_num=$1
    local ns="iris_${drone_num}"
    echo "[teamed] Waiting for MAVROS FCU connection on ${ns}..."
    for i in $(seq 1 45); do
        if docker exec "$XTDRONE_CTR" bash -c \
            "source /opt/ros/noetic/setup.bash && \
             export ROS_MASTER_URI=http://localhost:11311 && \
             export ROS_IP=127.0.0.1 && \
             timeout 2 rostopic echo -n 1 /${ns}/mavros/state 2>/dev/null | \
             grep -q 'connected: True'" 2>/dev/null; then
            echo "[teamed] ${ns} MAVROS connected to FCU"
            return 0
        fi
        sleep 2
    done
    echo "[teamed] WARNING: ${ns} MAVROS FCU connection not confirmed after 90s"
}

# ── Commands ──────────────────────────────────────────────────────────────────

cmd_start() {
    echo "============================================"
    echo " Teamed Planning — Starting System"
    echo " Mode: ${MODE} | Drones: ${NUM_DRONES}"
    echo "============================================"

    if [ "$MODE" = "sim" ]; then
        # 0. Ensure both containers are running (cold-boot safe)
        for CTR in "$XTDRONE_CTR" "$PLANNING_CTR"; do
            if ! docker ps --format '{{.Names}}' | grep -q "^${CTR}$"; then
                echo "[teamed] Starting container ${CTR}..."
                docker start "$CTR"
                sleep 2
            fi
        done

        # 1. Kill any orphaned sim processes from previous sessions before starting fresh.
        # This prevents multiple competing MAVROS instances fighting over PX4 UDP ports.
        # IMPORTANT: run pkill via direct docker exec (NOT bash -c) to avoid the self-match
        # bug where the bash wrapper's cmdline contains the pkill pattern, causing pkill to
        # send SIGTERM to the bash process itself and abort the cleanup prematurely.
        echo "[teamed] Cleaning up any orphaned processes from previous sessions..."
        docker exec "$XTDRONE_CTR" pkill -9 -f 'mavros_node' > /dev/null 2>&1 || true
        docker exec "$XTDRONE_CTR" pkill -9 -f 'px4'         > /dev/null 2>&1 || true
        docker exec "$XTDRONE_CTR" pkill -9 -f 'gzserver'    > /dev/null 2>&1 || true
        docker exec "$XTDRONE_CTR" pkill -9 -f 'gzclient'    > /dev/null 2>&1 || true
        docker exec "$XTDRONE_CTR" pkill -9 -f 'roslaunch'   > /dev/null 2>&1 || true
        docker exec "$PLANNING_CTR" pkill -9 -f 'teamed_planning' > /dev/null 2>&1 || true
        docker exec "$PLANNING_CTR" pkill -9 -f 'scripts/nodes'   > /dev/null 2>&1 || true
        sleep 2

        # 2. Start Gazebo + PX4 SITL + MAVROS + ground robot
        if docker exec "$XTDRONE_CTR" pgrep gzserver > /dev/null 2>&1; then
            echo "[teamed] Gazebo already running — skipping relaunch."
            echo "[teamed]   (to restart cleanly: ./main.sh --sim stop)"
        else
            echo "[teamed] Starting Gazebo + PX4 SITL + MAVROS + ground robot..."
            docker exec -d "$XTDRONE_CTR" bash -c \
                "source /opt/ros/noetic/setup.bash && \
                 source ~/catkin_ws/devel/setup.bash && \
                 source ~/PX4_Firmware/Tools/setup_gazebo.bash ~/PX4_Firmware/ \
                     ~/PX4_Firmware/build/px4_sitl_default && \
                 export ROS_PACKAGE_PATH=\$ROS_PACKAGE_PATH:~/PX4_Firmware:~/PX4_Firmware/Tools/sitl_gazebo && \
                 export DISPLAY=:0 && \
                 export ROS_MASTER_URI=http://localhost:11311 && \
                 export ROS_IP=127.0.0.1 && \
                 roslaunch px4 multi_vehicle_with_ground.launch gui:=true \
                     > /tmp/sim_bringup.log 2>&1"
            echo "[teamed] Waiting 25s for Gazebo + ground robot to initialise..."
            sleep 25
        fi

        # 2. Wait for MAVROS on each drone
        for i in $(seq 0 $(($NUM_DRONES - 1))); do
            wait_for_mavros $i
        done
        echo "[teamed] All MAVROS instances ready."

        # 2b. Switch all drones to GPS-only EKF mode; disable flight logging (prevents disk fill)
        echo "[teamed] Setting EKF2_AID_MASK=1 and SDLOG_MODE=0 on all drones..."
        for i in $(seq 0 $(($NUM_DRONES - 1))); do
            docker exec "$XTDRONE_CTR" bash -c \
                "source /opt/ros/noetic/setup.bash && \
                 source ~/catkin_ws/devel/setup.bash && \
                 export ROS_MASTER_URI=http://localhost:11311 && \
                 export ROS_IP=127.0.0.1 && \
                 rosservice call /iris_${i}/mavros/param/set \
                 'param_id: \"EKF2_AID_MASK\"
value: {integer: 1, real: 0.0}' && \
                 rosservice call /iris_${i}/mavros/param/set \
                 'param_id: \"SDLOG_MODE\"
value: {integer: 0, real: 0.0}'" 2>/dev/null && \
                echo "[teamed] iris_${i} params set" || \
                echo "[teamed] WARNING: could not set iris_${i} params"
        done

        # 2c. Wait for EKF2 to converge on all drones (checks local_position/odom publishing)
        # EKF2 needs 10-30s after GPS-only mode is set before valid position is available.
        # drone_node rejects takeoff if current_position is None, so this check is critical.
        echo "[teamed] Waiting for EKF2 to converge on all drones (up to 45s)..."
        for i in $(seq 0 $(($NUM_DRONES - 1))); do
            echo -n "[teamed]   iris_${i} EKF2..."
            for attempt in $(seq 1 45); do
                if docker exec "$XTDRONE_CTR" bash -c \
                    "source /opt/ros/noetic/setup.bash && \
                     export ROS_MASTER_URI=http://localhost:11311 && \
                     timeout 1 rostopic echo -n 1 \
                         /iris_${i}/mavros/local_position/odom 2>/dev/null | \
                     grep -q 'frame_id'" 2>/dev/null; then
                    echo " ready (${attempt}s)"
                    break
                fi
                if [ "$attempt" -eq 45 ]; then
                    echo " TIMEOUT — takeoff may fail, check MAVROS"
                else
                    sleep 1
                fi
            done
        done

        # 2c. Kill any stale planning nodes from a previous run (direct pkill, no bash -c)
        echo "[teamed] Cleaning up stale planning nodes (if any)..."
        docker exec "$PLANNING_CTR" pkill -9 -f 'teamed_planning' > /dev/null 2>&1 || true
        docker exec "$PLANNING_CTR" pkill -9 -f 'scripts/nodes'   > /dev/null 2>&1 || true
        sleep 1

        # 3. Start planning nodes inside the planning container
        echo "[teamed] Starting planning nodes..."
        docker exec -d "$PLANNING_CTR" bash -c \
            "source /opt/ros/noetic/setup.bash && \
             source ${PLANNING_WS}/devel/setup.bash && \
             export ROS_MASTER_URI=http://localhost:11311 && \
             export ROS_IP=127.0.0.1 && \
             roslaunch teamed_planning multi_robot_xtdrone.launch \
                 num_drones:=${NUM_DRONES} 2>&1 | \
             tee /tmp/planning_nodes.log"
    else
        # Real mode: start planning nodes directly (MAVROS already running on drones)
        echo "[teamed] Starting planning nodes (real hardware)..."
        bash -c \
            "source /opt/ros/noetic/setup.bash && \
             source ${PLANNING_WS}/devel/setup.bash && \
             export ROS_MASTER_URI=http://localhost:11311 && \
             export ROS_IP=127.0.0.1 && \
             roslaunch teamed_planning multi_robot_real.launch \
                 num_drones:=${NUM_DRONES} 2>&1 | \
             tee /tmp/planning_nodes.log" &
    fi

    # 4. Block until GCS node is confirmed up
    echo "[teamed] Waiting for GCS node..."
    READY=0
    for i in $(seq 1 30); do
        if run_ros "rosnode list 2>/dev/null | grep -q 'gcs_node'" 2>/dev/null; then
            READY=1
            break
        fi
        sleep 2
        echo "[teamed] ... waiting ($((i * 2))s)"
    done

    echo ""
    if [ "$READY" = "1" ]; then
        echo "[teamed] System is UP and ready."
        echo ""
        echo "  Next:  $0 --${MODE} takeoff"
        if [ "$MODE" = "sim" ]; then
            echo "  Logs:  docker exec -it ${PLANNING_CTR} tail -f /tmp/planning_nodes.log"
        else
            echo "  Logs:  tail -f /tmp/planning_nodes.log"
        fi
    else
        echo "[teamed] WARNING: GCS node not detected after 60s — check logs"
        if [ "$MODE" = "sim" ]; then
            echo "  Logs:  docker exec -it ${PLANNING_CTR} tail -f /tmp/planning_nodes.log"
        fi
    fi
}

cmd_takeoff() {
    echo "[teamed] Sending TAKEOFF_ALL..."
    run_ros "rostopic pub -1 /operator/command teamed_planning/RobotCommand \
        '{header: {stamp: now}, command: TAKEOFF_ALL, target_robot_id: ALL, param1: 0.0}'"
    echo ""
    echo "[teamed] Takeoff command sent. Drones will arm and take off sequentially."
    echo "  Monitor:  $0 --${MODE} status"
    echo "  Next:     $0 --${MODE} navigate <x> <y>"
}

cmd_land() {
    echo "[teamed] Sending LAND_ALL..."
    run_ros "rostopic pub -1 /operator/command teamed_planning/RobotCommand \
        '{header: {stamp: now}, command: LAND_ALL, target_robot_id: ALL, param1: 0.0}'"
    echo "[teamed] Land command sent — waiting 10s for drones to land..."
    sleep 10
    echo "[teamed] Stopping planning nodes..."
    if [ "$MODE" = "sim" ]; then
        docker exec "$PLANNING_CTR" pkill -9 -f 'teamed_planning' > /dev/null 2>&1 || true
        docker exec "$PLANNING_CTR" pkill -9 -f 'scripts/nodes'   > /dev/null 2>&1 || true
    else
        pkill -9 -f 'scripts/nodes' 2>/dev/null || true
    fi
    echo "[teamed] Done."
}

cmd_stop() {
    echo "[teamed] Full system shutdown..."

    # 1. Land drones first (best-effort — nodes may already be gone)
    echo "[teamed] Sending LAND_ALL (best effort)..."
    run_ros "rostopic pub -1 /operator/command teamed_planning/RobotCommand \
        '{header: {stamp: now}, command: LAND_ALL, target_robot_id: ALL, param1: 0.0}'" \
        2>/dev/null || true
    sleep 5

    # 2. Kill planning nodes (direct pkill, no bash -c to avoid self-match bug)
    echo "[teamed] Stopping planning nodes..."
    if [ "$MODE" = "sim" ]; then
        docker exec "$PLANNING_CTR" pkill -9 -f 'teamed_planning' > /dev/null 2>&1 || true
        docker exec "$PLANNING_CTR" pkill -9 -f 'scripts/nodes'   > /dev/null 2>&1 || true
    else
        pkill -9 -f 'scripts/nodes' 2>/dev/null || true
    fi

    if [ "$MODE" = "sim" ]; then
        # 3. Kill roslaunch first — it's the session manager and cascades to children
        echo "[teamed] Stopping simulation session (roslaunch → Gazebo → MAVROS → PX4)..."
        docker exec "$XTDRONE_CTR" pkill -f 'roslaunch' > /dev/null 2>&1 || true
        sleep 4

        # 4. Force-kill any remaining processes (direct pkill, no bash -c)
        echo "[teamed] Force-killing remaining processes..."
        docker exec "$XTDRONE_CTR" pkill -9 -f 'gzserver'    > /dev/null 2>&1 || true
        docker exec "$XTDRONE_CTR" pkill -9 -f 'gzclient'    > /dev/null 2>&1 || true
        docker exec "$XTDRONE_CTR" pkill -9 -f 'mavros_node' > /dev/null 2>&1 || true
        docker exec "$XTDRONE_CTR" pkill -9 -f 'px4'         > /dev/null 2>&1 || true
        sleep 2

        # 5. Kill rosmaster last (after children are gone)
        echo "[teamed] Stopping ROS master..."
        docker exec "$XTDRONE_CTR" pkill -9 -f 'rosmaster' > /dev/null 2>&1 || true
        docker exec "$XTDRONE_CTR" pkill -9 -f 'roscore'   > /dev/null 2>&1 || true
        sleep 1

        # 6. Purge PX4 ULog files to prevent disk fill-up over multiple sessions
        echo "[teamed] Purging PX4 flight logs..."
        for i in 0 1 2 3; do
            docker exec "$XTDRONE_CTR" rm -rf "/root/.ros/sitl_iris_${i}/log" || true
        done
    fi

    echo "[teamed] System stopped."
    echo "  To restart: ./main.sh --${MODE} start"
}

cmd_navigate() {
    local TARGET_X="${1:-}"
    local TARGET_Y="${2:-}"
    if [ -z "$TARGET_X" ] || [ -z "$TARGET_Y" ]; then
        echo "Usage: $0 [--sim|--real] navigate <x> <y>"
        echo "  Example: $0 --sim navigate 10.0 5.0"
        exit 1
    fi

    echo "[teamed] Starting navigation — goal: (${TARGET_X}, ${TARGET_Y})"

    # Send target position to ground robot via GCS
    echo "[teamed] Sending goal to ground robot..."
    run_ros "rostopic pub -1 /operator/goal geometry_msgs/PoseStamped \
        '{header: {stamp: now, frame_id: \"map\"}, \
          pose: {position: {x: ${TARGET_X}, y: ${TARGET_Y}, z: 0.0}, \
                 orientation: {w: 1.0}}}'"

    # Start drone mission (HOVERING → MISSION state)
    echo "[teamed] Sending START_MISSION_ALL to drones..."
    run_ros "rostopic pub -1 /operator/command teamed_planning/RobotCommand \
        '{header: {stamp: now}, command: START_MISSION_ALL, target_robot_id: ALL, param1: 0.0}'"

    echo ""
    echo "[teamed] Navigation started."
    echo "  Goal:     (${TARGET_X}, ${TARGET_Y})"
    echo "  Monitor:  $0 --${MODE} status"
    echo "  Stop:     $0 --${MODE} land"
}

cmd_rviz() {
    echo "[teamed] Launching RViz (GCS view)..."
    local RVIZ_CFG="/home/dcist/Teamed-Planning/src/teamed-planning/config/gcs.rviz"
    if [ "$MODE" = "sim" ]; then
        docker exec -it "$PLANNING_CTR" bash -c \
            "source /opt/ros/noetic/setup.bash && \
             export DISPLAY=:0 && \
             export ROS_MASTER_URI=http://localhost:11311 && \
             export ROS_IP=127.0.0.1 && \
             rviz -d ${RVIZ_CFG}"
    else
        bash -c \
            "source /opt/ros/noetic/setup.bash && \
             source ${PLANNING_WS}/devel/setup.bash && \
             export ROS_MASTER_URI=http://localhost:11311 && \
             export ROS_IP=127.0.0.1 && \
             rviz -d ${RVIZ_CFG}"
    fi
}

cmd_status() {
    echo "[teamed] Checking running nodes..."
    run_ros "rosnode list 2>/dev/null | grep -E 'drone_node|ground_node|gcs_node' || \
             echo '  (no planning nodes detected)'"
    echo ""
    echo "[teamed] Drone states:"
    for i in $(seq 0 $(($NUM_DRONES - 1))); do
        run_ros "timeout 2 rostopic echo -n 1 /drone_${i}/status 2>/dev/null | \
                 grep -E 'flight_state|mission_phase|position_[xyz]' || \
                 echo '  drone_${i}: no data'" 2>/dev/null || \
            echo "  drone_${i}: no data"
    done
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "$COMMAND" in
    start)     cmd_start ;;
    takeoff)   cmd_takeoff ;;
    land)      cmd_land ;;
    stop)      cmd_stop ;;
    navigate)  cmd_navigate "${ARGS[@]}" ;;
    rviz)      cmd_rviz ;;
    status)    cmd_status ;;
    *)
        echo "Unknown command: $COMMAND"
        echo "Usage: $0 [--sim|--real] <start|takeoff|land|stop|navigate <x> <y>|rviz|status>"
        exit 1
        ;;
esac
