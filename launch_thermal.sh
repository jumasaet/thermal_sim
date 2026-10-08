#!/bin/bash
# Lanza el simulador pseudo-térmico con ROS2.
# NO hacer source de /opt/ros/humble/setup.bash en este terminal.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LD_LIBRARY_PATH=/home/jumasaet/isaac_sim/exts/isaacsim.ros2.core/humble/lib
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-15}

echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "Lanzando pseudo-thermal con ROS2..."

~/isaac_sim/python.sh "$SCRIPT_DIR/run_pseudo_thermal.py" \
    --autoplay \
    --auto-track circular \
    --ros2 \
    "$@"
