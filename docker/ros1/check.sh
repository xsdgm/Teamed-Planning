#!/usr/bin/env bash
set -eo pipefail
source /opt/teamed-docker/environment.sh
rosnode ping -c 1 /gcs_node >/dev/null
rosservice call /gazebo/get_world_properties >/dev/null
for id in 0 1 2; do
    timeout 3 rostopic echo -n 1 "/iris_${id}/mavros/state" | grep -q 'connected: True'
done
