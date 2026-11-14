# Windows simulation and planning

The planning container shares the Gazebo container's network namespace, so ROS master, MAVROS, planning nodes, and RViz can communicate through localhost. Project source is mounted read-only; build/devel output is stored in Docker volumes.

The semantic model must exist at `src/teamed-planning/scripts/drone/models/segformer_b0_ade/model.onnx`. It was downloaded and exported for this setup. CPU inference uses two threads per drone.

From the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-planning.ps1 start
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-planning.ps1 takeoff
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-planning.ps1 navigate 10 5
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-planning.ps1 status
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-planning.ps1 rviz
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-planning.ps1 land
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-planning.ps1 stop
```

`start` restores missing images from the archive, verifies NVIDIA OpenGL rendering, starts Gazebo, compiles the planning package, waits for MAVROS and odometry, and starts the drone, segmentation, ground, and GCS nodes. `takeoff` and `navigate` are explicit commands. Wait until all three drones are hovering before starting navigation. `land` requests landing; `stop` stops both containers.

Gazebo and RViz appear as Windows desktop windows through [WSLg with NVIDIA rendering](../wslg/README.md). Compose exposes the WSL GPU through `/dev/dxg` and the WSL graphics libraries. The semantic model uses CPU ONNX Runtime. These commands operate simulation containers only.

## v9 + VTemp rolling200 in Gazebo

Start this mode with the original v9 episode-576 `best.pt` from
`artifacts/mappo-v9-mixed100to400-no-subgoal-800ep-20261105-153146/`.
The start command verifies the checkpoint hash and builds a separate planning
image with CPU PyTorch. Use the existing SegFormer ONNX model.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-planning.ps1 start -VTempCheckpoint 'C:\path\to\best.pt'
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-planning.ps1 takeoff
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-planning.ps1 status
# Once all three drones are hovering, try a goal beyond the rolling radius.
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-planning.ps1 navigate 350 0
docker exec teamedplanning-ad bash -c 'source /opt/ros/noetic/setup.bash && rostopic echo /drone_0/v9_planned_velocity'
```

The Gazebo launch extends the bundled forest along X and spawns the three
aircraft 18 m apart. This mode uses the camera-derived 12×12 grid at 2 m per
cell and a 95 m three-dimensional radio range with ROS relay envelopes.
The existing flight controller holds its 15 m takeoff altitude. The ground robot broadcasts its current
200 m rolling waypoint, reselects within 25 m, and stops within 2 m of the
final destination. Each v9 planner publishes a stamped map-frame `vx, vy`
in metres per second on `/drone_N/v9_planned_velocity`. The planning output
is independent of MAVROS setpoints. The existing flight state machine handles
takeoff and hover. The 350 m command can be used to observe planning output.
