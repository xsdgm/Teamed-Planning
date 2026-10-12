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

Gazebo and RViz appear as Windows desktop windows through [WSLg with NVIDIA rendering](../wslg/README.md). The semantic model continues to use CPU ONNX Runtime. These commands operate simulation containers only.
