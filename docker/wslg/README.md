# WSLg NVIDIA display

Gazebo and RViz open as Windows desktop windows through WSLg. The imported images retain the original forest, three Iris drones, and ground robot. The unused fourth drone is excluded by a temporary launch file.

This setup requires Docker Desktop with the WSL 2 backend, WSLg, and the Windows NVIDIA driver. Both containers use `/dev/dxg`, the WSL graphics libraries, and WSLg's X11 display. `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` selects the discrete GPU. The mounts `/mnt/host/wslg/.X11-unix` and `/usr/lib/wsl` were verified on this computer's Docker Desktop backend.

From the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-gazebo.ps1 start
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-gazebo.ps1 status
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-gazebo.ps1 stop
```

For the full planning stack, use `run-planning.ps1 start`, then `run-planning.ps1 rviz` with the same PowerShell options. See [planning commands](../planning/README.md).

Startup installs the small `mesa-utils` package once per container and checks that `glxinfo -B` reports `Accelerated: yes` and `D3D12 (NVIDIA ...)`. Startup fails if hardware rendering is unavailable. You can inspect `/tmp/wslg-renderer.log` in either container. Gazebo's actual renderer is also recorded in `/root/.gazebo/ogre.log`.

The previous Xvfb/noVNC desktop and port 6080 have been removed. Rendering uses the NVIDIA GPU; the semantic model still uses the image's CPU ONNX Runtime. Gazebo physics and ROS planning also use the CPU.

References: [Microsoft's WSLg container setup](https://github.com/microsoft/wslg/blob/main/samples/container/Containers.md), [GPU selection](https://github.com/microsoft/wslg/wiki/GPU-selection-in-WSLg).
