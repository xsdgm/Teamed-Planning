# Docker files

Docker image backups and the current simulation startup scripts live in this directory.

| File or directory | Purpose |
| --- | --- |
| `images.txt` | Image names and original configuration hashes |
| `teamedplanning-docker-images-2026-10-10.tar.gz` | Exported XTDrone and planning images |
| `wslg/` | WSLg display, NVIDIA renderer verification, and Gazebo startup |
| `planning/` | Windows ROS planning startup and operator commands |

Run Windows Gazebo from the repository root with `run-gazebo.ps1`; its Compose file is `compose.xtdrone.yaml`.
Run the complete simulation and planning stack with `run-planning.ps1`; see `planning/README.md` for takeoff and navigation commands.

The archive is an offline backup of both runtime images. `run-planning.ps1 start` imports it automatically if either image is missing. For manual import, run `docker load --input docker/teamedplanning-docker-images-2026-10-10.tar.gz` from the repository root.

The old Linux/X11 launch scripts and their shell/tmux configuration have been removed. The current launchers use `compose.xtdrone.yaml`.
