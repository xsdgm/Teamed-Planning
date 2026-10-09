# Windows Docker ROS 1 仿真

环境：Ubuntu 20.04、ROS Noetic、Gazebo 11、PX4 v1.13.2、MAVROS。
三个 Iris 无人机和一个差速地面车运行在同一容器内，ROS 网络无需跨 Windows 转发。
Conda 不参与容器环境。

在项目根目录的 PowerShell 中执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run-ros1.ps1 start
```

第一行只允许当前 PowerShell 窗口执行本项目脚本，不修改系统执行策略。

首次启动会下载基础镜像、构建 PX4；后续启动复用镜像。
浏览器打开 <http://localhost:6080/vnc.html?autoconnect=true&resize=scale> 查看 Gazebo 桌面。
使用软件渲染，不要求 NVIDIA GPU 或 Windows X Server。

```powershell
.\run-ros1.ps1 status
.\run-ros1.ps1 rviz       # 在浏览器桌面内打开 RViz；此终端保持运行
.\run-ros1.ps1 takeoff
.\run-ros1.ps1 navigate 10 5
.\run-ros1.ps1 land
.\run-ros1.ps1 stop
```

停止前先 land 并确认降落；stop 关闭整个仿真。

日志与容器终端：

```powershell
.\run-ros1.ps1 logs
.\run-ros1.ps1 shell
# 容器终端中：
source /opt/teamed-docker/environment.sh
rosnode list
rostopic list
```

仅主规划包挂载到容器，源码挂载为只读；catkin 的 build/devel 保存在 Docker 卷内。
改 Python/配置后重新启动容器；修改 docker/ros1 文件后重新执行 start 构建镜像。
现有 Linux deploy/main.sh 依赖原作者本地镜像，此 Windows 方案使用独立启动入口。

这个方案补建了原仓库未提交的仿真资源：基础障碍物场景、地面车、Iris 下视相机和 CPU LiDAR。
它不是原作者私有镜像中的完整森林场景。
语义模型默认禁用，沿用项目的零占据回退逻辑；完整语义感知需要另行下载 ONNX 模型、安装 onnxruntime，并在 entrypoint.sh 的规划启动命令中设置 seg_model_path 后重新构建镜像。
初次启动健康检查验证 Gazebo 服务、GCS 和三个 MAVROS 的 FCU 连接；起飞/导航通过单独命令触发。

验证记录：三个无人机成功进入 OFFBOARD 并在约 15 m 悬停，相机、LiDAR 和里程计均有数据。
导航命令可触发任务和拓扑图生成，地面车可运动；本次短程测试中因拓扑图稀疏在到达目标前停下。
完整目标到达和语义感知仍需继续调试项目的规划算法与模型，此配置提供可运行的 ROS/Gazebo 环境。
