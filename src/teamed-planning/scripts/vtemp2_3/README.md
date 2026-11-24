# World model

VTemp 车机导航及 VTemp2 训练见 [模拟器说明](../simulator/VTemp2.md)。

需要图世界模型时，使用 [VTemp3 训练入口](train_vtemp3.py)。原入口 `train_hierarchical.py` 也会转到同一训练器。

| 要做的事 | 从这里开始 |
| --- | --- |
| 训练图世界模型与强化学习策略 | 从 `train_vtemp3.py` 启动，采样和动力学实现在 `navigation/graph_rollouts.py` 和 `navigation/graph_dynamics.py` |
| 修改无人车控制、规划与安全约束 | `navigation/hierarchical_runtime.py`、`navigation/planning_graph.py`、`navigation/hierarchical_safety.py` |
| 准备场景、检查训练集和验证集 | [navigation/](navigation/) 下的 `scene_manifest.py`、`wildscenes_tasks.py`、`wildscenes_runtime.py` |
| 复查已训练策略或恢复检查点 | `evaluate_fixed_target.py`、`evaluate_safety_repairs.py`、`hierarchical_sources.py` |

VTemp 线的层级执行、安全、场景、慢层 PPO 和 VTemp3 图世界模型实现都在 [navigation/](navigation/)。原 `vtmp` 线（`rssm/`、`terrain/`、`survey/`）代码已移除，算法流程与失败复盘见 [vtmp_postmortem.md](../../../../experiments/world_model/vtmp_postmortem.md)。

## v9 随机地图训练与 WildScenes 验证

VTemp3 支持 `--validation-split synthetic_to_wildscenes`。训练清单绑定每张 v9 随机圆障碍地图的几何文件和校验和。验证清单绑定 WildScenes 原始栅格，并用 `source_unknown_space=blocked` 限制无人车的地面通行。`--ground-start-hold-seconds 30` 在两个阶段都保留前 30 秒的收图与规划。`--validation-distances` 单独指定验证任务的距离范围。

训练快层达成局部临时目标时，将 `--fresh-fast-world --fixed-mission-target` 与仅含 fast 更新的训练计划一起使用。本实验用 `--warmup-distances 100 400 --mission-distances 100 400`，在每个训练场景从起点 100–400 米的范围内随机指定一个临时目标。本局只需要到达这一个点。每架无人机的快层从第一帧到最后一帧都直接接收它的坐标。无人车收到同一坐标，按本地感知和规划行驶。地面车位置、周围机器人状态和地图仍通过原有传感与通信路径获取。世界模型用真实轨迹监督未来 4 秒的状态和动作预测。快层 PPO 复用 v9 的车端剩余路线测绘奖励。车端收到的地图能接通或缩短通往目标的路线、推进已探索边界时会得分，步耗和安全控制介入会扣分。到达目标的最后一步加 100，CBF 无可行动作时该步记 −100，普通超时保留最后一步的环境奖励。WildScenes 的地面安全违规按安全失败处理。慢层和几何候选选点都不参与。`--ground-start-hold-seconds 30` 让无人机从首帧就能行动，无人车在仿真时间达到 30 秒后才能移动。

该模式的训练和验证记录均使用 `fixed_target`。WildScenes 每条预选路线只下发自身终点坐标，验证距离由该清单和 `--validation-distances` 决定。单独使用 `--fresh-fast-world` 会沿用几何选点。

VTemp3 启动时，会把本次使用的导航模块平铺到实验目录的 `runtime_sources/world_model/`。例如，工作区的 `navigation/graph_dynamics.py` 在该快照中是 `graph_dynamics.py`。恢复时使用原实验目录和 `--resume latest`，训练器会加载该次实验的源码。更多入口和运行参数见 [scripts 说明](../README.md)。
