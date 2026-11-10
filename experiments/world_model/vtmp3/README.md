# vtmp3：A—B 任务选区、区域建图与车辆导航

vtmp3 的高层输出是两个世界坐标点 A、B。无人机后端负责探测两点之间的区域并向车辆送达地图；车辆根据实际收到的信息和最终目标 G 在导航图上规划前进，然后重新选区。车辆当前位置 C 独立于 A、B，允许但不强制 A=C。

本版本先使用可替换的随机建图后端训练高层世界模型，不加载 v11-2 权重。当前正在训练的 v11-2 不受此实现修改。后端已经模拟建图耗时、部分探测失败、通信失败和超时。它不执行真实无人机动力学或 CBF，因此其训练与测试成绩都必须标为 **proxy backend**，不能作为 v11-2 的真实效果。

## 第一阶段的执行循环

1. 车辆的局部感知和已送达证据形成观测。未知区域保持未知。
2. 在通信展开的几何范围内生成候选 A、B，世界模型预测每项任务造成的地图变化、车辆移动、信息增量、等待时间和失败事件。
3. 模型规划器选择预测效用最高的候选，发出 A—B 区域任务。
4. 车辆停留在 C，仿真时间推进到建图结果返回。失败可以返回部分地图，通信失败返回空地图。
5. 只把真正返回的区域证据合入地图。Dijkstra 在保守的已知自由栅格导航图上寻找到最终目标或最接近最终目标的可达节点的路径，车辆以 1 m/s 最多行驶 30 秒。
6. 记录完整转移，回到第 1 步，直到到达最终目标或耗尽时间/任务数预算。

模型的公共 RSSM 实现在 `world_model/rssm.py`，单目标版本 `model.py` 与两点任务版本 `ab_model.py` 继承同一个基础类；动作是 `[ax, ay, bx, by]/1000`。选点通过模型评分实现，不把 bootstrap 启发式采集器称为学习策略，也不声称模型直接回归坐标。当前规划器使用 **一个宏任务的预测视野**；训练包含默认四段自由推演损失，验证另外报告 1/2/4 段误差。多段预测训练不等于已经实现长期最优选区。

## 与 v11-2 对齐的接口

任务消息沿用 v11-2：

```json
{"survey": {"a": [480.0, 500.0], "b": [600.0, 500.0], "half_width": 12.0}}
```

`ProxyMapper.survey(task_id, task, vehicle_position)` 返回 `MappingResult`：

| 字段 | 含义 |
| --- | --- |
| `task_id` | 任务标识，用于重复回传去重 |
| `cell_ids`, `occupied` | 实际送达的世界坐标网格证据；未知区域不返回 |
| `elapsed_seconds` | 本次实际等待时间 |
| `status` | `complete`、`partial_failure`、`communication_failure` 或 `timeout` |
| `backend` | 回传来源和后端版本 |

`complete` 仅表示区域探测回传完成；如果障碍阻断 A—B，导航图仍然可能不连通。环境不会删除障碍、返回参考最短路径或补造图边。仿真真值只允许用于初始化场景、指定区域的传感模拟及执行后的物理净空审计。

配置读取 `mappo_training_v11_2.json`：3 架无人机、20–200 m 的 A—B 距离范围、12 m 区域半宽、2 m 地图分辨率、1 m 车辆净空。候选两端都在车辆 200 m 内；通信检查只描述可以重新展开的几何包络，不证明某个真实机群状态可执行。最终导航任务独立采用 100–400 m 起终点距离。

真实 v11-2 接入时需要一个适配器：发送任务，推进无人机与通信环境，提取**车辆确实收到的**地图证据，返回相同结果结构。不能把无人机内部尚未送达的地图当作回传结果。当前没有实现该真实策略适配器。

## 延迟与失败假设

以下为可配置的预训练假设，尚未用 v11-2 实测数据校准：

- 等待时间：`(8秒 + (A—B长度 + 车辆到较近端点的距离)/2 m/s) × U(0.5, 1.5)`。
- 探测失败概率：`0.15 + 0.15 × (A—B长度/200 m)²`。
- 独立通信失败概率：0.05；此时没有新地图送达。
- 超时上限：180 秒。
- 探测失败或超时仅返回从 A 起算约 20%–75% 长度的连续探测带；已返回网格的障碍信息仍为理想准确值。
- 车辆在建图阶段保持位置；等待时间计入总任务时间及效用成本。

这些假设支持先学习等待、失败和部分信息下的决策。它们没有模拟实际机群布局、置信度误差、地形语义、链路状态的持续演化，或无人机随车辆重新部署的全过程。后续应依据 v11-2 的任务长度、地形、初始机位和通信状态校准后端，并收集真实执行转移继续训练。

## 数据、训练及验证

每个 episode 保存观测地图、车辆/目标状态、四维 A—B 动作、信息增量/推进/时长/失败/到达结果，以及终止或预算截断标志。真值场景单独归档，不进入训练样本。训练与验证按场景种子和几何哈希隔离，测试场景不参与选模。

`--rounds 1`：用仅访问已知信息的 bootstrap 策略采集，训练世界模型，再在独立场景中执行真正由模型选择的 A—B 任务。

`--rounds 2`：在上一轮模型选点基础上，以 20% 随机任务探索收集新的训练场景，加入已有数据后继续训练。第二轮重新建立优化器，模型参数继承上一轮最佳权重；这不是恢复同一次优化器状态。

每轮保存 `latest.pt`、按验证损失选择的 `best.pt`、优化器状态、参数变化量、损失、预测基线对比和源码/数据哈希。每个运行目录冻结 Python 源码及 v11-2 配置。默认数据采集为单进程，PyTorch 使用两个 CPU 线程。

从仓库根目录运行小规模预检：

```bash
/home/bdi/miniconda3/envs/teamedplanning-2d/bin/python -I -B \
  src/teamed-planning/scripts/simulator/train_world_model.py --algorithm vtmp3 \
  --output artifacts/vtmp3-proxy-preflight-NEW \
  --train-scenes 2 --val-scenes 1 --test-scenes 1 \
  --max-steps 6 --max-seconds 700 --epochs 5 --rounds 1 --device cpu
```

扩大到初轮训练（输出目录必须尚不存在；GPU编号按资源安排）：

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /home/bdi/miniconda3/envs/teamedplanning-2d/bin/python -I -B \
  src/teamed-planning/scripts/simulator/train_world_model.py --algorithm vtmp3 \
  --output artifacts/vtmp3-proxy-training-NEW \
  --train-scenes 32 --val-scenes 8 --test-scenes 4 \
  --max-steps 32 --max-seconds 2400 --epochs 100 --rounds 2 --device cuda
```

实际进度见运行目录 `status.json`；每轮进度见 `training_*/status.json` 和 `metrics.jsonl`；逐任务选点、延迟、回传和预测效用见 `data/*.decisions.json` 及 `test_*_model.json`。独立测试同时运行 bootstrap 对照，仅用于描述性比较。

专项测试位于 `src/teamed-planning/scripts/tests/world_model/vtmp3/`：

```bash
cd src/teamed-planning/scripts
/home/bdi/miniconda3/envs/teamedplanning-2d/bin/python -E -s -B -m unittest discover -s tests/world_model/vtmp3 -v
```

当前车辆执行器是保守导航图上的有界运动学路径跟随器，尚未复用原差速车辆局部控制器。该简化和理想网格证据都需要在接入真实 v11-2 时做转移验证。

## 2026-11-10 运行记录

13 项专项测试通过，包含未探测地图隔离、未知区域禁止行驶、完整探测仍可能被障碍阻断、部分/通信失败、超时、去重、真实参数更新和权重重载。

预检位于 [vtmp3-proxy-preflight-20261110](../../../artifacts/vtmp3-proxy-preflight-20261110/)：2 个训练场景、1 个验证场景，5 次真实优化；另在 1 个测试场景执行模型选点和 bootstrap 对照。模型闭环 5 次任务后距终点约 65.69 m，因时间预算截断，未声称到达或优于对照。预检只验证流程。

初轮代理环境训练已启动：[vtmp3-proxy-training-20261110](../../../artifacts/vtmp3-proxy-training-20261110/)。32 个初始训练场景（220000–220031），8 个验证场景（320000–320007），第二轮增加 32 个模型采集场景（230000–230031），每轮训练 100 epochs；4 个独立测试场景（420000–420003）。每场景最多 32 次任务，在宏任务边界检查 2400 秒预算，最后一项任务可能跨过该时间阈值。采集单进程、低 CPU 优先级；优化阶段使用物理 GPU 2。启动进程 PID 3418853，实际阶段和子进程见 [status.json](../../../artifacts/vtmp3-proxy-training-20261110/status.json)，日志见 [运行日志](../../../artifacts/vtmp3-proxy-training-20261110.log)。此段记录启动事实，不表示两轮训练已经完成。
