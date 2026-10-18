# 当前 MAPPO 2D 算法

当前建图/物理环境对应 environment_version=4。1000 米地图的新训练批次为 `artifacts/mappo-1000m-20261018`；小地图第 1505 轮模型作为历史基线保留。旧版独立实验入口、动态置信度奖励、课程奖励调度、v3 专用比较脚本与旧回放生成器已删除。

## 保留的代码

- `../drone/mappo.py`：各无人机共享候选 Actor、集中 Critic、动作掩码与 PPO 模型。
- `mappo_experiment.py`：当前 2D 环境、证据送达账本、固定奖励、轨迹指标、PPO 更新；作为模块使用。
- `train_mappo.py` / `start_mappo_training.ps1`：唯一训练入口，支持检查点恢复。
- `validate_mappo_2d.py`：最佳模型与停止模型在新 2D 场景上的比较。
- `evaluate_large_world.py` / `large_world_scenarios.py`：冻结第 1505 轮模型的 1000 米地图测试。
- `navigation_metrics.py` / `scenarios.py`：导航图指标与可行场景生成。
- 建图、融合、无人车导航、链式通信、CBF 和回归测试为必要依赖。CADMM 与贪心比较基线保留。

## 当前参数与奖励

训练地图 120×120 米，三架无人机，高度 15 米、最大速度 6 米/秒；传感器 24×24 米、栅格 2 米，链路三维距离上限 95 米，CBF 安全层，移动无人车速度 1 米/秒。Actor 使用原 12 维特征。

固定团队奖励为 `路径代价改善/100 + 0.6×新送达观测面积/400`。路径改善在同一车辆位置比较建图前后代价；车辆到达本身没有奖励。原训练每回合最多 240 步，车辆到达会结束回合。验证按导航图连通、成图时间、路径效率和安全性选择模型。

1000 米测试保持传感器、通信与模型参数不变，采用随路线长度增加的预算，首次几何验证连通后再运行 30 秒；车辆到达不结束测试。该脚本是旧模型迁移测试。新的大地图训练使用下面的独立配置。

## 1000 米地图正式训练

配置：`../../config/algorithm/mappo_training_1000m.json`。从随机权重开始，2000 回合；固定障碍密度 22/80²（约 3438 个圆障碍），随机起点/方向，350、550、750 米任务混合，每张地图使用真值离线确认路线可行。参考路线只用于场景筛选和时间预算，不送入策略、奖励或车辆控制器。

预算为 `min(4000, max(600, ceil(3×参考路线长度+240)))` 秒。车辆到达不单独终止；任务图连续连通 30 步后结束，否则运行至预算。CBF 不可行时拒绝执行该动作，以 -10 的终止奖励记录这次安全失败。正常步奖励仍为上述两项。PPO 折扣 0.995、GAE λ=0.97。

保持原 12 维特征及 /100 尺度、传感器、链路和速度。训练种子 50000–57999，验证种子 60000/60001/60003 分别覆盖 350/550/750 米，保留测试种子 70000–70009。每 10 回合验证、每 25 回合保存快照，启动器默认使用大地图配置。首轮前不运行完整长回合验证；已先进行独立短程训练检查。

`progress.json` 记录已完成回合，`episode_progress.json` 每 25 步记录当前回合进度、已送达面积和最大链路距离。`training.jsonl` 区分 CBF 终止、预算耗尽和成图完成；车辆进展仅作诊断。`map_completed` 代表连续连通达到时长要求；`graph_task_connected` 代表回合末连通，不等同于连续完成指标。

## 运行

先激活安装了 `requirements-rl.txt` 的 Conda 环境，再从仓库根目录运行：

```powershell
python -I src/teamed-planning/scripts/simulator/train_mappo.py --config src/teamed-planning/config/algorithm/mappo_training_1000m.json --output artifacts/mappo-1000m-new
python -I src/teamed-planning/scripts/simulator/validate_mappo_2d.py --run artifacts/mappo-navigation-v4-20261017 --output artifacts/mappo-current-validation-new --seeds 32000 32001
python -I src/teamed-planning/scripts/simulator/evaluate_large_world.py --run artifacts/mappo-navigation-v4-20261017 --output artifacts/mappo-current-large-new
python -I -m unittest discover -s src/teamed-planning/scripts/simulator -p 'test_*.py'
```

以上是独立运行说明。当前大地图正式批次已经启动；重复运行应选新目录或明确恢复现有批次。输出目录必须为新目录；训练恢复使用 `--resume`。当前批次有用户停止标记 `STOP`，恢复前需要显式处理该标记。

模型保存在 `artifacts/mappo-navigation-v4-20261017/`：`best.pt` 是第 75 轮，`latest.pt` 是第 1505 轮。两者都保留。旧实验日志、结果、报告及 `artifacts` 中用于复现审计的原始源码快照保留；它们不属于当前运行代码。v4 检查点配置中的旧调度字段仅用于精确配置匹配，不再有对应课程调度实现。新验证会如实记录当前源码哈希及与训练快照的差异，不宣称源码完全一致。

## 当前研究限制

历史小地图模型在 1000 米长距离任务中出现停滞和 CBF 不可行。新的训练任务已覆盖需要随车前进的距离，但是否学会该行为必须由独立长距离验证与轨迹确认。车辆导航仍独立运行，可能停滞；不会用车辆到达奖励代替无人机导航图质量。
