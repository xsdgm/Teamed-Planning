# MAPPO 正式训练：探索到关键区域的奖励课程

**历史记录：本批训练已安全停止于第 732 回合。**检查确认旧 direct 模式与全邻可见门控引入死锁；当前代码/配置已改为 chain/cbf，启动器默认使用独立新目录，但没有启动新正式训练。以下为旧批次设置记录，停止/恢复旧批次不能直接套用新配置；详见 [停机检查与 CBF 修复](mappo_chain_cbf_inspection.md)。

2026-10-15 启动配置：2000 回合，每回合 120 秒仿真，共 240000 个联合决策步（最多 720000 次单机选点），CPU 单线程。从随机参数重新训练，旧 40 回合模型和实验结果保留。

动态奖励按整个训练进度变化，每回合内固定基础权重；同一个训练回合不会因帧数增加而另行切换阶段，因此 Actor 不需要训练进度这个额外输入。

| 训练阶段 | 回合范围（约） | 关键区额外权重 β | 外部首次探索权重 η | 熵系数 |
| --- | --- | ---: | ---: | ---: |
| 探索 | 1–300 | 0.2 | 0.6 | 0.03 |
| 平滑过渡 | 301–1400 | 0.2 → 2.5 | 0.6 → 0.05 | 0.03 → 0.005 |
| 聚焦 | 1401–2000 | 2.5 | 0.05 | 0.005 |

过渡使用 smoothstep，没有阶段跳变。整体置信度增益权重始终为 1；关键区域仍为车附近、目标附近、车与目标之间走廊的并集。每回合 20 秒滑动窗口对 β/η 进行最多 ±15% 的反馈修正；重点正增益占比的反馈目标从前期 35%–50% 逐步升至后期 60%–75%。重点区域平均置信度达到 0.8 后，额外任务奖励逐步衰减。反馈不能覆盖训练课程，也不保证实际占比达标。

为了使探索有相应动作，20 个非悬停名额中预留 5 个给本机周围不同方向（半径 18m），只使用本机位置、已有 ID、时间和配置，不读取未知真实收益。MAPPO、贪心和统一接口版 C-ADMM 共同使用该候选配置；原版保持原候选规则，独立披露。该变化需要通过 `exploration_candidates=0` 的单独训练消融验证，尚未完成，因此正式训练结果不能直接与此前零探索配额实验混为同一条件。

奖励、传感器、地图、通信和静止任务源的基本限制沿用 [2D 实现说明](../src/teamed-planning/scripts/simulator/MAPPO.md)。本次仍为 2D 直接上传训练，不代表完整中继或真实无人车运动场景验收。

## 验证与保存

训练场景从种子 10000–17999 随机抽样；验证使用 20000–20004；最终测试预留 30000–30009。三者互不重叠。每 50 回合做确定性验证，训练期间不使用测试集挑选模型。

验证使用固定后期奖励设置并报告原始物理指标；选模评分为 `confidence_area + 2*task_gain - 500*spacing_violation - 10000*collisions - 200*out_of_bounds`，只用于验证选模，不加入训练奖励。变化中的训练 reward 不用于跨阶段选优。每回合原子保存 `latest.pt`；验证改善时保存 `best.pt`；每 100 回合另存快照。保存模型、Adam 状态、Torch 随机状态、场景采样随机状态、训练位置和完整配置。中断后从最近完整回合恢复；正在进行的回合重新采样执行。

输出目录为 `artifacts/mappo-formal-20261015/`：

- `progress.json`：当前状态、已完成回合和最新指标。
- `training.jsonl` / `validation.jsonl`：逐回合指标与验证结果。
- `latest.pt` / `best.pt` / `episode_XXXXX.pt`：续训或推理模型。
- `stdout.log` / `stderr.log`：后台日志。
- `sources/`、`provenance.json`、`config.json`：源文件快照、哈希、运行时版本及配置。

## 启动、停止和恢复

在仓库根目录使用本机已有环境启动隐藏后台进程：

```powershell
& ./src/teamed-planning/scripts/simulator/start_mappo_training.ps1
```

正常停止：在输出目录创建名为 `STOP` 的文件，程序在完成当前回合后停止；停止后先移除这个单文件，再加 `-Resume` 恢复。脚本不会自动删除 STOP，也不会自动启动另一个训练进程。

```powershell
New-Item -ItemType File artifacts/mappo-formal-20261015/STOP
# 停止后恢复：
Remove-Item -LiteralPath artifacts/mappo-formal-20261015/STOP
& ./src/teamed-planning/scripts/simulator/start_mappo_training.ps1 -Resume
```

在其他 Python 环境中可直接运行：

```powershell
python src/teamed-planning/scripts/simulator/train_mappo.py --config src/teamed-planning/config/algorithm/mappo_training.json --output artifacts/my-training
```

最终测试由明确的评估命令单独运行，评估器自动从检查点读取探索候选配额。下例应在正式训练结束后执行：

```powershell
python src/teamed-planning/scripts/simulator/mappo_experiment.py --episodes 0 --steps 120 --checkpoint artifacts/mappo-formal-20261015/best.pt --test-seeds 30000 30001 30002 30003 30004 30005 30006 30007 30008 30009 --output artifacts/mappo-formal-test
```

训练以固定 2000 回合结束；不预先保证超过 C-ADMM。当前只启动一个训练随机种子，尚未执行动态奖励与固定奖励的多种子消融；后续结论需要这些实验支持。
