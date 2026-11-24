# MAPPO 二维仿真

训练和验证支持 v9、v10、v11、v11.2。配置与任务的对应关系见[算法配置](../../config/algorithm/README.md)。

| 任务 | 入口 |
| --- | --- |
| 当前车辆位置到终点的导航建图 | `train_mappo.py`，使用 `mappo_training_v9.json` |
| 局部 A–B 区域侦察、固定 A–B 路线 | `train_mappo.py`，使用 v10、v11 或 v11.2 配置 |
| 独立复评检查点 | `rescore_checkpoints.py --run <实验目录>`，默认使用该次训练的验证种子与采样次数 |
| WildScenes 与 vtemp 滚动临时目标 | `validate_wildscenes_v9.py`、`validate_wildscenes_vtemp.py` |

## 训练与恢复

在安装了 `requirements-rl.txt` 的项目环境中，从仓库根目录运行：

```bash
python -I src/teamed-planning/scripts/simulator/train_mappo.py \
  --config src/teamed-planning/config/algorithm/mappo_training_v9.json \
  --output artifacts/mappo-v9-new --workers 8
```

Linux 和 PowerShell 的 `start_mappo_training` 脚本均默认使用 v9。后台训练的日志和 `launch.json` 保存在输出目录。

训练器使用同一份冻结策略采集整批轨迹，再做同步 PPO 更新。它将权重、优化器、随机数状态和配置写入 `latest.pt`，按验证结果更新 `best.pt`。在实验目录中放置 `STOP` 可请求保存并停止。移除 `STOP` 后，使用原配置、原目录和 `--resume` 恢复。冻结实验的独立回放继续使用自身的源码快照。

v9 使用的检查点架构名为 `gat_relay_attention_v8`，当前车辆位置路线奖励为 `uav_remaining_route_mapping_v9`。训练使用 `drone/` 内的图注意力、协作、通信及安全模块。

回归测试及冻结回放的隔离运行命令见[模拟器测试说明](README.md#regression-tests)。
