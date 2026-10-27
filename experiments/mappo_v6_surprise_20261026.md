# MAPPO v6：任务地图惊讶度（2026-10-26）

直接在 `/home/bdi/team-planning-zql/Teamed-Planning` 修改；没有创建项目副本或 Git 工作树。v5.1 已有的未提交修改保留，旧配置默认关闭惊讶度。

## 奖励合同

`uav_mapping_route_quality_and_communication_v6_surprise`

PPO 使用 `r_train = r_task + r_surprise`。原有地图代价改善、成图 +100、路线质量、时间、CBF 干预及可行余量惩罚继续使用。v6 将前沿推进预算由 50 改为 45，给惊讶度预留最多 5 分/回合，因此二者未折扣累计上限仍为 50。这只是奖励尺度约束，不保证最优策略不变，也不保证不会冒险。

`r_surprise = min(beta * gate * clip(masked_MSE / error_scale, 0, 3), remaining_budget)`。

- beta 在本阶段内从 0.05 线性降到 0.005，每个同步批次固定。
- error_scale 是历史新证据预测均方误差的 EMA（权重 0.1，下限 1e-4）。不使用验证数据更新。
- gate = 新任务区域面积因子 `min(area/400,1)` × `(1-CBF_margin_risk)` × 必需链路可用 × 当前任务图未接通 × 非第 0 步。
- mask 只包含已知比例增加的空间格子；初始观测、未改变区域、padding、CBF 吸收失败步不支付惊讶度。
- 泛化面积奖励必须为 0，避免重复支付同类探索行为。

## 预测器与信息来源

`simulator/surprise_reward.py`：8×8 固定世界坐标任务区域栅格，每格已知比例与证据置信度，共 128 维。依据真正送达地面站的 ledger，采用固定任务走廊，不随车辆移动，不读取真实障碍地图或参考最短路。置信度使用现有 ledger 的证据置信度，不额外提供占据真值。

GRU(64) 根据历史地图特征和实际执行的水平速度统计（vx、vy 的机群均值和标准差，除以 6 m/s）预测下一步地图特征。输入使用 CBF 后实际执行动作；仿真提供三维 motion 时显式截取水平两维。预测头预测地图增量。它是训练端的共享团队奖励预测器，不增加 Actor 输入、部署通信字段或机载模型。它是对论文 NRM 思路的适配，不是完整 CORAL 复现。

## PPO、验证和恢复

同步采样阶段只携带观测转移数据。主进程收齐完整批次后，先用冻结的预测器、归一化尺度与 beta 计算一次奖励，再执行 GAE/PPO。预测器随后独立训练，每 32 步截断反传，每回合清空隐藏状态；不向策略反传预测误差。同一批 PPO 的奖励不变。停止时丢弃整批的现有规则不变。

检查点新增 `surprise`：模型、Adam 状态、误差尺度、已训练批次数。正常断点恢复须完整恢复；从旧策略初始化新阶段时可创建新预测器，从 v6 进入下一课程阶段时恢复已有预测器。策略权重格式不变，部署仅使用原策略模型。

`reward` / `uav_reward` / `extrinsic_reward` 记录原任务回报；`surprise_reward` 单独记录；`training_reward` 是 PPO 实际消费的奖励总和。另有 `surprise_error`、`surprise_scale`、`surprise_eligible_steps`、`surprise_weight` 和预测器损失。**验证、距离评估和选模保持任务评分，完全不加入惊讶度，也不训练预测器。**

## 使用

配置在 `src/teamed-planning/config/algorithm/`：

- `mappo_training_v6_surprise.json`：v6 基础配置。
- `mappo_v6_distance_curriculum.json`：引用 v6 配置的原五阶段距离课程。

在项目根目录运行第一阶段（正式训练尚未启动）：

```bash
/home/bdi/miniconda3/envs/teamedplanning-2d/bin/python -u \
  src/teamed-planning/scripts/simulator/train_distance_curriculum.py \
  --config src/teamed-planning/config/algorithm/mappo_v6_distance_curriculum.json \
  --output artifacts/mappo-v6-stage1 --workers 8 --through-stage 1
```

直接训练 `train_mappo.py` 时同样需要 v6 配置，限定 synchronous + velocity_mapping + 独立 PopArt Critic。不能把 v5.1 的原运行目录直接改配置 resume；新奖励应使用新训练产物目录。这里的 artifacts 是训练输出，不是另一个项目目录。

## 验证边界

测试覆盖新证据/安全门控、回合预算、CBF 失败不发奖、冻结评分、预测器可学习性、RNG 隔离、关闭新奖励时真实仿真采样兼容、配置约束、真实多进程训练及停止恢复一致性。

短程冒烟只验证数据流、数值、保存和验证协议；不能据此声称成图率或安全性改善。需要用相同种子、预算与多次随机初始化比较 v5.1 与 v6，并查看成图率、安全终止率、首次成图步数和路线质量，而非训练总回报。

实际执行结果：完整 simulator 回归共 183 项，成功结束（1 项按设计跳过），耗时 76.042 秒；git diff --check 通过。7 项 v6 专项测试包括真实仿真兼容和多进程停止/恢复与连续训练的逐位一致性。

最终干净冒烟输出：`artifacts/v6-surprise-smoke-20261026/verified/`。2 个 spawn 工作进程、4 回合×最多 60 步并完成固定场景验证；预测器训练 2 批，状态 completed。惊讶度累计分别为 0.039019、0.151181、0.119600、0.069491，任务奖励加惊讶度与实际 PPO 奖励和的误差小于 1e-6，实际 KL 均不超过配置上限。验证结果没有惊讶度字段。

正式 800 回合训练尚未启动；以上不是性能改善证据。项目父目录已重新核验，仅有 Teamed-Planning 一个项目目录。
