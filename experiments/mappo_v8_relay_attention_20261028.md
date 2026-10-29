# v8：中继状态证据的时效/跳数注意力（代码已接入；三阶段课程训练进行中）

状态：2026-10-28 实现并通过测试；**尚未启动训练，没有任何效果结论。** 方案见 [v8 设计](mappo_v8_temporal_attention_design.md)，基础是 [v7](mappo_v7_cooperation_20261027.md)（v7 代码仍未提交）。“跳频”按设计文档理解为转发跳数 hop count。

## 实现边界

对应设计 §5/§7 的**第一组实验**：一跳相对几何、RNG 骨架、双端 CBF、地图包和原任务广播都保持 v7 的同步契约，v7 的连通性论证不变。异步的只有**状态证据**：各无人机自身状态（位置/速度）和车辆状态（位置/速度/当前目标）。它们经逐跳中继缓存传播，每条都带原始测量时间和跳数。

未做：地图证据异步、一跳测量延迟/丢失（需先完成 §5 的 make-before-break 安全协议）、带宽代价奖励、GRU、ROS 消息与节点。训练仍只在 2D 仿真中进行。

## 组成

### 1. 中继缓存协议 [relay_cache.py](../src/teamed-planning/scripts/drone/relay_cache.py)

可部署逻辑，不含传输。每条条目：origin_id/boot_id/seq、source_stamp、sender_id、receive_stamp、hop_count、payload_type、frame_id、state、sigma、mission_id/goal_epoch、hop_limit、expires_at。

| 规则 | 实现 |
|---|---|
| 源时间 | 转发时 source_stamp 与 expires_at 不变；只把 hop_count+1，并替换 sender_id |
| 去重 | 同一 (origin, boot, seq) 只算一次；更短路径只更新传输元数据，不产生新的测量时间 |
| 顺序 | 同一 boot 中旧 seq 不覆盖新 seq；重启（新 boot、seq 归零）只在源时间更新时接受 |
| 时钟 | 年龄 = 本地时间 − (源时间 − 时钟偏差估计)；源时间超前超过 0.5 s 的条目隔离，不截断成“最新”；时钟偏差界按速度上限换算进 sigma |
| 未知源时间 | 标记 time_unknown，年龄取接收年龄 + 跳数 × 2 s 的单跳延迟上界 |
| 过期/容量 | 按 TTL 删除（无人机 30 s、车辆 60 s）；缓存上限 64，先删过期项，再删最旧项，车辆记录最后删 |
| 任务版本 | 车辆 goal_epoch 不能回退；frame_id 或 mission_id 不符的条目拒收 |
| 转发 | 每包最多 capacity 条：本机状态 → 车辆 → 上次未发的 → 更新的；hop_count 达到 hop_limit（16）后不再转发 |

### 2. 仿真链路层 [relay_channel.py](../src/teamed-planning/scripts/simulator/relay_channel.py)

每条有向物理链路一个事件队列。只发给发送时刻实际在通信半径内的无人机；最早在下一个决策步到达，因此同一步内不会递归传遍全网。每次转发都是新事件。丢包、延迟抖动（会造成乱序）、发送周期都由独立随机流抽样，不改变其他仿真随机数。车辆只发布自己的 `vehicle_state`（把任务广播包装成证据），不转发。

### 3. 控制器接入

[drone_main.py](../src/teamed-planning/scripts/drone/drone_main.py) 只从 `observation['relay_envelopes']` 读取已经到达的包，不读邻机 broadcast 字典。读完后生成本机状态，把决策时刻的缓存快照写入 `nav_obs['relay']`。[mappo_experiment.py](../src/teamed-planning/scripts/simulator/mappo_experiment.py) 负责投递和发送。PPO 存的是动作产生时的特征，训练时不会用之后补齐的缓存重建旧观测。

### 4. 网络 `gat_relay_attention_v8`（[relay_attention_mappo.py](../src/teamed-planning/scripts/drone/relay_attention_mappo.py)）

v7 编码（地图 GAT + 一跳邻机注意力，418 维）之外，新增 16 个中继 token，每个 16 维：相对源位置、源速度及其是否已知、年龄（s）、跳数、类型（无人机/车辆）、sigma、源时间未知、该来源当前是否在一跳内被直接测到、车辆目标（相对位置及是否已知）。来源 ID 只用于判断“是否被直接测到”，不作为 embedding。截断时车辆记录优先，然后按年龄、跳数排序，不按距离删除。

查询向量来自自身状态/传感器/地图编码。2 个头，每头 16 维；无人机和车辆分别有自己的 key/value 投影：

\[
\ell_{ie}=\frac{q_i^\top k_e}{\sqrt d}-\operatorname{softplus}(a_{h,\rm type})\frac{\Delta t_e}{\tau_{\rm type}}-\operatorname{softplus}(b_{h,\rm type})\log(1+h_e)
\]

z_e 包含 TGAT 式可学习时间编码 cos(ωΔt+φ) 和 log(1+h)。先做 hard mask，再在合法条目内 softmax 得到 α。之后乘**固定**的绝对可靠性门控，且不再重新归一化：

\[
\rho_e=\mathbf 1_{\rm valid}\exp[-\Delta t_e/\tau_{\rm type}-0.1\,(h_e-1)_+-(\sigma_e/95)^2],\quad c_i=\sum_e\alpha_{ie}\rho_e v_e
\]

τ 取无人机 10 s、车辆 30 s，都是设计值，未经实机验证。另外输出 4 个统计量：有效条目数、平均 ρ、最大 ρ、缓存为空标记。缓存为空时上下文为 0，不会产生 NaN。

`relay_attention` 对应设计 §8 的消融：`full`=v8-D，`age`=v8-A（只用年龄），`hop`=v8-B（只用跳数），`softmax`=v8-C（logit 中有年龄和跳数，但没有绝对门控）。模式保存在检查点 buffer 中。

宽度：原始输入 1826（v7 1554 + 16×17），编码后 454（418 + 32 + 4），参数 205,117。奖励与 v7 完全相同，reward_contract 沿用 v7 标签。feature_version = 8。

## 配置

[mappo_training_v8.json](../src/teamed-planning/config/algorithm/mappo_training_v8.json) = v7 配置，加上：`policy_architecture: gat_relay_attention_v8`、`relay_loss 0.1`、`relay_delay_jitter 2`（每跳延迟 1–3 步）、`relay_period 1`、`relay_capacity 16`、`relay_attention full`。这些通信参数是初始猜测，可以改。在 v7 或更早的架构上设置任何 relay_* 字段都会报错。

```bash
cd /home/bdi/team-planning-zql/Teamed-Planning
OPENBLAS_NUM_THREADS=1 /home/bdi/miniconda3/envs/teamedplanning-2d/bin/python -E \
  src/teamed-planning/scripts/simulator/train_mappo.py \
  --config src/teamed-planning/config/algorithm/mappo_training_v8.json \
  --output artifacts/mappo-v8-stage1 --workers 8
```

## 退回 v7 的路径

1. **配置级**：v8 全部是可选新增。使用 `mappo_training_v7.json` 时，代码路径与 v7 训练启动时完全一致。已用 v7 训练目录中的源码快照 `artifacts/mappo-v7-stage1-20261027-171143/sources` 和当前代码分别运行同样的 v7 回合并逐位比较：v7 best.pt 和新初始化的 v7 模型，seed 50003/60001，各 120 步。特征、动作、log prob、奖励、critic 状态和回合 summary 完全相同，只有计时字段 `decision_ms` 不同。
2. **检查点**：v7 的 best.pt 仍加载为 `CooperativeActorCritic`；v8 检查点由 `gat_actor.0.relay.mode` 识别，feature_version 为 8。两者不能互相 resume，也不能互作 init_checkpoint。
3. **代码级**：[mappo_v8_relay_attention_from_v7.patch](mappo_v8_relay_attention_from_v7.patch) 是 v7 快照到 v8 的完整代码差异，含 4 个新文件、6 个修改文件和 v8 配置。在仓库根目录执行 `patch -R -p1 < experiments/mappo_v8_relay_attention_from_v7.patch` 即可退回。已在副本上验证：反向应用后，所有 .py 都与 v7 快照逐字节相同，v8 新文件和配置都被删除。v7 已于 2026-10-28 单独提交为 `9b39b0c`（内容取自 v7 训练源码快照，不含任何 v8 文件），git 可直接退回；截至本记录尚未推送到远端。

## 验证

- [专项测试](../src/teamed-planning/scripts/simulator/test_relay_attention_v8.py) 15 项全部通过（7 s）。覆盖：三跳源时间不变、跳数逐跳 +1；旧 seq 不覆盖新 seq；同源重复证据只计一次；短路径只更新元数据；设备重启；时钟超前隔离；未知源时间；转发不刷新过期时间；goal_epoch 不回退；链路因果性/丢包率/抖动乱序/发送周期；空缓存输出为零；单个旧条目时 α=1，只靠门控缩小，门控上下文 = ρ × softmax 上下文；全部陈旧时 ρ 仍然小；token 排列与填充无关；改名不变；没有缓存时显式报错；四种模式检查点往返；v7 检查点照常加载；真实三机 Experiment 中，远端无人机经 2 跳收到车辆和队友状态（hop=2）；理想链路退化为每跳 1 步（age=hop）；配置契约；真实短训练、验证、检查点与 resume。
- 完整 simulator 回归：222 项，1 项按设计跳过，其余全部通过（110 s）。
- 开销：150 步回合 175.8 vs 158.1 ms/步（+11%），按 v7 的 78 s/回合估算约 87 s/回合。
- 冒烟（未训练策略）：每机约 3 个 token，车辆已知 98%，平均年龄 2.1 s，丢包约 10%，按真值计算的状态位置误差约 3.3 m（仅评估用，不进入 Actor）。

## 已知局限与下一步

- **3 架无人机、100–284 m 阶段中，大部分时间队友都在一跳内**（冒烟回合中远端比例为 0；但正式训练首批 16 回合中，队友状态约 20% 来自一跳以外，平均最大跳数 2.2，见下方训练记录）。车辆通常也已通过原任务广播 state[4:11] 获知。因此该设置下 v8 与 v7 的差别可能很小。要检验跳数注意力，需要更长的链或更多无人机（fleet_sizes、更远阶段），或让一跳之外的信息变成必需。
- 新增 per-episode 指标：relay_token_count、relay_vehicle_known、relay_remote_fraction、relay_age_s、relay_max_hop、relay_position_error_m，以及 relay_packets_*、relay_entries_*。
- 门控参数固定、不学习；logit 惩罚是学出来的。“固定衰减/均值聚合”这一对照组尚未实现。
- 旧的 `validate_mappo_2d.py`、`evaluate_large_world.py` 没有传 relay 参数（v7 也没有接入这两个脚本）。v8 请用 train_mappo 的验证或 train_distance_curriculum。
- 设计 §8 的实验矩阵（v7 / v8-A/B/C/D、多随机种子、报告区间）尚未运行。

## 正式训练：第 1–3 阶段距离课程（2026-10-28 启动）

经用户指示，于 **2026-10-28 13:21（北京时间）** 启动，从头训练，阶段依次衔接：

| 阶段 | 任务距离 | 回合 | 初始化 |
|---|---|---|---|
| 1 | 100–283.8 m（三机直链理论半径以内） | 800 | 随机权重 |
| 2 | 283.8–400 m（超出静止车辆时的直链覆盖，需要车辆移动） | 800 | 阶段 1 末权重 + Adam 状态 |
| 3 | 400–500 m | 800 | 阶段 2 末权重 + Adam 状态 |

- 课程配置：[mappo_v8_distance_curriculum.json](../src/teamed-planning/config/algorithm/mappo_v8_distance_curriculum.json)。基础为 `mappo_training_v8.json`，环带和评估种子与 v6.2-2 相同（61000–61005，各 4 次采样）。每阶段结束后，用阶段末权重在全部 5 个环带（含未训练的 500–600、600–750 m）上做距离评估，结果在各阶段的 `distance_summary.json` 和顶层 `distance_matrix.json`。
- 目录：`artifacts/mappo-v8-curriculum-s1to3-20261028-132114/`。调度器 PID 3089008，首个训练器 PID 3089009（后续阶段的 PID 见 `progress.json`）。使用 16 个采样进程：每批 16 回合恰好一轮并行；采样种子只依赖回合序号，因此结果与 8 进程相同，只是更快。命令：

```bash
OPENBLAS_NUM_THREADS=1 setsid nohup /home/bdi/miniconda3/envs/teamedplanning-2d/bin/python -E \
  src/teamed-planning/scripts/simulator/train_distance_curriculum.py \
  --config src/teamed-planning/config/algorithm/mappo_v8_distance_curriculum.json \
  --output artifacts/mappo-v8-curriculum-s1to3-20261028-132114 --workers 16 --through-stage 3
```

- 停止：`touch <目录>/STOP`；继续：同一命令加 `--resume`。
- **训练期间不要修改主目录源码**：阶段 2、3 会从主目录重新启动 train_mappo.py，届时读取的是当时的代码。每个阶段的 `sources/` 会自动保存实际使用的源码。
- 启动前用极小规模（每阶段 2 回合 × 5 步）在临时目录完整跑通三阶段：权重和 Adam 状态在阶段间正确传递（阶段 2 的 init_checkpoint sha 与阶段 1 末权重一致），每阶段的 5 环带评估都已生成。
- 首批（16 回合，677 s）：KL 7.7e-4，无拒绝，熵 1.838，EV_pre≈0（预期）；成图 0%，安全终止 25%；中继平均年龄 2.7 s，平均最大跳数 2.2，队友状态 20% 来自一跳以外，丢包 10%（与配置一致）。
- 粗略耗时：阶段 1 约 9–10 h；阶段 2、3 的回合预算更长，各约 1.5–2 倍；另加 3 次距离评估。预计全部约 2 天，以实际进度为准。
