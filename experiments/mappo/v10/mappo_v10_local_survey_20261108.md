新 v10：独立 a—b 区域探测与无人车通信约束

本版实现用户确认的任务：输入探测区域两端 a、b，提高无人机本机融合地图在该区域内的信息置信度，同时保持无人机网络与真实车辆当前位置 c 连通。版本名称为 v10，策略架构为 `gat_survey_v10`，reward contract 为 `uav_local_ab_confidence_with_vehicle_backbone_v10`，模型输入 `feature_version=11`，checkpoint 标记为 `survey_schema=1`。这些标记共同识别本版；特征编号 10 已用于 CoGraph v2。

实现已接入，正式微调于 2026-11-08 17:22（香港时间）启动，PID 2386975，8 个采样进程。运行目录为 [mappo-v10-local-survey-ab-800ep-20261108-172206](../../../artifacts/mappo-v10-local-survey-ab-800ep-20261108-172206/)，实时阶段与完成回合数见 [progress.json](../../../artifacts/mappo-v10-local-survey-ab-800ep-20261108-172206/progress.json)。训练先评估 24 条初始验证轨迹，再开始同步 PPO 更新。本文不表示训练已经完成，也不声称相对 v9 的效果改善。

启动前 94 项专项及相关回归测试通过；另外完成了 4 回合、2 次真实 PPO 更新、2 个进程采样及中断恢复验证，见 [preflight result](../../../artifacts/v10-local-survey-preflight-20261108-172022/result.json)。正式训练从冻结的 `runtime_sources` 启动，逐文件哈希、源权重 SHA256、环境变量与完整命令保存在运行目录，后续工作区修改不会改变本次训练。

**a、b 和 c 的含义。** a、b 在一局中固定，定义带圆形端帽的线段走廊；默认半宽 12 m、网格分辨率 2 m。c 是车辆实际位置，随车辆运动更新。初始 a=c 是允许的任务分布之一，但 a 不会自动跟随 c；即使几何端点一度相同，本版的信息来源、reward 和完成条件仍与 v9 不同。

车辆广播的 `position` 保持真实 c。独立任务放在以下字段，经实际接收到的 `ground_missions` 逐跳转发：

```json
{"survey": {"a": [480.0, 500.0], "b": [600.0, 500.0], "half_width": 12.0}}
```

Actor 保留 v9 的目标/车辆相对位置、机载图、邻机与中继编码，追加相对 a、相对 b、b−a、半宽和有效标记，共 8 个输入。任务必须来自本机实际接收的任务记录，检查原始 `source_time` 的 10 s 有效期；没有有效任务时新增特征为零。不能把广播中的真实车辆位置替换为 a，也不把集中 critic 的状态作为 actor 输入。[模型与任务编码](../../../src/teamed-planning/scripts/drone/survey_mappo.py)

**置信度来自本机占据证据。** 本版新增的 `LocalSurveyEvidence` 按无人机分别维护占据置信度账本，不把 v9 拓扑图的 `node_confidence` 直接当作区域信息质量。每个账本仅接受该机自己的相机观测，以及它实际收到的邻居证据包；不从车端地图、其他无人机未发送的内部状态或仿真真值取数。训练用证据包跟随真实图广播传播，保留原始 `source/frame`。相同来源、帧、世界网格单元只融合一次；重复转发不会成为新的观测。[证据融合实现](../../../src/teamed-planning/scripts/simulator/survey_reward_v10.py)

传感器似然使用仿真假设 0.8，每次有效占据/空闲观测贡献有符号的 `log(0.8/0.2)`。累计 log-odds 截断到 [−3, 3]。单个无人机 i 对单元 x 的归一化置信度为：

\[
q_i(x)=\frac{\tanh(|\ell_i(x)|/2)}{\tanh(3/2)}\,\mathbf{1}_{\text{seen}_i(x)}.
\]

未知单元置信度为 0；明确的障碍与明确的空闲区域都算有效地图信息。0.8 是仿真传感器假设，不是实机概率校准结果。新增矛盾证据可以降低置信度，reward 保留这种有符号下降。

任务网格集合 Ω(a,b) 在整局固定，未知单元始终保留在分母中。团队指标取每个位置的最高本机置信度，再对整个任务区域求平均：

\[
C_t=\frac{1}{|\Omega|}\sum_{x\in\Omega}\max_i q_{i,t}(x).
\]

这允许各机分工观测，不要求每架无人机都完整复制整张地图。将同一份未变化的证据复制给更多无人机，不会按接收者数量重复支付奖励；实际融合不同来源的信息仍可能提高某个本机账本的置信度。该空间汇总仅用于团队 reward 和集中 critic，不作为全局地图注入 actor。

初始观测先形成 C₀ 基线，不给第一步动作支付已有信息的奖励。每个实际执行步结束后，立即计入新位置取得的本机观测；下一步再广播这份观测时使用同一来源/帧标识去重。CBF 拒绝的动作不执行物理步，不虚构新位置或新观测。

**reward 与任务完成。** 当前每步团队 reward 为：

\[
r_t=100(C_t-C_{t-1})-0.01-0.05 I_t-0.1 M_t,
\]

其中 I 是沿用 v9 的归一化 CBF 干预成本，M 是沿用 v9 的 CBF 余量风险（`cbf_margin_scale=8`）。这两个量的定义未改。完成探测时额外加一次 100；CBF 不可行、实际碰撞、越界或通信断开终止时扣 100。物理失败步仍计入实际发生的当步成本，CBF 执行前拒绝只记录失败终止转移。预算耗尽单独记为超时；人工停止不算成功。

完成条件必须同时满足以下要求并连续保持 30 个物理步：

- 区域平均置信度 `survey_confidence >= 0.85`。
- 至少 90% 的任务单元达到单元置信度 0.8，即 `survey_coverage >= 0.9`。
- 无人机网络与车辆 c 连通，且没有当步安全失败。

没有车辆到达奖励，车辆到达也不单独结束探测任务。导航代价改善、图路线质量、前沿推进、探索面积和 surprise 奖励均关闭。车辆路径和到达仍可作诊断。模型选择使用 `survey_completed`，并计入区域置信度、覆盖与安全终止；验证另外报告“a 初始等于 c / 独立 a”与“车静止 / 车移动”四个组别。[环境和 rollout](../../../src/teamed-planning/scripts/simulator/survey_experiment.py)

**通信约束。** 复用实际一跳几何驱动的通信骨架和 CBF；通信半径 95 m，飞行高度 15 m，车辆最大线速度 1 m/s。初始机群必须与车连通。任务生成还检查机数、通信距离和走廊半宽形成的保守通信范围限制。该几何检查用于排除明显超范围任务，不保证所有场景内的 CBF 都可行，也不覆盖未建模的实机延迟和动态。训练环境继续使用同步、完整的一跳几何；中继状态证据的丢包/延迟按 v9 设置保留。

**微调配置。** [mappo_training_v10.json](../../../src/teamed-planning/config/algorithm/mappo_training_v10.json) 的主要设置如下：

| 设置 | 值 |
|---|---|
| 无人机数量 | 3 |
| 车辆初始 c 到 b 距离 | 80–200 m，按环带面积采样 |
| 独立 a 相对初始 c 的偏移 | 20–60 m，方向随机 |
| 初始 a=c 比例 | 25% |
| 车辆静止比例 | 50%；其余使用原车辆控制器朝 b 行驶 |
| 探测走廊 | 半宽 12 m，网格 2 m |
| 训练 | 800 回合，8 个采样进程，16 回合组成一个同步 PPO batch |
| 学习率 | 0.0001 |
| PPO / critic epochs | 4 / 8，保留 KL guard 和 PopArt |
| 折扣 / GAE | 0.995 / 0.97 |
| 熵系数 | 0.003 线性降至 0.001 |
| 回合预算 | `min(1000, max(400, ceil(3*|b-a| + 240)))` |
| 验证 / 快照间隔 | 每 48 / 32 回合 |
| 训练种子 | 80000 起，共 8000 个候选场景种子 |
| 验证种子 | 90000–90011，每个 2 次采样 |
| 保留测试种子 | 100000–100009，不用于选模 |

随机种子集合与原 v9 分开。25% 和 50% 是采样概率，不保证每个小批次恰好达到该比例。

迁移来源为 `artifacts/mappo-v9-mixed100to400-no-subgoal-800ep-20261105-153146/best.pt`，对应 v9 的第 576 回合检查点。只复制兼容的 actor 编码器、策略层和 `log_std`。扩展首层的 8 个任务列初始化为 0，因此迁移瞬间给定相同旧观测时的动作分布与 v9 一致；新列在 PPO 更新中可学习。critic、PopArt 统计和优化器均从新状态开始。本版不是在原 v9 输出目录继续 resume。

**显式任务与 API。** 配置中的 `survey_a`、`survey_b`、`survey_vehicle_start` 必须一起设置或一起为 null。三个值用于仿真初始布置；运行中的任务消息只需要 a、b、宽度，c 始终来自真实车辆广播。例如：

```json
{
  "survey_a": [480.0, 500.0],
  "survey_b": [600.0, 500.0],
  "survey_vehicle_start": [450.0, 500.0],
  "survey_half_width": 12.0
}
```

`SurveyTask(a, b, half_width, world_size=1000., resolution=2.)` 构造不可变区域；`to_message()` 只输出 a、b、half_width。`SurveyExperiment(..., survey_task=task, ...)` 在原 Experiment 的实际观测、通信和 CBF 流程上增加本机证据融合与信息 reward。训练入口仍为 `simulator/train_mappo.py`；`run_survey_episode` 管理探测完成、安全失败和预算终止，`make_survey_scene` 生成随机任务或显式任务场景。

专项检查分别位于 [模型测试](../../../src/teamed-planning/scripts/tests/simulator/test_survey_model_v10.py)、[reward 测试](../../../src/teamed-planning/scripts/tests/simulator/test_survey_reward_v10.py)、[训练集成测试](../../../src/teamed-planning/scripts/tests/simulator/test_survey_training_v10.py)。测试用于验证接口、记账、权重迁移和执行流程；训练效果须由独立场景验证与保留测试结果判断。
