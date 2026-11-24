# vtemp4 长距离导航的学习前视转向

vtemp4 在三机一车的二维仿真中运行。当前默认上层策略根据任务终点、车端已观测路网、历史指令和实驶轨迹指定车机共享航点。下层沿用冻结的 v9 MAPPO 控制无人机，地面车通过局部规划器跟踪已确认可通行的道路。[当前单组训练与评估入口](../../../src/teamed-planning/scripts/simulator/run_vtemp4_continuous_history.py) · [vtemp 与 vtemp1 规则基线](../../../src/teamed-planning/scripts/world_model/README.md)

任务要求是公里级连续导航。同一回合从起点运行到车辆进入最终目的地 2 米内、触发安全终止或耗尽整局时间预算。中途前视方向可多次调整，车辆位置、车端融合图和整局计时持续保留。首轮建议沿用已有课程的 1,000–1,500 米起终直距，在独立路线统计整局到达率，再按原版长程验证的 2、3、5 公里档位分别评估。对实景地图分别记录起终直距、可行参考路线长度和实际行驶里程。[公里课程](../../../src/teamed-planning/scripts/world_model/README.md) · [原版长程评测清单](../../../artifacts/vtemp-5km-400best-20261102/manifest.json)

## 车辆决策图与连续角度选点

正式单组入口新建任务时默认选择 `periodic_vehicle_graph_goal_residual_lstm`。网络预测相对最终目标方向的有符号偏角。终点相对车头的方位由观测中的正弦和余弦还原，两角相加并折回一个圆周，得到车体坐标中的首选方向。偏角为零时首选方向指向终点，正偏角向左，负偏角向右。策略在本次合法角域上采样，PPO 用同一分布重算实际发布角度的概率。默认选点间隔 30 秒，车辆推进权重 10，航点奖金比例 0.10。各版本用独立的动作标识保存检查点，续训按原任务清单选择策略。[偏角策略](../../../src/teamed-planning/scripts/simulator/vtemp4_body_graph.py) · [单组入口](../../../src/teamed-planning/scripts/simulator/run_vtemp4_continuous_history.py)

车体坐标对照用 `--policy-variant periodic_vehicle_graph_body_lstm`。每次决策以当前车位为原点、以车头为前轴，将道路、历史指令和车迹统一表示为前向和左向位移。终点与地面局部行驶点使用相对方位的正弦、余弦和归一化距离。图注意力与全历史 GRU/LSTM 读取这些特征。动作分布直接定义在相对车头的物理角度上，并在当次合法方向内归一化。与原 LSTM 配对的圆障碍实验共同使用 30 秒选点周期、推进权重 10、前视点奖金比例 0.10，并按相同地图与无人机种子顺序从头训练。[车体观测与车辆图](../../../src/teamed-planning/scripts/simulator/vtemp4_body_graph.py) · [物理角分布](../../../src/teamed-planning/scripts/simulator/vtemp4_body_angle.py)

世界坐标对照使用 `periodic_vehicle_graph_goal_lstm`。每个已观测道路节点有 8 个原始属性，另加道路节点到终点的世界坐标位移（米数除以 350）。道路边携带实测车迹确认的正反向通行次数和最近通行时间。车辆节点用注意力读取道路、当前局部行驶点和指令历史。车载近场观测用于标记车辆能直接接入的道路。终点独立参与状态编码，动作头还直接读取终点相对车头的方位正弦、余弦和距离（米数除以 350）。[网络与观测](../../../src/teamed-planning/scripts/simulator/vtemp4_vehicle_graph.py) · [通行证据](../../../src/teamed-planning/scripts/simulator/vtemp4_road_traversal.py)

每次选点读取本局此前发布的全部共享航点，包括重复发布的同一坐标。每条指令关联其生效期间已实际行驶的片段，按局部重规划和上层指令切换的时间分段。每段保留起终点、当时的局部点、持续时间和实际里程。单向 GRU 按行驶顺序编码一条指令的车迹，单向 LSTM 再按发布时间编码全部指令及其执行反馈。车辆用四头时间注意力读取各步 LSTM 状态，再将历史摘要与当前路图融合。首次选点使用空历史，本次发布的航点从下一次决策起可见。旧版车辆图对照可用 `--policy-variant periodic_vehicle_graph_goal` 运行，采用最近 8 条加附近最多 8 条更早指令的检索方式。[历史观测](../../../src/teamed-planning/scripts/simulator/vtemp4_decision_graph_history.py) · [时序编码](../../../src/teamed-planning/scripts/simulator/vtemp4_vehicle_graph.py)

上层 Actor 和 Critic 均去掉了 24 向地形摘要、无人机及通信状态、旧的固定时间运动与任务摘要、可选角度编码。合法弧只供输出时的采样、动作概率计算和落点检查使用。边界、已知障碍以及整环受阻时的既有勘察回退规则由执行器处理。原统一注意力对照使用 `--policy-variant periodic_unified_attention_goal_no_grid`（无近场像素）或 `--policy-variant periodic_unified_attention_goal`（有近场像素）。复现 r8 的时钟与奖励权重时，各组均指定 `--review-interval-s 30 --progress-reward-weight 8`。[输出约束](../../../src/teamed-planning/scripts/simulator/vtemp4_periodic_history.py)

新版车辆图 LSTM 从第 0 个仿真秒起默认每隔 30 秒指定一次共享航点。15 秒对照使用 `--review-interval-s 15`。每次决策先计算车位到最终目标的直线距离。距离小于或等于 200 米时，直接指定最终目标的地图坐标。距离超过 200 米时，由网络根据车辆决策图、终点和执行历史输出相对车头的连续角度，再沿该方向取距车 200 米的地图坐标。两种情况下，航点都广播给无人机，并在地图上固定到下一次定时决策。车辆在两次决策之间可以继续平移、转向或绕障。航点的到达、受阻和进展情况都不改变上层决策时间。地面车每 5 秒规划下一段路。[定时决策与执行器](../../../src/teamed-planning/scripts/simulator/vtemp4_periodic_history.py)

第一阶段用 16 个物理 CPU 核心并行采样 400 局长程随机局部绕障任务。每批 16 个完整回合，收齐后更新 PPO，共 25 次更新。训练完成后在留出的两张长程随机障碍图上评估该策略。

## 早期的固定航点三组对照

三组各从头训练 400 局，每 16 局做一次 PPO 更新，共 25 次。训练入口给三组分配相同的地图生成种子和机群种子，第一阶段仅使用起终直距 1,000–1,500 米的在线随机局部绕障图。[对照入口](../../../src/teamed-planning/scripts/simulator/run_vtemp4_waypoint_comparison.py)

| 组别 | 上层决策 |
| --- | --- |
| 连续角度选点 | 指定新点时，以此刻车头为零度，用连续角度在半径 200 米的圆周上确定一个地图坐标。保持动作沿用此前确认的坐标和广播编号。 |
| 圆周候选选点 | 通过空间注意力从同一圆周上 48 个方向中选点，相邻方向相差 7.5°。持续接近航点或沿已知路径取得进展时保持坐标和广播编号。 |
| 固定航点＋历史注意力 | 在相同的选点模型中增加对过去 15、30、45、60 秒车辆轨迹的注意力输入。两种固定航点模型使用逐参数相同的初始权重。 |

半径和车头角仅在指定新点的瞬间参与计算。此后即使车辆平移、转向或局部路线改变，也会广播相同的世界坐标。三组都由地面车规划接下来行驶的路段。上层每 15 个仿真秒复核，地面每 5 秒重规划。训练完成后，程序在两张留出的随机障碍图和左、右绕湖各一张开发图上逐图比较净推进、折返次数和整局到达率。[配对评估](../../../src/teamed-planning/scripts/simulator/evaluate_vtemp4_waypoints.py)

## 早期实验的滚动航向策略

上层默认每 15 个仿真秒重评一次。每次重评先读入车端当时已收到的路网、地形和机群状态，再由网络选择保持或转向。保持时，前视方向在地图坐标系中固定，前视点随车前移。转向时，网络输出相对上一次前视方向的连续有符号转角，按一整圈取模更新方向。例如上次指向正东，本次左转 37°，新的前视方向就是东偏北 37°。车辆此刻为绕树产生的车头偏转不计入该角度。

前视距离默认上限为 200 米，执行层沿指令方向生成标记，靠近地图边界或终点时缩短距离。局部规划器默认每 5 秒更新一次行驶点。当前视标记落在树上时，规划器仍可沿已知可通行区域向该方向行驶。默认进入终点 25 米范围后转为最终接近，车辆到终点 2 米以内才算任务成功。[闭环运行时](../../../src/teamed-planning/scripts/simulator/vtemp4_runtime.py)

## 统一注意力对照组的慢层观测

沿用 vtemp2 公里级三机仿真的车端观测来源。每次重评用同一时刻的车载测量和已经送达车端的消息构造以下输入。

| 输入 | vtemp4 每次重评读取的内容 |
| --- | --- |
| 车辆和长程任务 | 本车速度、车头朝向和角速度，最终目的地相对车的位置、剩余时间预算，以及当前地面行驶点的相对位置和有效位。 |
| 融合路网 | 已融合的道路节点、观测边、置信度、观测次数和源观测年龄。按前视方向读取从车旁可接入的连续通路。 |
| 地形 | 将本车实测及无人机已回传的三态地形按位置融合，沿走廊分段统计空闲、占用和未知比例。尚无观测的区段记录为 100% 未知。有栅格的对照组额外读取车载近场的占据和未知像素。 |
| 机群和通信 | 一跳可见无人机的相对位置与速度，以及实际送达车端的中继状态、消息年龄、跳数和不确定度。 |
| 当前前视指令与执行情况 | 当前前视方向及其有效位、该方向保持时长、最近一个重评周期的车辆位移和向最终目的地的净进展。沿同向滚动前视点时继续累计保持时长。 |

无栅格的统一注意力对照在打包上层观测和保存 PPO 样本时，从共用车端输入中去掉两张 16 × 16 像素平面。该对照以当时车头为零度，沿 24 个方位读取证据，每条走廊沿程分八段。这些方位只用于读取地图，动作可取连续角度。[慢层观测构造](../../../src/teamed-planning/scripts/ground/vtemp4_observation.py) · [走廊统计](../../../src/teamed-planning/scripts/world_model/navigation/navigation_evidence.py)

### 早期滚动航向策略的图注意力

道路节点先沿观测到的边经过两层图注意力聚合。慢层再用车的状态和当前前视指令生成查询，对最多 64 个道路节点和 24 × 8 个走廊地形分段分配权重。每个地形分段保留相对车的位置以及可通行、阻断、未知三类比例。因此，即使某片河面没有道路节点，它的阻断信息也能直接参与注意力计算。

在学习到的相关性分数上，每离车 50 米额外减 1。相关性相同时，车旁的道路和地形得到更大权重。较远处的已回传信息仍参与计算，完整方向走廊的编码也同时送入保持或转向的决策层。50 米是初始距离尺度，效果需在完整任务上评估。[慢层策略](../../../src/teamed-planning/scripts/ground/vtemp4_policy.py)

## 学习信号

单棵树后仍有路接回原方向时，地面局部规划器有空间自行绕行。连续障碍封住原方向而侧面有通路时，上层可以选择较大的转角。

PPO 只更新慢层网络。评估按路线输出整局到达、剩余距离、用时及换向记录。比较大绕行策略时，可结合独立路线上的到达率、首次有效转向位置和局部可绕行路段的换向次数。

### 慢层奖励

每次重评选定方向后，用这次决策至下次重评或回合结束之间的车辆运动结算一次奖励。当前使用以下初始系数。

`r = (w × (d_before − d_after) − 2 × C_turn) / D₀ − Δt / T + R_end`。新建车辆图 LSTM 训练的终点净推进权重 `w` 默认为 10，其他上层变体的默认权重为 2。目标感知策略可通过 `--progress-reward-weight` 显式指定权重。续训沿用训练清单保存的决策周期与权重。

| 项目 | 定义 |
| --- | --- |
| 最终目标进展 | `--progress-distance euclidean` 使用车到终点的直距。`--progress-distance traversable` 在原版 V9 圆点障碍图上使用车到终点的近似最短可通行路程。两种模式的 `D₀` 均取起终直距，整局按带符号的差值累加。 |
| 可选的前视点奖金 | 车辆图 LSTM 的圆形障碍对照使用 `--waypoint-progress-bonus-ratio 0.10`。自主选点时计算 `q = clip((d_before − 航点到终点的直距) / 200米, 0, 1)`，区间奖金为 `0.10 × q × max(w × (d_before − d_after) / D₀, 0)`。每段在 `heading_reviews` 中记录方向得分与实际奖金。规则直接指定终点、车辆停滞或退后的区间奖金为零。默认比例为 0。 |
| 换向代价 | 周期选点比较相邻两条指令各自发布时的世界方位，取最小夹角 θ。首次选点、同向重发（夹角不超过 10⁻⁶ 弧度）和规则直接指定终点时，`C_turn = 0`。策略主动换向时取 `0.5 + θ / 180°` 米。对起终直距 1,000 米的任务，90° 换向扣 0.002 分，180° 掉头扣 0.003 分。 |
| 用时 | `Δt` 是本次决策实际经过的仿真秒数，`T` 是整局时间预算。用完整个预算时累计扣 1 分。 |
| 回合结局 | 到达最终目的地 +10，原生安全终止 −5，整局超时和进行中均为 0。超时结束回合，整局累计用时扣分为 −1。训练采样切段时，从下一状态的价值估计接续回报。 |

慢层按有限时长任务计算整局无折扣回报（γ = 1）。这样大绕行最初远离终点的负进展能被后续接回路线的正进展抵消。到达前视点、保持方向和无人机新增地图面积均不给额外奖励。

圆点图的路线奖励实验使用 `--progress-distance traversable --progress-reward-weight 10 --review-interval-s 30 --waypoint-progress-bonus-ratio 0`。每局从仿真器的完整静态障碍图预计算到终点的路程，用相邻决策时车辆的实际位置结算。完整地图和路程场只用于训练与评估计分。若某个车位无法安全接回路网，就暂存最近一次有效路程，重新接入时再结算净变化。逐局记录的 `route_progress_m` 是车辆缩短的可通行路程，`route_unavailable_intervals` 是至少一个端点无法接入的区间数；原有 `remaining_m` 仍表示直距。

## 正式训练 · 双侧绕湖

每次同步 PPO 更新并行运行 16 个完整公里级任务。四类各占 4 个：湖挡直线且左岸的参考路程较短、湖挡直线且右岸较短、湖在直线路径旁、原版 V9 随机圆障碍。左右按车从起点面向终点定义。挡路湖泊的两侧都有连续地面通路，走任一侧都可完成任务。奖励沿用上文的终点进展、用时和小换向代价。两岸路线、湖岸几何和任务分类仅供地图核验、分层采样和离线评估。

野外部分有 10 个独立地形，每个各有三种湖位。训练使用 6 个地形，开发和测试各留出 2 个。30 个野外任务的起终直距为 1,105–1,210 米。20 个挡路任务在湖中央横断面至少需要偏离直达线 132–205 米，沿两侧的参考路程相差 79–190 米。两侧参考路线均通过了原生碰撞几何核验。[三种湖位预览](../../../artifacts/vtemp4-lake-curriculum-v1/lake_routes.png) · [净空核验结果](../../../artifacts/vtemp4-lake-curriculum-v1/diagnostics/native-route-check.json)

训练从 seed 43 的初始慢层权重开始，V9 始终冻结，共 50 次更新、800 局。每轮 16 局使用相同慢层权重和各自的机群种子，收齐后更新一次。每次更新保存慢层、Adam 和随机状态，完整回合可在恢复时复用。第 10、25、50 次更新分别在相同的 10 张开发图上与保持初始前视方向的基线配对比较，记录两侧实际绕过湖泊的次数、最终到达和直达场景的换向。[正式训练入口](../../../src/teamed-planning/scripts/simulator/run_vtemp4_formal.py) · [野外地图生成器](../../../src/teamed-planning/scripts/simulator/build_vtemp4_lake_suite.py) · [正式野外场景清单](../../../artifacts/vtemp4-lake-curriculum-v1/scenes.json)

使用 `teamedplanning-2d` 环境，在仓库根目录执行下列命令。启动器在输出目录中冻结仿真与训练代码。训练进度写入 `progress.json`，逐局进度保存在 `episodes/`，阶段对照结果位于 `development/`。如需恢复已有运行，在第二条命令上使用原输出路径并加 `--resume`。

```bash
python -I -B src/teamed-planning/scripts/simulator/build_vtemp4_lake_suite.py
python -I -B src/teamed-planning/scripts/simulator/run_vtemp4_formal.py --output artifacts/vtemp4-lake-formal-run
```

## 历史课程 · 溪谷过桥与随机障碍

早期的溪谷课程按左侧绕行、随机圆障碍、右侧绕行和野外直达轮换。用 `--curriculum circles` 可单独使用原版 24 张随机训练图和 10 张随机留出测试图。

野外部分包含 10 组独立的溪谷地形，训练、开发、测试按地形分为 6、2、2 组。每组的三种桥位共享起终点、河道、植被和两岸林道。大绕行场景只有一座侧桥可过，两岸之间的水面全线阻断，距起终连线最近的桥面约 244–337 米。野外训练集中，3 组在出发后 114–127 米遇河，另外 3 组在 388–535 米遇河。直达版本沿直线具备 1.5 米净空。桥位、参考线路和任务分类仅用于生成、采样与离线评价。[生成器](../../../src/teamed-planning/scripts/simulator/build_vtemp4_detour_suite.py) · [场景清单](../../../artifacts/vtemp4-detour-curriculum-v1/scenes.json) · [同一地形的三种桥位](../../../artifacts/vtemp4-detour-curriculum-v1/paired_routes.png)

在仓库根目录使用 `teamedplanning-2d` 环境运行。首次运行时可用下列首条命令重建野外地图。

```bash
python -I -B src/teamed-planning/scripts/simulator/build_vtemp4_detour_suite.py

python -I -B src/teamed-planning/scripts/simulator/train_vtemp4.py train \
  --curriculum mixed --field-data artifacts/vtemp4-detour-curriculum-v1 \
  --output artifacts/vtemp4-heading-train

python -I -B src/teamed-planning/scripts/simulator/train_vtemp4.py evaluate \
  --checkpoint artifacts/vtemp4-heading-train/heading.pt \
  --compare-always-hold \
  --output artifacts/vtemp4-heading-eval
```

`heading.pt` 保存慢层权重和前视设置，`episodes/` 下的逐局记录包含重评时刻、转角、共享标记及地形回传统计。评估默认使用检查点的课程，早期检查点默认使用随机圆障碍图。可用 `--curriculum mixed --split development` 在混合开发集上评估旧权重。`--compare-always-hold` 用同图、同起点和同一机群随机种子再跑一次固定世界方向策略，按场景类别报告到达率、剩余距离和换向次数。`--collector-seconds` 可用于短时联调，存在截断时只报告观察结果，整组成功率留空。

### 原随机障碍图的四次试跑

这次试跑按预定顺序使用 16 张原随机圆障碍图。前三次更新各取 4 段 240 秒采样，分别执行 64 个慢层决策，截断处接续价值估计。第 4 次使用 4 个完整任务，执行 1,021 个决策，四个任务均超时且无碰撞。全部 1,213 个训练决策中有 411 次采样换向。

确定性部署以换向概率超过 50% 为阈值。最后一次更新后，换向输出层的偏置加权重绝对值之和为 −0.5882，输入该层的各个分量经 tanh 后在 [−1, 1] 内。因此对任一正常输入，换向概率至多约 35.7%，确定性部署都会保持方向。训练中出现的换向体现了采样探索。[训练配置](../../../artifacts/vtemp4-heading-km-pilot-20261119-ECJKW0/manifest.json) · [四次更新记录](../../../artifacts/vtemp4-heading-km-pilot-20261119-ECJKW0/training.json)

训练前后都用相同种子在两条留出路线运行完整回合。1,117 米路线行驶 525.67 米，超时距终点 888.47 米；1,480 米路线行驶 372.66 米，超时距终点 1,292.41 米。初始和训练后策略各执行了 536 次重评，换向和碰撞均为零，两条路线的结果逐项相同。[配对评估](../../../artifacts/vtemp4-heading-km-pilot-20261119-ECJKW0/paired_evaluation.json) · [最终权重](../../../artifacts/vtemp4-heading-km-pilot-20261119-ECJKW0/heading.pt)

### 岔口换向对车辆的影响

在开发地形 `valley-dev-00` 的三个桥位版本中，车辆从正式起点出发，前视方向始终对准初始终点方位。每 15 秒核对慢层实际输入。第 60 秒车距离正前候选桥位还有约 99 米，无人机回传已让直桥与河面阻断有不同的观测。到第 240 秒，两侧桥位仍全部未知，左绕、右绕两组 17 次慢层输入逐元素相同。此时有证据判断直行受阻，选择哪一侧仍需要进一步探索。[车端观测对照](../../../artifacts/vtemp4-detour-curriculum-v1/diagnostics/development-observations.json)

在预先选定的开发任务 `valley-dev-00-detour_left`，车辆从过桥前的岔口出发。前 15 秒两组使用相同的前视方向，轨迹逐点一致。接下来一组始终保持，另一组每次重评按离线参考路线前方 80 米处校正指向。整段 300 秒内，保持组行驶 1.50 米、最大侧移 0.15 米，参考引导组换向 3 次，行驶 62.88 米、最大侧移 25.80 米。两组都未进入预定出口的 10 米范围。这个局部对照说明换向能改变车辆在岔口的运动，整局绕行能力仍须用网络在独立路线上的到达结果检验。[配对回放记录](../../../artifacts/vtemp4-detour-curriculum-v1/diagnostics/development-guidance.json) · [复现脚本](../../../src/teamed-planning/scripts/simulator/check_vtemp4_guidance.py)

同一任务也从正式起点运行了完整的 3,730 秒。保持组最大左移 4.2 米，全程留在起点一侧。离线参考引导组最大左移 278.1 米，在第 2,563 秒登桥，第 2,731 秒到达对岸。两组均零碰撞，超时时各距最终目的地 1,099 米、915 米。这个整局对照验证了给定绕行方向后，冻结的执行层在这张地图上能过河。网络自主发现通路和整局到达仍需独立评估。[完整轨迹与判岸记录](../../../artifacts/vtemp4-detour-curriculum-v1/diagnostics/development-full-guidance.json)

## 独立野外场景 · 溪谷林区

这个 1,440 × 1,440 米的合成场景用于检验策略是否能从直达方向转向侧面的过河通路。地图包含树丛、草甸、岩石、林道和一条贯穿林区的河流。起终点分处两岸，直距 1,130 米，木桥偏离直达线约 263 米。绕行参考路线长 1,361 米，按 1.5 米连续净空核验，封桥后两岸断开。[野外地图预览](../../../artifacts/vtemp4-field-valley-v1/field_preview.png) · [场景文件](../../../artifacts/vtemp4-field-valley-v1/case.json)

车辆在米制二维占据栅格上行驶，无人机在固定高度获得自身附近的观测。高程和树冠用于描述与展示地貌。策略通过车端已有测量和实际收到的回传选择方向。桥坐标和参考通路保存在离线审计文件中，用于核对几何可行性。本轮慢层训练与配对评估继续使用原先的随机圆障碍地图。[地图生成器](../../../src/teamed-planning/scripts/simulator/generate_vtemp4_field.py) · [几何审计](../../../artifacts/vtemp4-field-valley-v1/audit.json)

在仓库根目录可用已有慢层权重运行独立回放。下面的 `--seconds 60` 只检查开局感知和执行，省略该选项会运行到任务终止。

```bash
python -I -B src/teamed-planning/scripts/simulator/replay_vtemp4_field.py \
  --checkpoint artifacts/vtemp4-heading-train/heading.pt \
  --seconds 60 --output artifacts/vtemp4-field-valley-v1/replay-60s.json
```

第 3 次更新的慢层权重已在该场景运行 60 个仿真秒。车行驶 17.41 米，接收三机共 175 个地形观测块，完成 36 次路网融合，碰撞与净空违规均为零。开局时车辆距离桥区仍较远。[短时回放记录](../../../artifacts/vtemp4-field-valley-v1/replay-60s.json)

## 同目录的早期实验

`scripts/simulator/` 下的三个早期代理脚本和对应实验记录来自地面阶段目标与空中探测点的代理实验。其中连续坐标实验用角度和距离生成探测位置。上面的 `train_vtemp4.py` 入口独立训练前视转向策略。[代理实验](../../../src/teamed-planning/scripts/simulator/vtemp4_proxy_smoke.py) · [公里级路线实验](../../../src/teamed-planning/scripts/simulator/vtemp4_km_route_proxy.py) · [连续探测坐标实验](../../../src/teamed-planning/scripts/simulator/vtemp4_km_coordinate.py)
