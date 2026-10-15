# MAPPO 2D 实现与复现实验

当前默认方案已改为移动无人车闭环、240 秒回合与 C-ADMM 路径代价改善 + 加权探索奖励。上一静止地面批次停于第 637 回合。配置、奖励公式与验证说明见 [移动无人车方案](../../../../experiments/mappo_moving_ground.md)。下文早期实验记录中的固定地面端说明仅对应历史批次。

2026-10-15 停机检查后，已加入固定链式通信、实际证据中继和固定高度 2D CBF 安全层。旧正式训练停于第 732 回合；[检查、修复和回放结果](../../../../experiments/mappo_chain_cbf_inspection.md)。16:23 按用户指令在独立目录启动 chain/cbf、CPU 的新正式训练批次，见 [新批次记录](../../../../experiments/mappo_chain_cbf_training.md)。

正式动态奖励训练入口为 `train_mappo.py`；训练配置和后台启动、断点恢复说明见 [正式训练说明](../../../../experiments/mappo_formal_training.md)。以下保留原小规模实验复现方式。

新增模块：`drone/mappo.py`（共享候选 Actor、集中 Critic、观测白名单与共同动作筛选）、`simulator/mappo_experiment.py`（联合步采样、PPO/GAE、送达证据账本和评估）、`simulator/mappo_report.py`（离线轨迹回放）。原默认算法仍为 C-ADMM。

## 运行

在已有 2D Python 环境中安装 `requirements-rl.txt`（CPU 版 PyTorch 即可），从仓库根目录运行：

```powershell
python -m pip install -r src/teamed-planning/scripts/simulator/requirements-rl.txt
python src/teamed-planning/scripts/simulator/test_mappo.py
python src/teamed-planning/scripts/simulator/mappo_experiment.py --episodes 40 --steps 48 --output artifacts/mappo-2d
python src/teamed-planning/scripts/simulator/mappo_report.py artifacts/mappo-2d/results.json
```

本机 PyTorch CPU 版已统一安装到 Conda 环境 `teamedplanning-2d`，与其他依赖共用同一个环境。无需配置 `PYTHONPATH`。本机复现可使用：

```powershell
conda activate teamedplanning-2d
python -m pip install "torch==2.14.1+cpu" --index-url https://download.pytorch.org/whl/cpu
& "$env:USERPROFILE/.conda/envs/teamedplanning-2d/python.exe" src/teamed-planning/scripts/simulator/mappo_experiment.py --episodes 40 --steps 48
```

后台启动器默认使用该 Conda 解释器，并以 `-I` 隔离模式启动，忽略外部 `PYTHONPATH` 和用户级包路径；启动前检查 PyTorch 是否来自所选环境。已有训练批次的源文件快照和环境版本记录保留历史内容。

`--episodes 0 --checkpoint artifacts/mappo-2d/model.pt` 仅评估；`--drop .2` 对各算法应用同一概率的随机丢包（轨迹不同导致接收机会不同，并非完全相同的数据包轨迹）；`--no-preference` 和 `--no-exploration` 支持奖励项消融，必须重新训练后比较。默认种子 7–46 用于训练，1001–1003 用于独立地图和目标测试；目前只训练了一个随机种子，尚未完成多次训练的统计验证。

## 接入现有控制器

`DroneController(navigation_algorithm='mappo', mappo_checkpoint='.../model.pt')` 使用训练模型；也支持 `cadmm`、`cadmm_controlled`、`greedy`。ROS 参数见 `config/algorithm/drone.yaml`，默认不启用学习策略。模型只保存参数，使用 `weights_only=True` 加载；缺少模型时报明确配置错误。已加载模型推理失败则悬停并计数。训练场景参数为 120m 地图、95m 通信、24m 方形相机、6m/s、15m 高度、三机；不能把该小场景模型视为已适配默认 ROS 场景。

## 对比版本的准确含义

- `cadmm_original`：原版算法及回退，按实际相机设置 `fov_radius=12m`；没有共同动作筛选，也没有增加悬停动作。
- `cadmm_controlled`（统一接口版）：复用原候选生成与 C-ADMM 代价、对偶更新；增加悬停动作，并使用与 MAPPO/贪心相同的合法性掩码。调用原求解器时设置 `fixed_candidates=True`，禁止它再增加偏移候选或投影生成新点。原来的链邻居惩罚保留为优化目标的一部分；公共硬筛选使用直接上传约束。因此它是适配版，不能当成原算法结果。
- `greedy`：新增加的确定性基线，分数为局部低置信度收益代理减去距离、偏离任务走廊和悬停代价；没有读取真实未知收益。
- `mappo`：候选逐点共享网络打分，可变动作掩码；按联合时间步收集整队动作；Critic 对各机合法特征做掩码平均汇总；共享团队奖励，GAE、裁剪 PPO、熵项和梯度裁剪。没有模仿学习预训练。

MAPPO 与两组适配基线使用相同建图器、图融合、相机观测、控制器和预算。原版算法的外围规则不同，必须独立披露。共同筛选检查下一秒线段的三维间距，并要求所有预期邻机都存在；该保守规则可能降低同时移动比例。邻机预测依赖上一轮收到的拟定航点，不保证双方同时改选动作时仍然安全；报告另行测量实际运动线段最小间距。

## 信息与奖励边界

编码器只读取本机位置、当前融合图、已收到任务的位置/目标/路径、已收到邻机位置和航点。嵌套原始广播、全局机器人状态、速度、障碍列表不会传入网络。无有效任务或预期邻机缺失时只保留悬停；本适配器不额外缓存任务。上游 ROS 已有消息超时过滤，但完整路径时效、图消息历史与中继缓存一致性仍需进一步验收。ROS 构造观测时已去除“缺失邻机位置用本机位置补齐”的行为。

奖励不使用融合图的观测计数。2D 训练给真实广播图包附加 `evidence_audit` 审计载荷，由地面接收端接到后再计分；它不进入 Actor/Critic，也不要求部署增加消息。每源/帧去重，每帧每世界格仅投一次质量加权票。空闲和障碍均计入有效证据；无效传感器回退不计分；冲突可使置信度下降并产生负奖励。该账本衡量送达证据重建的置信度，尚未验证与实际地面拓扑融合置信度之间的校准关系。

置信度用截断有符号 log-odds 的 `tanh(abs(log_odds)/2)` 表示，传感器可靠度暂设 0.8、截断 ±3；这是假设，尚无独立实测校准。重点区域采用车附近 18m、目标附近 18m、车目标线段半宽 12m 的并集。固定尺度 ZC=ZA=400m²、ZD=18m，距离权重 0.03、每秒代价 0.01。重点权重初值 1、范围 [0.1,2]，外部首次观测权重初值 0.15、范围 [0.05,0.5]，外部项每步封顶；20 秒正增益窗口缓慢调整权重。账本不因区域移动重置。

## 已实现与尚未验收

本轮验证真实 2D 仿真闭环、训练更新、独立场景评估、证据去重、未知空闲不误计、冲突负增益、白名单隔离、动作置换等价与缺失任务/邻机等待。输出包含整体/重点/外部收益、首次面积、全员移动比例、距离、实际间距违约、近距碰撞、接收中断、回退和决策耗时（当前计时包含整个控制器处理，不是单独网络耗时）。

另报告独立几何真值上的 Brier 误差与校准误差（ECE），真值只进入评估汇总；重点区域平均置信度达到 0.8 后逐步降低额外奖励。现已补充 2D 证据中继与 CBF 回归。尚未完成：运动无人车任务、通信延迟和独立多训练种子实验、完整奖励消融、真实传感器校准与无效重复观测指标、XTDrone/ROS 实飞验证。2D 地面障碍不会被当成飞行高度上的障碍。近距碰撞评估阈值为无人机中心距 2m，不代表真实机体几何碰撞模型。当前结果只用于判断实现能否训练运行，不是完整方案验收。
