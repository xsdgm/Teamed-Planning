# 2026-10-16 修复后二维强化学习训练

本轮从 `origin/强化学习` 的 `fbd29ed` 建立 `codex/rl-convergence-fixes`。新的环境版本为 2，从随机种子 42 重新初始化；旧模型和旧批次不能直接续训到新条件。

## 修复内容

- 同节点的图外连接使用与其他路径相同的 `1/epsilon` 惩罚；批量预测和断连前沿代理同步修正。
- 新的本机传感器帧更新视野内已有可通行节点；相同帧不重复计数，视野外及无效传感器回退不计数。
- Python 图消息携带每个来源节点的单调累计计数，融合取逐来源最大值。重复包、绕圈转发以及地面端同时收到原包和中继包，不会再次叠加相同来源计数。消息/接收图深拷贝，后续本机观测不会修改已经发出的历史包。
- Actor 的置信度特征与路径代价共用 `node_confidence`。旧的无来源 ROS/历史包只能保守取最大计数，不能提供相同的完整溯源；本轮验收限于 Python 二维仿真。
- 每个场景以固定种子确定性采样，目标距障碍至少 3.5 米，并检查相同净空栅格上的四连通路线。检查结果和参考路线不提供给控制器或学习器。
- 地面控制器保留仍可执行的图路径；收到新地图后只在缺少图路径、前沿等待或局部阻断等条件下重规划，避免周期性退回已走过的入口。
- 正式训练、独立评估和诊断统一使用到达目标后的终止条件。
- 增加 PPO、动作、奖励稀疏度及真实证据接收统计；训练前保存未训练基线。默认每 25 回合验证，每回合原子保存 latest，每 100 回合保留快照。

奖励系数、Actor/Critic 网络层数和 CBF 约束保持当前方案参数，仍为两项固定奖励。此次基础模块、场景及地面导航都发生改变，新旧批次不能直接拼接曲线或将差异全部归因于 MAPPO 学习。

## 验证

地面导航独立验收将充分观测的参考拓扑图提供给 GroundRobotController，仅用于验证控制闭环；没有把参考图用于强化学习训练。五个验证场景结果：

| 场景 | 到达步数 | 最终距目标 m | 碰撞 |
| --- | ---: | ---: | ---: |
| 20000 | 183 | 1.863 | 0 |
| 20001 | 165 | 1.431 | 0 |
| 20002 | 193 | 2.000 | 0 |
| 20003 | 147 | 1.647 | 0 |
| 20004 | 163 | 1.719 | 0 |

20001 因原目标净空不足进行了三次场景采样。记录位于 `artifacts/rl-fixes/ground-navigation.json`。新增回归覆盖单节点奖励反例、标量/批量一致性、重复包与中继回环、独立来源增量、本机帧去重、消息快照、地面端去重、旧包保守处理、场景可行性、PPO 分项、丢包与几何连通区别及地面闭环。

最终 47 项测试全部通过，日志为 `artifacts/rl-fixes/regression-final.log`。完整 8000 个训练种子生成检查没有失败，3049 个种子需要重采样，最多 11 次、平均 1.619875 次；记录位于 `artifacts/rl-fixes/scene-audit.json`。

3 回合完整短训练输出位于 `artifacts/rl-fixes/smoke`，包含每回合验证、latest/best/快照和配置/源码记录，共 615 个联合步。三个采样训练场景均到达、无碰撞和链路违约，模型参数及优化器状态有限，参数 L2 变化约 0.2066。两个确定性验证场景的平均奖励从未训练基线 2.1161 增至 22.8369，但到达率仍为 0/2。该结果只证明更新和保存链路可运行，不构成收敛、导航泛化或优于基线的证据。详细核验位于 `artifacts/rl-fixes/smoke-verification.json`。

正式批次已经从随机初始化启动，PID 3744，预算 2000 回合，独立输出到 `artifacts/mappo-corrected-20261016`。启动后已确认实际训练回合完成、分项指标有限及检查点保存，标准错误日志为空。进展和选模结果以运行目录中的 `progress.json`、`training.jsonl`、`validation.jsonl` 为准，PID 和本段状态是启动检查时的记录。

## 运行

使用 Conda 环境 `teamedplanning-2d` 或明确指定 Conda Python。当前机器已经建立的诊断/训练环境位于：

`C:/Users/Administrator/.codex/visualizations/2026/10/15/01a14037-907a-7550-92aa-9cbe790f5590/rl-conda`。

该环境使用 Python 3.11.17、PyTorch 2.14.1+cpu、NumPy 2.4.6、SciPy 1.17.1。BLAS 使用 OpenBLAS，避免 NumPy MKL 和 PyTorch 的 OpenMP 库冲突。启动器会加入所选 Conda 环境的 DLL 路径，使用隔离模式启动隐藏后台进程。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ./src/teamed-planning/scripts/simulator/start_mappo_training.ps1 -Python '<Conda环境绝对路径>/python.exe'
```

安全停止：在本批输出目录创建 `STOP`，程序完成当前回合后保存退出。移除该文件后，用相同配置、相同输出目录和 `-Resume` 恢复。不要向旧批次目录写入新训练。

```powershell
python -I -m unittest discover -s src/teamed-planning/scripts/simulator -p 'test_*.py' -v
python -I src/teamed-planning/scripts/simulator/scenarios.py --start 10000 --count 8000
python -I src/teamed-planning/scripts/simulator/validate_ground_navigation.py --steps 360
```
