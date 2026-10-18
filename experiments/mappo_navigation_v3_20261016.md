# 导航修复后的 MAPPO 重训练

2026-10-16 13:55（Asia/Hong_Kong）启动，从随机初始化训练，未续用旧优化器或旧环境回放。

- 输出：`artifacts/mappo-navigation-v3-20261016/`。
- 启动 PID：15244，运行状态以该目录 `progress.json` 和实际进程为准。
- Conda：`C:/Users/Administrator/.codex/visualizations/2026/10/15/01a14037-907a-7550-92aa-9cbe790f5590/rl-conda`。
- CPU，PyTorch 2.14.1+cpu，Torch/OPENBLAS 单线程。
- 环境版本 3，包含地面入口选择、路线更新、分量路径边权、安全距离和终点连接修复。
- 2000 回合，每回合最多 240 步，随机种子 42。
- 学习率 0.0003，每批 PPO 更新 4 次；固定探索权重 0.6、熵系数 0.01。
- 每 25 回合在种子 20000–20004 验证，每 100 回合保存独立快照。
- 启动时先评估未训练模型，结果保存为 `initial_validation.json`；完成后进入训练。
- `provenance.json` 保存源文件摘要，`sources/` 保存启动时 Python 源码副本，`config.json` 保存实际配置。

启动依赖检查通过，进程正常进入初始验证，无 stderr 错误。训练是否收敛需根据后续验证判断。最佳模型仍按现有验证奖励选择，应同时观察车辆到达率、终点距离和碰撞指标。

此前版本 2 的训练输出保留在 `artifacts/mappo-corrected-20261016/`。
