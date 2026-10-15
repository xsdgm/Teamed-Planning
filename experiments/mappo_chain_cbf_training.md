# 链式通信与 CBF 正式训练批次

2026-10-15 16:23（香港时间），按用户指令启动新批次。启动时已验证后台进程完成实际训练回合并保存检查点，标准错误日志为空。

- 输出目录：`artifacts/mappo-chain-cbf-20261015/`
- 环境：Conda `teamedplanning-2d`，Python 3.11.17，PyTorch 2.14.1+cpu，隔离模式 `-I`，不使用 GPU。
- 初始化：随机种子 42，从头训练；不混入旧批次的 732 回合记录。
- 预算：2000 回合，每回合 120 个一秒联合仿真步。
- 通信与控制：固定相邻链路、逐跳证据中继、CBF 最小间距和最大链路距离过滤。
- 奖励：保留前期探索、后期关键区聚焦的平滑课程，加入局部接收信息下的间距惩罚与安全过滤介入代价。
- 每 50 回合使用独立验证场景评估，每回合保存 `latest.pt`，验证改善时保存 `best.pt`，每 100 回合保存独立快照。

具体配置见 [mappo_training.json](../src/teamed-planning/config/algorithm/mappo_training.json)。该运行目录下的 `config.json`、`provenance.json` 和 `sources/` 保存实际启动配置和主要源文件快照。

实际进度以运行目录内 `progress.json`、`training.jsonl`、`validation.jsonl` 为准，不以本文静态描述为准。需要停止时，在本批目录创建 `STOP` 文件；停止后先移除该单文件，再使用启动器 `-Resume` 恢复。

旧 `artifacts/mappo-formal-20261015/` 仍停止于第 732 回合，保留其检查点、日志和 STOP 文件。新旧批次执行条件不同，不应拼接比较训练曲线；CBF 的验证范围与限制见 [停机检查报告](mappo_chain_cbf_inspection.md)。
