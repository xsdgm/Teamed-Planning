# 早期 MAPPO 至 v8 实验记录

[原始日志归档](../../artifacts/mappo-early-through-v8-logs.tar.gz) 保存本工作区清理时仍有的运行记录。下表中的报告沿用实验当时的目录名、配置名和命令；当前训练配置见[配置索引](../../src/teamed-planning/config/algorithm/README.md)。

| 批次 | 实验报告 |
| --- | --- |
| 初代和千米地图 v2 | [初始二维训练](early/mappo_2d_initial.md)、[固定奖励训练](early/mappo_formal_training.md)、[移动车辆](early/mappo_moving_ground.md)、[链式通信与 CBF](early/mappo_chain_cbf_training.md)、[链式通信检查](early/mappo_chain_cbf_inspection.md)、[修正记录](early/mappo_corrected_20261016.md)、[千米地图、v2 复评、v3 子目标接力、route_graph 与链外任务](early/mappo_1000m_training_20261018.md) |
| 早期离散导航 v3、v4 | [v3 训练](early/mappo_navigation_v3_20261016.md)、[测试](early/mappo_navigation_v3_test_20261017.md)、[失败分析](early/mappo_navigation_v3_failure_analysis_20261017.md)、[v4 独立验证](early/mappo_v4_2d_validation_20261018.md)、[千米迁移](early/mappo_v4_1000m_transfer_20261018.md)、[最佳轮次分析](early/mappo_v4_best_epoch_diagnosis_20261018.md)、[平台期分析](early/mappo_v4_plateau_analysis_20261018.md) |
| 连续速度 v3 至 v5.2 | [速度策略、v4 奖励、v5.1 和 v5.2 对照](early/mappo_v3_velocity_20261022.md) |
| v6 原型与发布 | [MLP 惊讶度奖励](v6-v8/mappo_v6_surprise_20261026.md)、[早期 GAT](v6-v8/mappo_v6_gat_20261026.md)、[v6.2-1 与 v6.2-2](v6-v8/mappo_v6_2_20261026.md)、[正式 v6](v6-v8/mappo_v6_official.md)、[权重配对比较](v6-v8/mappo_v6_paired_comparison_20261027.md) |
| v7、v8 | [v7 动态通信骨架](v6-v8/mappo_v7_cooperation_20261027.md)、[v8 中继注意力与后续运行](v6-v8/mappo_v8_relay_attention_20261028.md) |

v6 的[发布清单](v6-v8/mappo_v6_release.json)记录了检查点与哈希，[配对比较数据](v6-v8/mappo_v6_paired_comparison_20261027.json)列出两组权重的评估结果。

保留的三个代表权重位于原运行路径。

| 模型 | 检查点 |
| --- | --- |
| 千米地图 v2 | [best.pt](../../artifacts/mappo-1000m-v2-20261020/best.pt) |
| 正式 v6，即 v6.2-2 GAT | [stage_01_final.pt](../../artifacts/mappo-v6.2-2-gat-stage1-20261026-150400/stage_01_final.pt) |
| v8 bridge15，100–400 米混合任务 | [best.pt](../../artifacts/mappo-v8-mixed100to400-bridge15-20261030-142558/best.pt) |
