# 实验索引

| 实验方向 | 报告与记录 |
| --- | --- |
| MAPPO 初代至 v8 | [按训练阶段查阅报告、日志和保留的权重](mappo/mappo_early_through_v8.md) |
| MAPPO v9 · WildScenes 导航 | [二维场景与基线](mappo/v9/wildscenes_v9_2d_20261106.md)、[局部感知适配](mappo/v9/wildscenes_perception_20261106.md)、[导航修复与长程验证](mappo/v9/v9_navigation_repair_20261110.md) |
| MAPPO v9 · 机群规模对照 | [三机与四机受控验证](mappo/v9/v9_fleet_controlled_20261111.md)、[车端入口选择修复与复评](mappo/v9/v9_fleet_entry_repair_20261111.md) |
| MAPPO v10 | [A–B 区域探测任务与微调记录](mappo/v10/mappo_v10_local_survey_20261108.md) |
| CoGraph（历史记录，代码已移除） | [v1 多车多目标协同建图](cograph/cograph_v1_20261102.md)、[v2 单任务训练与去中心化车队](cograph/cograph_v2_20261103.md) |
| 地面导航 | [二维导航修复](ground_navigation/ground_navigation_fixes_20261016.md)、[导航图完整性](ground_navigation/navigation_graph_completeness_20261017.md)、[断连原因](ground_navigation/navigation_graph_root_causes_20261017.md)、[建图修复验证](ground_navigation/navigation_graph_fixes_20261017.md) |
| 车机前视导航 | [vtemp4 公里级前视转向训练与评估](world_model/vtemp4/README.md) |
| 世界模型 | [实验总览](world_model/README.md)、[vtmp2 地形预测](world_model/vtmp2/README.md)、[vtmp3 区域选点](world_model/vtmp3/README.md) |

v9 的[长程批量验证](mappo/v9/validate_v9_long_batch.py)、[机群规模迁移验证](mappo/v9/validate_v9_fleet_transfer.py)、[迁移结果分析](mappo/v9/analyze_v9_fleet_transfer.py)和[受控对照验证](mappo/v9/validate_v9_fleet_controlled.py)与报告放在同一目录。报告中的验证命令从仓库根目录运行。

训练配置见[算法配置索引](../src/teamed-planning/config/algorithm/README.md)。本地运行的日志、权重和冻结源码见各报告引用的 [artifacts 目录](../artifacts/)。
