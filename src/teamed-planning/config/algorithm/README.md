# 算法配置

| 用途 | 配置 | 对应任务 |
| --- | --- | --- |
| ROS 运行 | [无人机](drone.yaml)、[地面车](ground.yaml)、[GCS](operator.yaml) | 三类节点的运行参数 |
| MAPPO v9 | [v9](mappo_training_v9.json) | 以车辆到达终点为目标的导航 |
| MAPPO 区域探测 | [v10](mappo_training_v10.json)、[v11](mappo_training_v11.json)、[v11.2](mappo_training_v11_2.json) | A–B 区域调查。vtmp3 也读取 v11.2 的任务参数 |
| vtemp2 | [分层策略](vtemp2_hrl.json) | 慢策略的子目标、时限、距离课程与奖励 |
| vtmp2 | [粗地形与前沿](world_model_vtmp2.json)、[GOOSE 地形预训练](world_model_vtmp2_goose.json)、[GOOSE 离线抽象验证](world_model_vtmp2_abstract_goose.json) | 地形预测、观测证据与前沿选点 |
| 历史 vtmp2 模型 | [精细语义模型](world_model_vtmp2_semantic_legacy.json)、[二值模型](world_model_vtmp2_binary_legacy.json) | 对应各自的模型与数据 |
| vtmp3 | [代理场景](world_model_vtmp3.json)、[WildScenes](world_model_vtmp3_wildscenes.json) | A–B 任务选区世界模型 |
| 场景划分 | [公里任务](world_model_vtmp2_km_scenes.json)、[600 米区域](world_model_wildscenes_scenes.json) | 二值 vtmp2 与 WildScenes vtmp3 的场景清单 |

[早期 MAPPO 至 v8 实验索引](../../../../experiments/mappo/mappo_early_through_v8.md) 汇总了训练报告、原始日志及权重保留状态。
