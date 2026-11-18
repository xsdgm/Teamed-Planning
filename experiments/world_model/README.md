# 世界模型导航算法

vtmp2 与 vtmp3 的世界模型实现统一放在 [world_model](../../src/teamed-planning/scripts/world_model/) 下，使用统一训练入口并通过配置选择架构。当前 vtmp2 默认训练原版 WildScenes 的粗区域边界外推模型，不启动仿真。已有精细语义模型、二值二维原型、旧 RSSM 与 vtmp3 的既有数据和权重契约仍保留。

**2026-11-13 后续要求：**vtmp2 改为真实森林数据集离线训练语义地形补全模型，接入完整系统后只在 Docker 三维仿真验证，不再返回二维模拟器采样、训练或评估。接口适配与已知差距见 [vtmp2 训练与部署要求](vtmp2/README.md)。下文二维入口保留用于历史复现，尚不是新部署链路。

公共 [models.py](../../src/teamed-planning/scripts/world_model/models.py) 保留旧 RSSM 基础实现，并通过模型工厂选择粗区域外推、精细语义补全、二值地形、旧单目标或 A—B 模型。旧架构参数名保持兼容；粗区域模型使用独立的 schema 6，拒绝混用 schema 5 精细语义及更早的权重。

| 版本 | 实现 | 测试 | 说明 |
| --- | --- | --- | --- |
| vtmp2 默认 | [frontier_model.py](../../src/teamed-planning/scripts/world_model/terrain/frontier_model.py) | [test_frontier.py](../../src/teamed-planning/scripts/tests/world_model/vtmp2/test_frontier.py) | 16m 粗区域、已观测边界外 32m 的结构外推，辅助无人机探索 |
| vtmp2 历史精细语义 | [semantic_model.py](../../src/teamed-planning/scripts/world_model/terrain/semantic_model.py) | [test_semantic.py](../../src/teamed-planning/scripts/tests/world_model/vtmp2/test_semantic.py) | 原版 WildScenes 的 14 类、2m 整窗口语义补全 |
| vtmp2 历史二值模型 | [decoupled_model.py](../../src/teamed-planning/scripts/world_model/terrain/decoupled_model.py) | [tests/world_model/vtmp2](../../src/teamed-planning/scripts/tests/world_model/vtmp2/) | 独立车机决策与简化 WildScenes 二维原型 |
| vtmp2 历史 RSSM | [model.py](../../src/teamed-planning/scripts/world_model/rssm/model.py) | [tests/world_model/vtmp2](../../src/teamed-planning/scripts/tests/world_model/vtmp2/) | 单目标驾驶／侦察互斥任务 |
| vtmp3 | [ab_model.py](../../src/teamed-planning/scripts/world_model/survey/ab_model.py) | [tests/world_model/vtmp3](../../src/teamed-planning/scripts/tests/world_model/vtmp3/) | [A—B 区域选点与代理建图训练](vtmp3/README.md) |

各阶段的职责由功能模块区分：`mission.py` / `planner.py` 处理单目标任务，`ab_environment.py` / `ab_planner.py` 处理两点任务。`survey/ab_model.py` 保存模型及训练逻辑，保留与单目标不同的数据契约和损失；两种任务不能混用数据或权重。

## 目录整理

此前整理将原先的 5 个小模块合并到对应职责中；新独立车机版本使用独立模块扩展：

| 原模块 | 当前实现 |
| --- | --- |
| `rssm.py` | [models.py](../../src/teamed-planning/scripts/world_model/models.py)：共享循环模型与版本选择 |
| `reconnaissance.py` | [runtime.py](../../src/teamed-planning/scripts/world_model/rssm/runtime.py)：配置侦察范围与适配冻结控制器 |
| `ab_contract.py`、`ab_mapping.py`、`ab_navigation.py` | [ab_environment.py](../../src/teamed-planning/scripts/world_model/survey/ab_environment.py)：任务契约、建图后端、已接收地图与导航循环 |

各类和函数的计算逻辑保持不变。`model.py` 和 `ab_model.py` 分别保留两种任务的模型类、`WorldModel` 别名和训练损失；`ab_wildscenes.py` 保留独立的真实地形后端。迁移后的 `ProxyMapper`、`ReceivedMap` 和 `SurveyNavigationEnv` 仍分别负责产生证据、使用已接收证据及执行任务循环。

源码哈希清单已同步采用合并后的文件。检查点参数结构与动作契约保持不变，续训仍严格检查源码和数据；既有实验继续使用其冻结源码快照。

## 训练与验证

统一入口位于 [simulator/train_world_model.py](../../src/teamed-planning/scripts/simulator/train_world_model.py)。从仓库根目录启动：

```bash
python -I -B src/teamed-planning/scripts/simulator/train_world_model.py \
  --algorithm vtmp2 --output artifacts/vtmp2-semantic-NEW --device cuda

python -I -B src/teamed-planning/scripts/simulator/train_world_model.py \
  --algorithm vtmp3 --output artifacts/vtmp3-NEW --device cuda
```

入口虽保留在历史的 `simulator` 目录，vtmp2 默认分派至 [frontier_training.py](../../src/teamed-planning/scripts/world_model/frontier_training.py)，
只读取真实数据，不创建模拟环境。默认把原版 K-01/V-01 的语义证据重新聚合成粗区域，分别训练和验证。
`--data-manifest` 指向原始点云转换出的语义证据清单，不是旧模型的预测；缺少原始数据时明确报错，不回退到模拟采样。
历史精细语义配置在 [world_model_vtmp2_semantic_legacy.json](../../src/teamed-planning/config/algorithm/world_model_vtmp2_semantic_legacy.json)，
二维配置在 [world_model_vtmp2_binary_legacy.json](../../src/teamed-planning/config/algorithm/world_model_vtmp2_binary_legacy.json)。

配置为 [world_model_vtmp2.json](../../src/teamed-planning/config/algorithm/world_model_vtmp2.json) 和 [world_model_vtmp3.json](../../src/teamed-planning/config/algorithm/world_model_vtmp3.json)，通过 `algorithm` 与 `model_architecture` 明确选择版本。也可以使用 `--config` 指定配置；命令行选项覆盖配置默认值。

从 `src/teamed-planning/scripts` 运行全部世界模型测试：

```bash
python -E -s -B -m unittest discover -s tests/world_model -t . -v
```

历史实验保留启动时的源码快照；目录整理与公共核心提取不会改变已启动的训练进程。旧版 `world_model/mission_pipeline.py` 和新版 `world_model/ab_pipeline.py` 作为任务流程实现保留，推荐通过统一入口启动。
