# 世界模型导航算法

vtmp2 与 vtmp3 是同一个世界模型导航算法的不同迭代版本，统一放在 `src/teamed-planning/scripts/world_model/` 下。组织方式沿用无人机的 `SurveyActorCritic → RouteABActorCritic`：共享基础实现、按任务扩展模型、通过配置选择版本，使用统一训练入口。

公共 [rssm.py](../../src/teamed-planning/scripts/world_model/rssm.py) 实现编码器、循环记忆、先验/后验和预测头；[models.py](../../src/teamed-planning/scripts/world_model/models.py) 提供 `make_model(config)`。单目标和 A—B 版本分别继承基础类并指定动作、状态和结果维度；参数名保持兼容已有权重。

| 版本 | 实现 | 测试 | 说明 |
| --- | --- | --- | --- |
| vtmp2 | [model.py](../../src/teamed-planning/scripts/world_model/model.py) | [tests/world_model/vtmp2](../../src/teamed-planning/scripts/tests/world_model/vtmp2/) | [单目标驾驶／侦察任务](vtmp2/README.md) |
| vtmp3 | [ab_model.py](../../src/teamed-planning/scripts/world_model/ab_model.py) | [tests/world_model/vtmp3](../../src/teamed-planning/scripts/tests/world_model/vtmp3/) | [A—B 区域选点与代理建图训练](vtmp3/README.md) |

各阶段的职责由功能模块区分：`mission.py` / `planner.py` 处理单目标任务，`ab_mapping.py` / `ab_environment.py` / `ab_planner.py` 处理两点任务。`ab_training.py` 保留与单目标不同的数据契约和损失；两种任务不能混用数据或权重。

统一入口位于 [simulator/train_world_model.py](../../src/teamed-planning/scripts/simulator/train_world_model.py)。从仓库根目录启动：

```bash
python -I -B src/teamed-planning/scripts/simulator/train_world_model.py \
  --algorithm vtmp3 --output artifacts/vtmp3-NEW --device cuda
```

配置为 [world_model_vtmp2.json](../../src/teamed-planning/config/algorithm/world_model_vtmp2.json) 和 [world_model_vtmp3.json](../../src/teamed-planning/config/algorithm/world_model_vtmp3.json)，通过 `algorithm` 与 `model_architecture` 明确选择版本。也可以使用 `--config` 指定配置；命令行选项覆盖配置默认值。

从 `src/teamed-planning/scripts` 运行全部世界模型测试：

```bash
python -E -s -B -m unittest discover -s tests/world_model -t . -v
```

历史实验保留启动时的源码快照；目录整理与公共核心提取不会改变已启动的训练进程。旧版 `world_model/mission_pipeline.py` 和新版 `world_model/ab_pipeline.py` 作为任务流程实现保留，推荐通过统一入口启动。
