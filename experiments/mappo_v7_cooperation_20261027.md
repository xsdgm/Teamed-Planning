# v7：不绑定编号的协同探索（代码已接入，策略效果待训练）

## 正式训练启动记录

经用户指示，已于 **2026-10-27 17:14:09（北京时间）** 启动独立后台训练；初始主进程 PID 为 1702140。使用 teamedplanning-2d 环境、800 回合、8 个采样进程、每批 16 回合、3 架无人机、100–283.8 m 任务距离，从头训练 v7。启动检查时 8 个工作进程均已进入实际仿真采样；这不是训练完成或效果提升结论。

本次目录：[mappo-v7-stage1-20261027-171143](../artifacts/mappo-v7-stage1-20261027-171143/)。当前状态以该目录的 [progress.json](../artifacts/mappo-v7-stage1-20261027-171143/progress.json) 和各回合 `running/` 记录为准；[console.log](../artifacts/mappo-v7-stage1-20261027-171143/console.log) 保存控制台输出。配置和训练源码已由训练器自动快照。以下实现与验证部分记录的是启动前的检查结果。

## 目标与实现边界

无人机共享一个策略，各机依据自己的地图、局部传感器、任务广播和实际可见邻居输出水平速度。靠近车辆维持通信、居中充当中继、远端继续探索，都是随位置和任务变化的行为，不预先指定给 drone_0、drone_1 等编号。

这里把“自主协作”落实为**学习协同移动与空间分工**。通信骨架由确定性的局部几何规则生成，安全投影也不是神经网络；没有宣称已经学出了离散角色分配器。能否形成有用的接力链，需要正式训练与消融实验回答。

v7 只改动 Python 仿真、训练和检查点恢复路径。ROS/PX4 实飞接口尚未接入这种速度策略；原有 real 模式限制继续生效。v8 的消息年龄/跳数模型、vtemp 的中间目标层均仅设计。

## 为什么原来的 v6 不够

[ChainState](../src/teamed-planning/scripts/drone/chain_state.py) 根据 fleet_ids 固定前驱、后继；CBF 必须保住这些边。共享团队奖励能够鼓励协作，但不能解除固定链条。原 [GAT](../src/teamed-planning/scripts/drone/graph_attention_mappo.py) 编码的是导航地图节点与可通行边，不是无人机消息图。

## v7 的三个组成部分

### 1. 动态局部通信骨架

[local_backbone.py](../src/teamed-planning/scripts/drone/local_backbone.py) 使用相对邻域图（relative neighborhood graph，RNG）。对当前可见的 i、j，仅在不存在第三节点 k 同时满足下式时保留 ij：

\[
\max(\|p_i-p_k\|,\|p_j-p_k\|)<\|p_i-p_j\|.
\]

车辆也作为节点；所有距离按当前仿真的三维位置计算。见证点 k 若能够删除 ij，一定也在两端的通信半径内，因此各端使用完整、同步的一跳几何信息即可独立计算同一条边。保留等距边，编号只用于识别消息对象，不参与距离排序或职责分配。

连通性依据：任一连通单位球通信图中的最小生成树边，都不能被一个两段均更短的见证点删除，否则替换跨割边会降低生成树权重。因此 RNG 包含连通骨架。每步用已有 CBF 保护本步 RNG 边，再在下一步重新选边，允许拓扑随位置改变。

**成立条件不能省略**：初始通信图连通；邻居几何完整、同步；所有无人机按同一控制周期协同执行；速度有界；车辆速度不超过指定界；每个局部 CBF 可行。遇到不可行仍按现有机制吸收终止，不把零速度当成安全保证。该论证不覆盖时延、单侧丢包、无线遮挡、多径、实际飞控动力学或定位误差。v7 配置明确要求 drop=0。

骨架可以是分支结构，不要求排成一条直线。远距离任务会使有效通信连接需要多段中继，但是否主动保持合适的中继位置取决于训练所得策略。

### 2. 地图注意力 + 邻机集合注意力

[cooperative_mappo.py](../src/teamed-planning/scripts/drone/cooperative_mappo.py) 新增 `gat_cooperative_v7`：

| 输入/模块 | 内容 |
|---|---|
| 自身状态与传感器 | 34 维状态 + 2×12×12 像素；保留任务方向、车辆位置、历史自身运动和 CBF 余量 |
| 地图分支 | 继承 64 个局部地图节点的 GAT；仅使用自己观测/实际收到并融合的地图 |
| 动态骨架状态 | 当前保留边数量、直接车链路、端点均值/最远端点、最长边比例；替换固定角色、序号和前后端点 |
| 邻机分支 | 最多 15 架其他无人机，填充为 16 槽；相对 xyz、相对水平速度、是否骨架邻居、距离、有效标记 |
| 融合 | 自身/地图编码作为 query，邻机 token 作为 key/value，masked attention 输出 32 维上下文 |
| 动作 | 2 维高斯 latent → 半径受限水平速度 → CBF 安全投影 |

原始输入宽度 1554，编码后宽度 418。不含无人机编号 one-hot、固定 rank 或远端无人机真实状态。新骨架/邻机分支使用当前直接相对几何；任务和地图仍通过已有接收/转发路径到达，各机地图仍可能不同、残缺。v7 尚未显式建模这种地图时效差异。

全部邻居缺失时注意力上下文为零，填充槽不参与匹配。输出对邻居槽位排列不变；重命名设备不会改变相同物理输入的动作。策略没有直接访问车辆完整地图或 simulator full_state 中的机群状态。

### 3. 团队学习与实际转发

[mappo_experiment.py](../src/teamed-planning/scripts/simulator/mappo_experiment.py) 选择新 Navigator，继续通过实际邻居广播转发融合图及训练用证据审计包。审计包仅用于确认奖励对应的证据确实送达车辆，不输入 Actor。

奖励沿用 v6 的车辆端成图、任务路径改善、前沿推进、有界惊讶度，以及时间、CBF 干预/余量成本。中继无人机虽然可能少发现新区域，仍获得同一个团队优势；远端发现但未送达车辆的证据不会凭空计入送达收益。没有给“拥有中继编号”或“宣称是中继”直接奖励。

Critic 在训练时聚合各 Actor 的合法输入及集中状态；Actor 分散执行。PPO 记录并优化 CBF 前的采样 latent 概率，不用投影后的速度计算 log probability。继承同步整批采样、GAE、KL 限制、独立 PopArt Critic 与冻结惊讶度评分。

原有 `chain_*` 指标名称在 v7 中保留以兼容训练/惊讶度记录，其值改为本步实际保留的动态骨架边；新增 `backbone_connected`、`backbone_required_links`。成图成功还须满足机群与车辆连通。评估端的全局重建仅用于指标与 Critic，不给 Actor 补全缺失观测。

## 配置与兼容

新配置：[mappo_training_v7.json](../src/teamed-planning/config/algorithm/mappo_training_v7.json)。architecture=`gat_cooperative_v7`、communication=`backbone`、feature_version=7；actor_input_version=2 仅代表基础状态/传感器约定，不能拿来认定 v6/v7 检查点兼容。

v6 文件、默认行为、旧模型加载分支保留。v7 新开训练目录；不能把 v6 检查点直接当作 v7 恢复。当前配置为 3 架、100–283.8 m 阶段，未启用中间目标，未扩大原阶段范围。800 回合是待运行配置，不表示已经训练。

项目环境为 `/home/bdi/miniconda3/envs/teamedplanning-2d/bin/python`。当前 shell 注入 Isaac Sim 的 PYTHONPATH，会导致 OpenCV/NumPy 冲突；命令使用 `-E` 忽略外部 Python 环境变量，无需改动 Conda 包：

```bash
/home/bdi/miniconda3/envs/teamedplanning-2d/bin/python -E -m unittest discover \
  -s src/teamed-planning/scripts/simulator -p test_cooperative_v7.py -v

# 正式训练入口；本次未启动该长训练
/home/bdi/miniconda3/envs/teamedplanning-2d/bin/python -E \
  src/teamed-planning/scripts/simulator/train_mappo.py \
  --config src/teamed-planning/config/algorithm/mappo_training_v7.json \
  --output artifacts/mappo-v7-stage1 --workers 8
```

## 验证与下一步实验

[专项测试](../src/teamed-planning/scripts/simulator/test_cooperative_v7.py) 覆盖几何规则的双端对称、重命名/排列不变、等距情况、多个随机点集的连通分量保持、CBF 连续步内连通、远端转发位置不冒充物理邻居、空邻居处理、真实短程训练/验证/检查点/恢复，以及实际远端证据多跳送达。

首轮完整 simulator 回归：206 项、1 项按设计跳过，其余通过（104.929 秒）。随后增加并增强了一个真实多跳/角色交换测试，v7 专项 8 项全部通过（7.033 秒）：把 drone_1 放到车辆近端、drone_0 放到远端，远端证据经实际邻机转发后进入车辆账本，远端区域的导航节点也进入车辆融合图，且通信持续连通。短训练只验证实现链路，不代表已经学会中继行为。

## 四页汇报的构建

根目录 [vtemp.tex](../vtemp.tex) 是中文 Beamer 源文件；[vtemp.pdf](../vtemp.pdf) 是已编译的四页 16:9 宽屏报告。使用官方 Tectonic 0.17.0 便携二进制（压缩包约 9.7 MB），按需获取宏包；系统字体为 Noto Sans CJK SC 与 DejaVu Sans。已下载编译器位于临时目录 `/tmp/vscode-agent-codex-sandbox-4Uy6ew/v7-tex/tectonic`，缓存也在该临时根目录；源文件不依赖这些绝对路径。

安装 Tectonic 后在项目根目录运行 `tectonic vtemp.tex` 即可重建。也可使用含 xeCJK、Beamer、TikZ 的 XeLaTeX 环境编译两次。首次 Tectonic 构建需要网络下载宏包，后续可复用缓存。编译中间产物放在忽略版本管理的 `build/vtemp/`。

正式实验固定场景种子、总采样预算和多随机初始化，对比 v6、v7 去邻机注意力、完整 v7。记录：车辆成图率/到达率、远端证据送达率及延迟、通信中断比例、CBF 不可行率、首次成图时间、路径质量、各机移动与证据贡献；交换编号和初始位置，检验职责跟随物理状态；单独关闭某个实际中继转发作反事实评估。不要用注意力热图或一条漂亮轨迹证明协作已经学成。
