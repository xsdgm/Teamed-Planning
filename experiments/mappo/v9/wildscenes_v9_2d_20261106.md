# WildScenes 实测森林二维地图：vtemp 临时目标方案 + v9 权重

后续已增加[双分辨率局部感知适配](wildscenes_perception_20261106.md)。本文记录的是原2m无人机观测实验；当前工作区验证入口默认0.5m精细建图，运行原单栅格行为需显式指定 `--mapping-resolution 2` 并使用新输出目录。旧正式结果与冻结源码仍保留，不能将新配置结果混入本文实验。

本实验把 WildScenes 的真实点云及语义标签转换成原比例二维占据地图，使用 vtemp 在原 5 公里验证中采用的滚动临时目标方案，搭配冻结 v9 权重。用户要求公里级任务，不要求起终点直线相距 5km。数据集的采集总里程不作为导航任务距离。

本次正式结果保存在 `artifacts/wildscenes-v9-20261106/vtemp_rolling200/`，不向 `pre/` 写入任何文件。此前无临时目标的测试仅为基线，结果保留于根实验目录的 `evaluation/` 与 `evaluation_control/`；四个基线回放/报告已从 `pre/` 移至 `baseline_no_subgoal/report/`，不能将其当作 vtemp 验证。

## 固定模型与运行方式

- 权重：`artifacts/mappo-v9-mixed100to400-no-subgoal-800ep-20261105-153146/best.pt`，第 576 轮。
- SHA256：`7ac3aedcc3876902773b2241f60e839eb3e3b5630ddeb3c8be660421f2c24198`。
- 使用同一训练目录的 `runtime_sources` 和 `config.json`；验证时明确覆盖 `subgoal_radius=200`、`subgoal_arrival=25`。不修改权重、训练配置文件或冻结运行时，也不使用工作区里后来修改的控制器。
- 无人机保持 24×24m、2m 分辨率的局部观测；地面车保持 8×8m、0.5m 分辨率。保持三机、固定高度、原通信/中继与随机策略设置。
- 沿用二维运动和理想局部二值地图传感器。高度仅用于离线区分低矮障碍和高处树冠；不模拟三维飞行、坡地动力学、相机成像或林区无线传播。

## vtemp 临时目标点机制

复用 `artifacts/vtemp-5km-400best-20261102` 中 `rolling200` 的原实现。vtemp 归档与冻结 v9 的 `ground/subgoal_manager.py` 字节相同，SHA256 为 `e36971e1a194210a972c38c9d393e7f4c1d6ee970fc077d885aa141fef4e2702`；`ground/ground_main.py` 也字节相同。无需补丁运行时。

车辆以自身位置为中心，在半径 200、133.33、66.67 米的三圈、每圈 24 个方向上选择候选，按车辆到候选距离、候选到终点距离及失败惩罚之和评分。同分优先靠近终点。进入中间目标 25 米范围后重新选择；180 秒没有靠近或相关图增长、600 秒没有靠近，或相对最近距离后退超过 40 米时，也触发失败重选。接近临时目标 50 米内，可一次性吸附到 20 米内最近的已映射图节点。候选只依赖车的位置、收到的融合图、时间和已知世界边界，不读取实景地图真值或离线路线。

临时目标同时成为地面车行驶目标及广播任务目标，无人机通过原通信/中继获得它。回放显示地面车当前选择的目标，不表示所有无人机同一帧收到切换。换点不重置 episode、车辆位置或已融合地图；任务成功始终要求车辆到最终目标 2 米内，临时目标到达不算任务成功。此处复现原 5km 验证的实现，不额外声称包含通信风险候选评分或切换确认机制。

`test_wildscenes_vtemp.py` 用真实 best576 权重和冻结运行时验证初始选点、到达切换、停滞重选、广播目标、actor 实际收到的目标特征及终点保持不变，共 6 项测试。逐例结果记录 `subgoal_events`，包含选择、到达、失败重选、吸附事件；轨迹额外保留每个事件发生步。

## 数据与地图

官方数据：<https://data.csiro.au/collection/csiro:61541>；官方代码：<https://github.com/csiro-robotics/WildScenes>。数据许可为 CC BY-NC-SA 4.0，完整归属声明保存于下载清单。

下载 V-01 和 K-01 的完整 `Clouds`、`Labels`、`poses3d.csv`，每序列另取六张实拍图片及二维标定作为来源对照。原始文件不重命名；下载验证长度和 S3 ETag/MD5，并记录 SHA256。位姿时间与文件名以 Decimal 无损匹配，兼容小数尾零差异。

点云通过 `R(qx,qy,qz,qw) @ p_sensor + t` 对齐。仿真只平移原点，比例为 1 米对应 1 米。初版网格为 0.5m，另以 0.25m 检查窄通道的离散误差，正式公里级任务采用后者。网格需有地面语义支持才可标为可通行；小孔洞的地面高度插值半径上限为 1m，其余区域保持未知并阻挡。树干、倒木、石块等在车辆 2m 高度包络内形成障碍；较高树冠不直接压成地面障碍。泥地和水体在本版保守抽象中阻挡。这些是明确的地图转换假设，并非真实车辆通行性标注。原 v9 未定义车高，2m 包络是本次建图假设。每格至少 3 点的阈值保持相同，因此分辨率变化也影响稀疏点支持，差异不完全来自边界量化。

地图、位姿、逐帧数据和转换参数的散列/统计保存在各地图目录的 `map_metadata.json`、`source_scans.json`。此地图为实测几何的二维近似，存在标注、遮挡与重建误差。

## 任务与评价

先在地图上按 1.5m 安全净空搜索公里级参考路线；若不足，保留该检查结果，再显式按 0.3m 车体半径建立物理可行任务，并在任务清单记录区别。禁止自动放大地图、删除障碍、拼接成伪真实长地图或向控制器提供采集轨迹/参考路线。

参考路线采用八邻接米制栅格最短路，禁止斜穿角，并逐段通过连续扫掠几何检查；它不是连续空间的严格最优路线。比较多个连通分量，按最长已验证路线选择，而非只选择面积最大的分量。任务按几何选择后冻结，正反方向共享地图，不视作独立林区。每个任务预设三个随机样本，所有成功、超时与失败均保留。预算为 `max(600, ceil(3 * reference_length_m + 240))`，不以策略表现调整地图或任务。

正式任务清单为 `maps/K-01-r025/cases_physical_multicomponent.json`：1.01087km 路线正/反向，直线746.54m；1.51418km路线正向，直线938.31m。三个任务均按0.3m物理净空验证，起终点另满足1.5m净空。每例三个固定随机样本，共九次。

**公平性限制：上述三例的起终点在1.5m净空栅格中属于不同连通分量。它们是整个v9系统的实景迁移压力测试，并非对原1.5m安全约束可解的常规导航基准；失败不能单独归因为权重泛化。** 独立审计保存在 `diagnostics/geometry_audit/`。另外建立200m、全路径满足1.5m净空的短程对照，单独报告，不混入公里级成功率。

分别报告：终点到达、无碰撞到达、0.3m 物理扫掠违规、碰撞尝试、1.5m 安全净空违规、通信/编队违规、实际行程、剩余距离和终止原因。环境回退碰撞运动时仍记录原始尝试；不能把回退后无穿透当作无碰撞。严格成功要求到达且上述安全指标无违规。

## 复现入口

下载工具需要 `requests` 和 `boto3`；转换及验证使用既有 `/home/bdi/miniconda3/envs/teamedplanning-2d/bin/python`（NumPy/SciPy/PyTorch）。渲染使用带 NumPy/Pillow 的 Python。

```bash
python src/teamed-planning/scripts/simulator/download_wildscenes.py \
  --root artifacts/wildscenes-v9-20261106 --sequence K-01 --workers 16

/home/bdi/miniconda3/envs/teamedplanning-2d/bin/python \
  src/teamed-planning/scripts/simulator/prepare_wildscenes.py \
  --sequence-dir artifacts/wildscenes-v9-20261106/raw/WildScenes3d/K-01 \
  --output artifacts/wildscenes-v9-20261106/maps/K-01

/home/bdi/miniconda3/envs/teamedplanning-2d/bin/python \
  src/teamed-planning/scripts/simulator/select_wildscenes_cases.py \
  --map artifacts/wildscenes-v9-20261106/maps/K-01/forest_map.npz \
  --output artifacts/wildscenes-v9-20261106/maps/K-01/cases_safety.json \
  --sequence K-01 --target-lengths 1000 1200 --clearance 1.5
```

实际采用的任务清单、运行参数与结果以最终验证输出的 `manifest.json`、`results.json` 和逐案例 JSON 为准。`diagnostics/` 与 `artifacts/wildscenes-runner-tests/` 中的短程或步数截断测试仅检查接口，不计入正式公里级结果。

正式0.25m地图与两组验证的命令如下；已有结果目录再次执行需要加 `--resume`，其校验会拒绝不同输入、代码或权重混入。

```bash
/home/bdi/miniconda3/envs/teamedplanning-2d/bin/python \
  src/teamed-planning/scripts/simulator/prepare_wildscenes.py \
  --sequence-dir artifacts/wildscenes-v9-20261106/raw/WildScenes3d/K-01 \
  --output artifacts/wildscenes-v9-20261106/maps/K-01-r025 --resolution 0.25

/home/bdi/miniconda3/envs/teamedplanning-2d/bin/python \
  src/teamed-planning/scripts/simulator/select_wildscenes_cases.py \
  --map artifacts/wildscenes-v9-20261106/maps/K-01-r025/forest_map.npz \
  --output artifacts/wildscenes-v9-20261106/maps/K-01-r025/cases_physical_multicomponent.json \
  --sequence K-01 --target-lengths 1000 1500 --clearance 0.3 --endpoint-clearance 1.5

/home/bdi/miniconda3/envs/teamedplanning-2d/bin/python \
  src/teamed-planning/scripts/simulator/validate_wildscenes_vtemp.py \
  --cases artifacts/wildscenes-v9-20261106/maps/K-01-r025/cases_physical_multicomponent.json \
  --output artifacts/wildscenes-v9-20261106/vtemp/evaluation --samples 3 --workers 3

/home/bdi/miniconda3/envs/teamedplanning-2d/bin/python \
  src/teamed-planning/scripts/simulator/select_wildscenes_cases.py \
  --map artifacts/wildscenes-v9-20261106/maps/K-01-r025/forest_map.npz \
  --output artifacts/wildscenes-v9-20261106/maps/K-01-r025/cases_control_200m.json \
  --sequence K-01-control --target-lengths 200 --clearance 1.5

/home/bdi/miniconda3/envs/teamedplanning-2d/bin/python \
  src/teamed-planning/scripts/simulator/validate_wildscenes_vtemp.py \
  --cases artifacts/wildscenes-v9-20261106/maps/K-01-r025/cases_control_200m.json \
  --output artifacts/wildscenes-v9-20261106/vtemp/evaluation_control --samples 3 --workers 2

/home/bdi/miniconda3/envs/teamedplanning-2d/bin/python \
  src/teamed-planning/scripts/tests/simulator/test_wildscenes_vtemp.py
```
