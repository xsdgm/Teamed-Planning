# WildScenes 2D 局部感知适配

本次修改针对 WildScenes 点云投影地图、VTemp rolling200 临时目标和 v9 best576 权重。无人机新增同位置、同 24×24m 视野的精细局部观测，默认 0.5m（48×48），仅用于语义转换和拓扑建图。它是感知分辨率设置的调整，并非从原 2m 观测中推断出未提供的细节。

原 12×12、2m 占据图继续进入 Actor 和送达证据账本，网络结构和权重不变。地面感知保持 8×8m、0.5m。新增 `mapping_known_mask` 区分覆盖缺失与实测占据；未知、边界外以及任何与阻挡细格相交的区域仍禁止通行。建图保留 1.5m 净空及原米制节点/边参数，重新构造建图器以同步像素缓存。

主要实现位于 `scripts/drone/local_mapping.py`、`drone_main.py` 和 `scripts/simulator/wildscenes_map.py`。当前控制器直接支持双分辨率；冻结 v9 通过显式子类适配，历史 `runtime_sources`、训练配置和权重未被编辑。临时目标选择规则仍为原 VTemp 200m / arrival25。

## 使用

`validate_wildscenes_vtemp.py` 现在默认 `--mapping-resolution 0.5`，支持 0.25、0.5、1、2m。指定 2m 时使用原单栅格流程。新旧协议、感知配置和适配源码散列写入 manifest；不同配置不能混用 resume，回放也会拒绝签名或感知配置不一致的逐例文件。历史实验结果保持在原目录，新运行应使用新目录。

```bash
/home/bdi/miniconda3/envs/teamedplanning-2d/bin/python \
  src/teamed-planning/scripts/simulator/validate_wildscenes_vtemp.py \
  --cases artifacts/wildscenes-v9-20261106/maps/K-01-r025/cases_control_200m.json \
  --mapping-resolution 0.5 --samples 3 --workers 2 \
  --output artifacts/wildscenes-v9-20261106/perception_r05_full_control
```

此命令是后续完整对照的入口；本次实际执行的是下面列出的单样本验证，不把它冒称为完整六次对照。

## 验证结果

49项测试通过：局部感知7项、栅格适配9项、速度策略与图编码17项、冻结VTemp6项、冻结v9精细感知集成5项、报告隔离5项。检查包括4m安全直通道恢复节点和边、未知缺口保持阻挡、坐标与尺度、Actor传感器特征逐元素不变、实际收到的证据包仍是12×12@2m、冻结资产散列不变，以及异常后的运行时恢复。另验证相同设置resume可复用、改变分辨率时拒绝复用且不写坏原manifest。

实际 K-01 重放结果保存在 `artifacts/wildscenes-v9-20261106/perception_r05_20261106/`：

| 运行 | 原2m | 新0.5m |
|---|---|---|
| 1km正向r0，前30步截断检查 | 融合图最多0节点/0边；车程6.857m | 最多2节点/0边；车程0.982m |
| 200m反向r0，原完整预算 | 最多23节点/36边；第144步CBF终止 | 最多60节点/94边；第146步CBF终止 |

2m重放的前30步采样轨迹与原正式结果逐项一致。新200m运行行驶79.832m，无碰撞尝试、物理扫掠或1.5m净空违规，但仍未到达终点。这证明新感知进入了实际建图、通信和车辆融合链路，不证明导航成功率提高。

独立离线抽样在两条公里路线各取11个位置，0.5m使空图窗口从10/11、11/11降到0/11、0/11，新增节点和边均通过源地图1.5m连续净空检查。这些位置不是实际无人机轨迹，不计入导航成功率；公里路线仍大多仅形成孤立节点。

## 剩余限制

精细感知不能补齐未采集区域，不能让原1.5m约束下不连通的公里任务变得可解，也未修复不可达临时目标或CBF编队恢复。Actor仍看到原2m传感器图，但细化后的图结构会改变其图特征；送达证据及桥接校验仍受2m网格限制。后续改变这些模块应分别验证。

检查摘要、原始案例、局部探针与源码快照：

- `artifacts/wildscenes-v9-20261106/perception_r05_20261106/verification.json`
- `artifacts/wildscenes-v9-20261106/perception_r05_20261106/report/control_reverse.html`
- `artifacts/wildscenes-v9-20261106/diagnostics/perception_adaptation_probe/resolution_probe_20261106T145254.json`
- `artifacts/wildscenes-v9-20261106/perception_r05_20261106/implementation/`
