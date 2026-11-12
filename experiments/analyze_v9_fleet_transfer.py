"""Read completed frozen-v9 fleet evaluation and generate a report and route figure."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    status = json.loads((root/'status.json').read_text())
    assert status['status'] == 'completed', 'Only report a completed experiment'
    verification = json.loads((root/'verification.json').read_text())
    assert all(verification.values()), verification
    results = json.loads((root/'results.json').read_text())
    comparison = json.loads((root/'comparison.json').read_text())
    cases = json.loads((root/'cases.json').read_text())
    assert len(results) == status['total'] and all(r['status']=='completed' for r in results)
    assert all(r['limit'] is None for r in results), 'Smoke tests are not full-task evaluations'
    assert all(r['hold_seconds']==30 and r['max_hold_displacement_m'] < 1e-9 for r in results)
    assert all(r['first_motion_step'] is None or r['first_motion_step'] > 30 for r in results)
    assert all(not r['success'] or r['remaining_m'] <= 2. for r in results)
    assert all(r['steps'] <= r['budget_steps'] for r in results)
    by_key = {(r['case'],r['sample'],r['fleet'],r['layout']):r for r in results}
    samples = sorted({r['sample'] for r in results})
    for case in (2,3,4):
        for sample in samples:
            a, b = (by_key[(case,sample,n,'native')] for n in (3,4))
            assert a['sample_seed']==b['sample_seed'] and a['budget_steps']==b['budget_steps']
    for sample in samples:
        a = json.loads((root/'data'/f'case4_r{sample}_n3_native.json').read_text())
        b = json.loads((root/'data'/f'case4_r{sample}_n4_preserve_base.json').read_text())
        assert a['initial_positions'][:3] == b['initial_positions'][:3]
        assert a['initial_positions'][-1] == b['initial_positions'][-1]
    paired = [p for p in comparison['pairs'] if p['both_strict_success']]
    shorter = sum(p['four']['travel_m'] < p['three']['travel_m'] for p in paired)
    faster = sum(p['four']['steps'] < p['three']['steps'] for p in paired)
    def pct(value):
        return '不可比较' if value is None else f'{value:+.2f}%'
    def outcome(row):
        return {'vehicle_arrival':'到达','cbf_infeasible':'CBF 无可行动作',
                'budget_exhausted':'预算耗尽'}.get(row['termination'],row['termination'])
    lines = ['# v9 冻结策略：先探测 30 秒的三机与四机完整任务对照', '',
        '日期：2026-11-11。使用原 best576 权重与冻结运行时，未训练、未放宽安全约束。', '',
        '所有任务均先让无人机探测 30 个仿真秒，地面车在此期间保持原地，收图、融合、规划照常。'
        '30 秒后解除运动锁定，不额外强制重规划；等待时间计入原任务总预算。'
        '逐步核验了等待期间车辆位移为零。', '',
        f'共完成 {len(results)} 次完整任务：三个案例 × {len(samples)} 个策略采样种子 × '
        f'两种机数，另加 Case 4 的 {len(samples)} 次初始队形控制。', '',
        '| 编队 | 到达 | 无违规到达 | CBF 安全终止 | 预算耗尽 |',
        '|---|---:|---:|---:|---:|']
    for fleet in ('3','4'):
        group = comparison['groups'][fleet]
        lines.append(f"| {fleet} 架 | {group['arrivals']}/{group['completed']} | "
            f"{group['strict_successes']}/{group['completed']} | {group['cbf_failures']} | {group['budget_exhausted']} |")
    lines += ['',f"双方均无违规到达的 {len(paired)} 对任务中，四机有 {shorter} 对里程更短、"
              f"{faster} 对耗时更少。配对总里程变化 {pct(comparison['paired_travel_change_pct'])}，"
              f"配对总仿真时间变化 {pct(comparison['paired_time_change_pct'])}。",
              '负数表示四机减少；未完成任务不能因路程较短而被认作更优。', '',
              '## 逐次结果', '',
              '| 案例 / 种子 | 三机结果 | 四机结果 | 三机里程 m | 四机里程 m | 里程变化 | 三机时间 s | 四机时间 s | 首次连通 s（三／四） |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for pair in comparison['pairs']:
        a,b=pair['three'],pair['four']
        lines.append(f"| Case {pair['case']} / r{pair['sample']} | {outcome(a)} | {outcome(b)} | "
            f"{a['travel_m']:.2f} | {b['travel_m']:.2f} | {pct(pair['travel_change_pct'])} | "
            f"{a['steps']} | {b['steps']} | {a['first_connection_step']:.0f} / {b['first_connection_step']:.0f} |")
    lines += ['', '首次连通指车辆当时位置至终点在已交付地图中的连通时间，-1 表示从未连通。'
              '车辆在移动，因此它不能单独衡量远处地图质量。时间包括起步等待和转向；'
              'CBF 安全终止的时间包含最后一次未执行的失败决策。', '',
              '## Case 4 初始队形控制', '',
              '冻结代码在起点位于地图右半侧且机数不为三时，会把全部 UAV 的横向初始偏移镜像。'
              '本补测保留原三架初始位置，在车位 +(27m,18m) 增加第四架，其他代码和配置保持一致。', '',
              '| 种子 | 三机里程 m | 默认四机里程 m | 保留原起点四机里程 m | 控制组结果 | 控制组时间 s | 控制组首次连通 s |',
              '|---|---:|---:|---:|---|---:|---:|']
    for sample in samples:
        a,b,c = (by_key[(4,sample,n,layout)] for n,layout in
                 ((3,'native'),(4,'native'),(4,'preserve_base')))
        lines.append(f"| r{sample} | {a['travel_m']:.2f} | {b['travel_m']:.2f} | {c['travel_m']:.2f} | "
                     f"{outcome(c)} | {c['steps']} | {c['first_connection_step']:.0f} |")
    lines += ['', '## 安全与复现核验', '',
              '所有任务保存逐步位置、车辆实际计划航点、逐步标量指标和终止原因。'
              '独立从车辆轨迹计算里程，并与模拟器累计里程核对。三个原始三机 r0 '
              '任务均逐帧复现已有 wait30 实验，除运行耗时外的摘要指标一致。', '',
              '报告中的“无违规”按冻结仿真记录的车／机碰撞、越界、机间距违规、链路违规、'
              '通信中断及 CBF 安全终止判定，不代表真实飞行安全验证。', '',
              '```json',json.dumps({k:v['safety_sums'] for k,v in comparison['groups'].items()},
                                  ensure_ascii=False,indent=2),'```', '',
              '权重 SHA-256：`'+json.loads((root/'provenance.json').read_text())['checkpoint_sha256']+'`。'
              '权重、原训练配置与本轮冻结源码未变。', '',
              '## 解释范围', '',
              '- 三个任务直线距离约 209、301、392 米，是此前选定的成功场景；不能外推为独立地图泛化率或公里级 WildScenes 结果。',
              '- 固定权重只在三机条件下训练，本轮测量直接迁移到四机的实际表现。',
              '- 同一对任务使用相同策略采样种子；增加一架会改变随机数消耗顺序，并非前三架动作噪声逐步相同。',
              '- 主对照沿用默认初始化；Case 4 应同时阅读保留原起点的补测。',
              '- 有效前方信息、策略协同、图融合和地面规划共同影响行程；本试验不能独立证明某一项是绕路的根因。', '',
              '## 轨迹与数据', '',
              '![三机／四机车辆实际轨迹](ground_routes.png)', '',
              '灰色障碍仅供离线展示，不是策略输入的全局地图；彩色线为车辆实际轨迹，'
              '× 表示未到达时的终止位置。', '',
              '[逐次 CSV](results.csv) · [完整配对指标](comparison.json) · '
              '[配置与来源](provenance.json) · [完整性核验](verification.json)', '']
    (root/'report.md').write_text('\n'.join(lines))

    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig,axes=plt.subplots(len(samples),3,figsize=(15,4.6*len(samples)),squeeze=False)
    for row,sample in enumerate(samples):
        for col,case_no in enumerate((2,3,4)):
            ax=axes[row,col]
            case=cases[str(case_no)]
            all_points=[]
            variants=[(3,'native','#2563eb','3 UAVs'),(4,'native','#e67e22','4 UAVs')]
            if case_no==4:
                variants.append((4,'preserve_base','#8b5cf6','4 UAVs, fixed base'))
            for fleet,layout,color,label in variants:
                info=json.loads((root/'data'/f'case{case_no}_r{sample}_n{fleet}_{layout}.json').read_text())
                points=np.asarray([t['positions'][-1] for t in info['trajectory']])
                all_points.append(points)
                ax.plot(points[:,0],points[:,1],color=color,lw=1.7,
                        ls='--' if layout=='preserve_base' else '-',label=label)
                if not info['strict_success']:
                    ax.scatter(*points[-1],marker='x',s=65,color=color,zorder=6)
            bounds=np.vstack(all_points+[np.asarray([case['start'],case['goal']])])
            low,high=bounds.min(0)-20,bounds.max(0)+20
            for x,y,radius in case['obstacles']:
                if low[0]-radius<x<high[0]+radius and low[1]-radius<y<high[1]+radius:
                    ax.add_patch(Circle((x,y),radius,color='#94a3b8',alpha=.65,lw=0))
            ax.scatter(*case['start'],marker='o',s=38,color='#0f172a',zorder=8,label='Start')
            ax.scatter(*case['goal'],marker='*',s=110,color='#16a34a',zorder=8,label='Goal')
            ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),xlabel='x (m)',ylabel='y (m)',
                   title=f'Case {case_no}, sample r{sample}')
            ax.set_aspect('equal',adjustable='box')
            ax.grid(alpha=.15)
            ax.legend(fontsize=7,loc='best')
    fig.suptitle('Frozen v9 best576, UAV survey first (30 s): vehicle routes; x = unsuccessful termination',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.98))
    fig.savefig(root/'ground_routes.png',dpi=160)
    plt.close(fig)
    print(json.dumps(dict(report=str(root/'report.md'),groups=comparison['groups'],
        paired_travel_change_pct=comparison['paired_travel_change_pct'],
        paired_time_change_pct=comparison['paired_time_change_pct']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
