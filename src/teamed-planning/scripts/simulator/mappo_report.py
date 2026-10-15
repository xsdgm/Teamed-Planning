"""Create an offline, scrubbable 2D experiment replay from saved results."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('results',type=Path)
    args=parser.parse_args()
    data=json.loads(args.results.read_text(encoding='utf-8'))
    labels=dict(mappo='MAPPO',greedy='贪心',cadmm_controlled='统一接口 C-ADMM',cadmm_original='原版 C-ADMM')
    rows=[]
    summary=['# MAPPO 首轮 2D 实验','',
        f"训练：{data['config'].get('training_episodes_total',len(data['training']))} 回合；每回合 {data['config']['steps']} 秒；测试种子：{data['config']['test_seeds']}。",'',
        '| 算法 | 置信度面积增益 m² | 重点正增益占比 | 外部首次观测 m² | 全员同时移动比例 | 间距违约步数 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for algorithm,label in labels.items():
        results=[v for v in data['evaluation'] if v['algorithm']==algorithm]
        def avg(k): return mean(v[k] for v in results)
        confidence=avg('task_gain')+avg('outside_gain')
        vals=[f'{confidence:.1f}',f'{avg("task_share"):.1%}',f'{avg("outside_new"):.1f}',
              f'{avg("simultaneous_moving"):.1%}',f'{avg("spacing_violation"):.1f}']
        summary.append('| '+label+' | '+' | '.join(vals)+' |')
        rows.append('<tr><td>'+label+'</td>'+''.join('<td>'+v+'</td>' for v in vals)+'</tr>')
    summary.extend(['','数值为测试场景均值，单个训练随机种子；不构成收敛或显著优越结论。',
        '这是 120×120m、三机、固定地面任务源、1Hz 控制、24m 方形相机、95m 直接上传的初步实验。',
        '奖励只统计真实接收图包携带的训练审计证据，假定二值传感器可靠度 0.8，未做真实传感器校准。',
        '审计证据仅供训练奖励，执行端不读取；尚未实现中继证据转发、动态图像分割误差或完整鲁棒性验收。',
        '原版与统一接口版外围规则不同，优化目标也与 MAPPO 不同；不能将差异全部归因于优化器。',''])
    args.results.with_name('summary.md').write_text('\n'.join(summary),encoding='utf-8')
    payload=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MAPPO 2D 实验</title>
<style>body{margin:28px;background:#101b2c;color:#e7eef8;font:16px system-ui}main{max-width:1200px;margin:auto}h1{font-size:27px}p{color:#aabbd0;line-height:1.7}table{border-collapse:collapse;width:100%;margin:24px 0}td,th{padding:12px;text-align:right;border-bottom:1px solid #34445b}td:first-child,th:first-child{text-align:left}canvas{background:#18283b;width:100%;max-width:640px;border-radius:14px}select,button{padding:9px;margin:5px;border-radius:7px}input{width:50%}.layout{display:flex;gap:24px;flex-wrap:wrap}.note{max-width:450px}</style>
<main><h1>多无人机 MAPPO · 2D 初步实验</h1><p>共享 Actor + 集中 Critic；三架无人机同时推进。固定地面任务源，直接上传。测试地图与目标独立于训练种子。</p>
<table><thead><tr><th>算法</th><th>置信度增益 m²</th><th>重点占比</th><th>外部首次观测 m²</th><th>全员同时移动</th><th>间距违约步数</th></tr></thead><tbody>ROWS</tbody></table>
<div><select id="scenario"></select><button id="play">播放 / 暂停</button><input id="time" type="range" min="0" value="0"><span id="clock"></span></div>
<div class="layout"><canvas id="map" width="640" height="640"></canvas><div class="note"><h2>轨迹回放</h2><p>蓝、橙、粉分别为三架无人机；白色方块为地面任务源；黄色星号为目标。绿圈是独立评估用场景障碍，仅用于画图与传感器物理，不作为策略输入。</p><pre id="stats"></pre><p>单个训练种子、三个测试场景的初步结果，尚不足以证明收敛或优于基线。置信度采用假设的传感器可靠度，尚未完成独立校准。没有验证真实飞行与中继通信。</p></div></div></main>
<script>const data=PAYLOAD;const menu=document.querySelector('#scenario'),slider=document.querySelector('#time'),canvas=document.querySelector('#map'),ctx=canvas.getContext('2d');const colors=['#5faaff','#ffb458','#fa7ac4'];let playing=false;for(const key of Object.keys(data.trajectories)){let opt=document.createElement('option');opt.value=key;opt.textContent=key;menu.append(opt)}
function draw(){let s=data.trajectories[menu.value],t=Math.min(+slider.value,s.trace.length-1),z=5;slider.max=s.trace.length-1;ctx.clearRect(0,0,640,640);const pt=p=>[20+p[0]*z,620-p[1]*z];ctx.strokeStyle='#304357';for(let n=0;n<=120;n+=20){ctx.beginPath();ctx.moveTo(20+n*z,20);ctx.lineTo(20+n*z,620);ctx.stroke();ctx.beginPath();ctx.moveTo(20,620-n*z);ctx.lineTo(620,620-n*z);ctx.stroke()}
for(let i=0;i<3;i++){ctx.strokeStyle=colors[i];ctx.lineWidth=2;ctx.beginPath();for(let k=0;k<=t;k++){let p=pt(s.trace[k].positions[i]);k?ctx.lineTo(...p):ctx.moveTo(...p)}ctx.stroke();let p=pt(s.trace[t].positions[i]);ctx.fillStyle=colors[i];ctx.beginPath();ctx.arc(...p,6,0,Math.PI*2);ctx.fill();ctx.fillStyle='#fff';ctx.fillText('D'+i,p[0]+8,p[1])}let ground=pt([15,60]);ctx.fillStyle='white';ctx.fillRect(ground[0]-5,ground[1]-5,10,10);let goal=pt(s.goal);ctx.fillStyle='#ffe177';ctx.font='24px system-ui';ctx.fillText('★',goal[0]-10,goal[1]+8);document.querySelector('#clock').textContent=(t+1)+' 秒';let v=s.trace[t];document.querySelector('#stats').textContent='本步奖励: '+v.reward.toFixed(3)+'\\n重点增益: '+v.task_gain.toFixed(1)+' m²\\n外部增益: '+v.outside_gain.toFixed(1)+' m²\\n团队移动: '+v.distance.toFixed(1)+' m\\n全员移动: '+(v.simultaneous_moving?'是':'否')+'\\n间距违约: '+v.spacing_violation}
menu.onchange=()=>{slider.value=0;draw()};slider.oninput=draw;document.querySelector('#play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){slider.value=(+slider.value+1)%(+slider.max+1);draw()}},220);draw();</script></html>'''
    # Obstacles are omitted from replay until their exact geometry is stored;
    # avoid reconstructing a scene from a potentially different RNG version.
    html=html.replace('绿圈是独立评估用场景障碍，仅用于画图与传感器物理，不作为策略输入。','回放显示保存的实际轨迹；未绘制未保存的障碍物几何。')
    html=html.replace('ROWS',''.join(rows)).replace('PAYLOAD',payload)
    args.results.with_name('replay.html').write_text(html,encoding='utf-8')


if __name__=='__main__':
    main()
