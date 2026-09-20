"""绘制候选在 LLM 调用前被拒绝的中文诊断图；论文图仍由英文脚本生成。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


batch=Path(sys.argv[1])
output=Path(sys.argv[2])
root=batch/'smoke/single/movies/scenario_000'
records=[json.loads(line) for line in (root/'execution_trace.jsonl').open(encoding='utf-8')]
ranking=next(row for row in records if row.get('kind')=='profile_ranking_input')
step=int(ranking['step'])
state=next(row for row in records if row.get('kind')=='state' and int(row['step'])==step)
prediction=next(row for row in records if row.get('kind')=='prediction' and int(row['step'])==step)

font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family']=font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams['axes.unicode_minus']=False
fig,(ax,detail)=plt.subplots(1,2,figsize=(14,6.5),gridspec_kw={'width_ratios':[1.7,1]})
ego=np.asarray(state['ego'],float)
agents=np.asarray(state['agents'],float)
active=np.flatnonzero(np.asarray(state['active'],bool))
route=np.asarray(next(row for row in records if row.get('kind')=='llm_input' and int(row['step'])==step)['state']['route'])
ax.plot(route[:,0],route[:,1],color='#7f8c8d',lw=2,label='自车参考路线')
ax.scatter(ego[0],ego[1],marker='*',s=180,color='#1f9d72',label='自车当前位置')
for agent_id in active:
    ax.scatter(*agents[agent_id,:2],s=70,color='#e67e22')
    ax.annotate(f'背景车 {agent_id}',agents[agent_id,:2])
joint=prediction.get('joint') or {}
for i,(agent_id,positions) in enumerate(zip(joint.get('agent_ids',[]),joint.get('positions',[]))):
    points=np.asarray(positions,float)
    ax.plot(points[:,0],points[:,1],color='#2878b5',alpha=.8,lw=1.6,
            label='无攻击意图的 Diffusion 预测' if i==0 else None)
ax.set_title(f'第 {step} 帧：候选在调用 LLM 前被拒绝')
ax.set_xlabel('世界坐标 X（米）'); ax.set_ylabel('世界坐标 Y（米）')
ax.axis('equal'); ax.grid(alpha=.2); ax.legend(fontsize=9)

detail.axis('off')
rejections=ranking['context'].get('candidate_rejections',{})
profile=ranking['context']['profile']
lines=[
    '拒绝阶段：结构化候选生成器',
    '拒绝原因：outside_ttc_band',
    f"被拒候选数：{rejections.get('outside_ttc_band',0)}",
    '允许 TTC 区间：1.0–4.0 秒',
    f"画像累计帧数：{profile.get('frames',0)}",
    f"传给 LLM 的候选数：{len(ranking['context'].get('candidates',[]))}",
    '真实 LLM API 调用：否',
    'LLM 稀疏锚点：无',
    '攻击 Diffusion 轨迹：无',
    '',
    '蓝线为系统继续运行时的普通 Diffusion 预测，',
    '不是攻击轨迹，且没有被执行为攻击。',
]
detail.text(.02,.98,'\n'.join(lines),va='top',fontsize=12)
fig.tight_layout()
output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(output,dpi=160)
print(json.dumps({'output':str(output),'step':step,'rejections':rejections},ensure_ascii=False))
