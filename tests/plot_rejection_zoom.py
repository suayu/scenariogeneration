"""为拒绝攻击绘制逐车局部轨迹和导致拒绝的指标。"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


batch, output = map(Path, sys.argv[1:3])
root=batch/'smoke/single/movies/scenario_000'
records=[json.loads(line) for line in (root/'execution_trace.jsonl').read_text().splitlines()]
event=next(r for r in records if r.get('kind')=='attack_rejected')
state=next(r for r in records if r.get('kind')=='state' and r['step']==event['step'])
road_file=next((batch/'smoke/single/carla/initial_data').glob('*.json'))
lanes=json.loads(road_file.read_text())['road_network']
ids=event['rejected_joint']['agent_ids']
fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
for row,agent_id in enumerate(ids):
    ax=axes[0,row]
    actual=np.asarray(state['agents'][agent_id])
    raw=np.asarray(event['original_joint']['positions'][row])
    corrected=np.asarray(event['rejected_joint']['positions'][row])
    for lane in lanes:
        line=np.asarray(lane)
        ax.plot(line[:,0],line[:,1],color='#c4ccd4',linewidth=.8,zorder=0)
    ax.scatter(actual[0],actual[1],s=75,color='#e28a39',label='Actual start',zorder=4)
    ax.plot(raw[:,0],raw[:,1],color='#2375bc',label='Raw Diffusion')
    ax.plot(corrected[:,0],corrected[:,1],'--',color='#d34d54',label='Dynamics projected')
    if agent_id==event['attack_intent']['target_id']:
        anchors=np.asarray(event['attack_intent']['anchors'])
        ax.plot(anchors[:,0],anchors[:,1],'s:',color='#8a45ab',label='LLM anchors')
    points=np.vstack((actual[None,:2],raw,corrected))
    low,high=points.min(axis=0)-3,points.max(axis=0)+3
    ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),xlabel='World X (m)',ylabel='World Y (m)',
           title=f'BG {agent_id}: '+('attack target' if agent_id==event['attack_intent']['target_id'] else 'other traffic'))
    ax.set_aspect('equal')
    ax.legend(fontsize=8,loc='best')
    ax.grid(alpha=.15)
metric=event['rejected_metrics']
names=[f"BG {m['agent_id']}" for m in metric]
road=[m['road']['max_corner_distance_m'] for m in metric]
initial=[m['road']['initial_max_corner_distance_m'] for m in metric]
jerk=[m['dynamics']['max_jerk']['value'] for m in metric]
ax=axes[1,0]
x=np.arange(len(ids))
ax.bar(x-.17,initial,width=.33,label='Actual initial corner',color='#7797b3')
ax.bar(x+.17,road,width=.33,label='Max projected corner',color='#d34d54')
ax.axhline(1.8,color='black',linestyle='--',label='Road limit 1.8 m')
for i,value in enumerate(road):
    ax.text(i+.17,value+.025,f'{value:.3f}',ha='center',fontsize=10)
ax.set(xticks=x,xticklabels=names,ylabel='Distance to road centerline (m)',ylim=(0,max(road)+.35),
       title='Rejection: predicted vehicle corner beyond road corridor')
ax.legend(fontsize=8)
ax=axes[1,1]
ax.bar(names,jerk,color='#58a58b')
ax.axhline(12,color='black',linestyle='--',label='Jerk limit 12 m/s³')
for i,value in enumerate(jerk):
    ax.text(i,value+.3,f'{value:.3f}',ha='center')
ax.set(ylabel='Maximum jerk (m/s³)',ylim=(0,14),title='Dynamics check passed')
ax.legend(fontsize=8)
fig.suptitle(f"Step {event['step']} | {event['reason']} | plan and trajectories rejected before execution")
output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(output,dpi=150)
print(json.dumps({'figure':str(output),'reason':event['reason'],'road_m':road,'jerk':jerk}))
