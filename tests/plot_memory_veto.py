"""展示离线记忆组的提案分歧及未执行的锚点。"""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

batch=Path(sys.argv[1]); output=Path(sys.argv[2])
folder=batch/'smoke/parallel_memory/movies/scenario_000'
records=[json.loads(line) for line in (folder/'execution_trace.jsonl').read_text().splitlines()]
state=next(x for x in records if x.get('kind')=='state' and x['step']==0)
pred=next(x for x in records if x.get('kind')=='prediction' and x['step']==0)
trace=next(x['planner_trace'] for x in records if x.get('kind')=='llm_output')
risk=next(x for x in trace['events'] if x['role']=='proposer_risk')
feasibility=next(x for x in trace['events'] if x['role']=='proposer_feasibility')
anchors=np.asarray(risk['plan']['anchors'],float)
fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
ax=axes[0]
ego=np.asarray(state['ego'],float)
ax.scatter(ego[0],ego[1],marker='*',s=180,c='black',label='Ego start')
for i in np.flatnonzero(state['active']):
    a=np.asarray(state['agents'][i],float)
    ax.scatter(a[0],a[1],s=35,c='#8f9aa3')
for i,agent_id in enumerate(pred['joint']['agent_ids']):
    positions=np.asarray(pred['joint']['positions'][i],float)
    if agent_id==risk['plan']['attack_target_id']:
        ax.plot(positions[:,0],positions[:,1],color='#2b77b4',label='Normal Diffusion forecast (not attack-guided)')
ax.plot(anchors[:,0],anchors[:,1],'s--',color='#c64a57',lw=2,label='Risk agent LLM anchors (not executed)')
ax.set(xlabel='World X (m)',ylabel='World Y (m)',title='Proposal at frame 0')
ax.axis('equal'); ax.grid(alpha=.2); ax.legend(fontsize=8)
ax=axes[1]; ax.axis('off')
text=('Risk proposer: hard_brake, target BG 5\n'
      'Feasibility proposer: no attack\n'
      'Reason: ego nearly stationary; interaction judged infeasible\n\n'
      'Selected: no attack (no attack-guided Diffusion)\n'
      'Actual attack frames: 0\n'
      'Measured D: null (no valid attack window)\n'
      f"Planner time: {trace['elapsed_seconds']:.2f} s; API calls: 2; memory hits: {trace['memory_hits']}")
ax.text(.04,.95,text,va='top',fontsize=12,linespacing=1.5,transform=ax.transAxes)
fig.suptitle('Frozen multi-background evaluation: disagreement before attack execution')
output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(output,dpi=150)
print(json.dumps(dict(image=str(output),selected=trace['selected_plan'],memory_hits=trace['memory_hits']),ensure_ascii=False))
