"""把未形成 LLM 攻击的前置筛查指标可视化。"""
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

batch,output=map(Path,sys.argv[1:3])
trace=next((batch/'smoke/single/movies').glob('scenario_*/execution_trace.jsonl'))
records=[json.loads(line) for line in trace.read_text().splitlines()]
states={r['step']:r for r in records if r.get('kind')=='state'}
queries=[r for r in records if r.get('kind')=='profile_ranking_input']
steps=[r['step'] for r in queries]
speed=[];front=[];count=[]
for query in queries:
    s=states[query['step']]
    ego=np.asarray(s['ego']); bg=np.asarray(s['agents'][2])
    speed.append(float(np.linalg.norm(bg[2:4])))
    front.append(float((bg[:2]-ego[:2])@np.array([math.cos(ego[4]),math.sin(ego[4])])))
    count.append(len(query['context']['candidates']))
fig,(ax,other)=plt.subplots(2,1,figsize=(10,7),sharex=True,constrained_layout=True)
ax.plot(steps,speed,'o-',label='BG 2 speed')
ax.axhline(3,color='#d34d54',linestyle='--',label='3 m/s needed for 1 m/s² over 3 s')
ax.axhline(1,color='#e4a437',linestyle=':',label='1 m/s target filter')
ax.set(ylabel='Speed (m/s)',title='No slow-down candidate: target speed below template requirement')
ax.legend();ax.grid(alpha=.2)
other.plot(steps,front,'s-',color='#3676ae',label='BG 2 ahead of ego')
other.axhspan(5,45,alpha=.12,color='#58a58b',label='Target range 5–45 m')
other.set(xlabel='Simulation frame',ylabel='Forward distance (m)',title='Target was in range; speed/template blocked the proposal')
other.legend();other.grid(alpha=.2)
fig.suptitle('Route-weight experiment: all six planning points had 0 structured candidates; no LLM attack trajectory')
output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(output,dpi=150)
print(json.dumps(dict(steps=steps,speeds_mps=speed,forward_m=front,candidate_counts=count)))
