"""四组隔离校准的规划耗时、真实请求数与实际执行对照图。"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

batch=Path(sys.argv[1])
rows=json.loads((batch/'calibration_comparison.json').read_text())['rows']
names=['Single','Critic','Parallel','Memory']
seconds=np.asarray([r['planner_seconds'] for r in rows])
calls=np.asarray([r['actual_api_calls'] for r in rows])
frames=np.asarray([r['attack_executed_frames'] for r in rows])
fig,axes=plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True)
colors=['#517fa8','#d59a4c','#7757a8','#62a386']
for ax,values,title,ylabel in zip(axes,(seconds,calls,frames),
                                  ('Planner latency','Actual API calls','Executed attack frames'),
                                  ('Seconds','Requests','Frames')):
    bars=ax.bar(names,values,color=colors)
    ax.set(title=title,ylabel=ylabel)
    ax.tick_params(axis='x',rotation=20)
    ax.set_ylim(0,max(values)*1.25 if max(values) else 1)
    for bar,value in zip(bars,values):
        ax.text(bar.get_x()+bar.get_width()/2,value+.03*max(values),f'{value:.2f}' if title=='Planner latency' else str(int(value)),
                ha='center',fontsize=9)
fig.suptitle('Derived single-attacker calibration: all modes D=0.0962, zero background collisions')
output=batch/'calibration_comparison.png'
fig.savefig(output,dpi=150)
print(output)
