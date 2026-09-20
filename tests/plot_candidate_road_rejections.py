"""从已记录状态重算被道路筛查拒绝的稀疏候选的最大车角距离。"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from policies.ego_profile import project


batch,output=map(Path,sys.argv[1:3])
trace=next((batch/'smoke/single/movies').glob('scenario_*/execution_trace.jsonl'))
records=[json.loads(line) for line in trace.read_text().splitlines()]
road_file=next((batch/'smoke/single/carla/initial_data').glob('*.json'))
lanes=[np.asarray(line) for line in json.loads(road_file.read_text())['road_network']]
states={r['step']:r for r in records if r.get('kind')=='state'}
rows=[]
for query in (r for r in records if r.get('kind')=='profile_ranking_input'):
    step=query['step']; state=states[step]
    ego=np.asarray(state['ego'],float)
    forward=np.array([np.cos(ego[4]),np.sin(ego[4])])
    lateral=np.array([-forward[1],forward[0]])
    for target in np.flatnonzero(state['active']):
        bg=np.asarray(state['agents'][target],float)
        rel=bg[:2]-ego[:2]
        speed=float(np.linalg.norm(bg[2:4]))
        heading=abs(np.arctan2(np.sin(bg[4]-ego[4]),np.cos(bg[4]-ego[4])))
        if not 5<rel@forward<45 or abs(rel@lateral)>7 or heading>.7 or speed<1 or abs(rel@lateral)>=1.5:
            continue
        decels=[1.,2.]+([.25,.5] if speed<3 else [])
        for decel in decels:
            if speed-decel*3<0:
                continue
            t=np.arange(4,dtype=float)
            anchors=bg[:2]+(speed*t-.5*decel*t*t)[:,None]*forward
            times=np.linspace(0,3,61)
            pos=np.column_stack([np.interp(times,t,anchors[:,axis]) for axis in range(2)])
            velocity=np.diff(anchors,axis=0)
            segments=np.minimum(times.astype(int),2)
            yaw=np.arctan2(velocity[segments,1],velocity[segments,0])
            maximum=0.
            for along,across in ((1,1),(1,-1),(-1,1),(-1,-1)):
                corner=pos+along*bg[5]/2*np.column_stack((np.cos(yaw),np.sin(yaw)))+across*bg[6]/2*np.column_stack((-np.sin(yaw),np.cos(yaw)))
                distance=np.min(np.stack([project(corner,lane) for lane in lanes]),axis=0)
                maximum=max(maximum,float(distance.max()))
            rows.append(dict(step=step,target=int(target),deceleration=decel,max_corner_distance_m=maximum,
                             threshold_m=1.8,rejected_by_road=maximum>1.8))
fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
for decel in sorted(set(r['deceleration'] for r in rows)):
    selected=[r for r in rows if r['deceleration']==decel]
    ax.plot([r['step'] for r in selected],[r['max_corner_distance_m'] for r in selected],
            'o-',label=f'{decel:g} m/s² slow-down template')
ax.axhline(1.8,color='black',linestyle='--',label='Road limit 1.8 m')
ax.set(xlabel='Planning frame',ylabel='Maximum candidate corner distance (m)',
       title='Structured attack anchors rejected before LLM: road corridor')
ax.legend();ax.grid(alpha=.2)
output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(output,dpi=150)
print(json.dumps(dict(proposals=len(rows),road_rejected=sum(r['rejected_by_road'] for r in rows),
                      worst=max(rows,key=lambda r:r['max_corner_distance_m']) if rows else None,
                      closest=min(rows,key=lambda r:r['max_corner_distance_m']) if rows else None)))
