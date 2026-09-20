"""把一次未进入 LLM 的候选模板及道路角点超限量绘制出来。"""
import json
import math
import pickle
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from policies.ego_profile import project

batch=Path(sys.argv[1]); out=Path(sys.argv[2])
root=batch/'smoke/single/movies/scenario_000'
records=[json.loads(line) for line in (root/'execution_trace.jsonl').open()]
r=next(x for x in records if x.get('kind')=='state' and x['step']==40)
with open('/home2/zhaoyx/scenario-dreamer/metadata/simulation_environment_datasets/scenario_dreamer_waymo_200m_pickles/6_5.pkl','rb') as f: data=pickle.load(f)
lanes=[np.asarray(line)[:,:2] for line in data['lanes'] if len(line)>1]
ego=np.asarray(r['ego'],float); states=np.asarray(r['agents'],float); active=np.asarray(r['active'],bool)
forward=np.array([math.cos(ego[4]),math.sin(ego[4])]); lateral=np.array([-forward[1],forward[0]])
fig,axes=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
colors=['#cf4b55','#9b59b6','#e69f00','#1f77b4']; metrics=[]
for lane in lanes:
    axes[0].plot(lane[:,0],lane[:,1],color='#bbc4cb',lw=.8)
axes[0].scatter(ego[0],ego[1],marker='*',s=140,c='black',label='ego at step 40')
for i in np.flatnonzero(active):
    s=states[i]; relative=s[:2]-ego[:2]; along=relative@forward; side=relative@lateral
    axes[0].scatter(s[0],s[1],s=65,label=f'BG {i} start')
    if not (5<along<45 and abs(side)<=7 and abs(math.atan2(math.sin(s[4]-ego[4]),math.cos(s[4]-ego[4])))<=.7): continue
    speed=np.linalg.norm(s[2:4]);
    if speed<1: continue
    t=np.arange(4,dtype=float); proposals=[]
    if abs(side)<1.5:
        for decel in (1.,2.,.25,.5) if speed<3 else (1.,2.):
            if speed-decel*3>=0:
                proposals.append((f'BG {i} slow_down {decel}',s[:2]+(speed*t-.5*decel*t*t)[:,None]*forward))
    else:
        blend=(t/3)**2*(3-2*t/3)
        anchors=s[:2]+t[:,None]*s[2:4]-blend[:,None]*side*lateral
        proposals.append((f'BG {i} cut_in',anchors))
        if abs(math.atan2(math.sin(s[4]-ego[4]),math.cos(s[4]-ego[4])))>.12:
            proposals.append((f'BG {i} merge',anchors))
    for name,anchors in proposals:
        times=np.linspace(0,3,61)
        pos=np.column_stack([np.interp(times,np.arange(4),anchors[:,j]) for j in range(2)])
        velocity=np.diff(anchors,axis=0); segment=np.minimum(times.astype(int),2)
        yaw=np.arctan2(velocity[segment,1],velocity[segment,0]); maximum=0; at=0
        for a,b in ((1,1),(1,-1),(-1,1),(-1,-1)):
            corners=pos+a*s[5]/2*np.column_stack((np.cos(yaw),np.sin(yaw)))+b*s[6]/2*np.column_stack((-np.sin(yaw),np.cos(yaw)))
            distances=np.min(np.stack([project(corners,lane) for lane in lanes]),axis=0)
            j=int(np.argmax(distances))
            if distances[j]>maximum: maximum=float(distances[j]); at=float(times[j])
        metrics.append(dict(candidate=name,target=int(i),max_corner_distance_m=maximum,time_s=at,road_limit_m=1.8))
        axes[0].plot(pos[:,0],pos[:,1],lw=2,label=name)
        axes[0].scatter(anchors[:,0],anchors[:,1],s=16)
axes[0].set(xlabel='World X (m)',ylabel='World Y (m)',title='Screened sparse candidate anchors (no LLM call)')
axes[0].axis('equal'); axes[0].legend(fontsize=7); axes[0].grid(alpha=.2)
axes[1].barh([x['candidate'] for x in metrics],[x['max_corner_distance_m'] for x in metrics],color=colors[:len(metrics)])
axes[1].axvline(1.8,color='black',ls='--',label='Road limit 1.8 m')
axes[1].set(xlabel='Maximum predicted vehicle-corner distance (m)',title='Candidate road screening',xlim=(0,max([x['max_corner_distance_m'] for x in metrics]+[1.8])+.3))
for j,x in enumerate(metrics): axes[1].text(x['max_corner_distance_m']+.02,j,f"{x['max_corner_distance_m']:.3f} m",va='center')
axes[1].legend()
fig.suptitle('Scene 6_5.pkl, step 40: candidate templates rejected before LLM/Diffusion')
out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=150)
print(json.dumps(metrics,ensure_ascii=False))
