"""可视化真实起点的道路违规角点，不推测缺失未来轨迹。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

root = Path('/home2/zhaoyx/scenario-dreamer')
arm = root/'experiments/riskweaver_projection_smoke_20260917_r1/smoke/single'
records = [json.loads(line) for line in (arm/'movies/scenario_000/execution_trace.jsonl').read_text().splitlines()]
state = [r for r in records if r.get('kind') == 'state' and r['step'] == 10][-1]
lanes = json.loads((arm/'carla/initial_data/scenario_0000_initial.json').read_text())['road_network']
car = np.asarray(state['agents'][1])
yaw = car[4]
rotation = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
corners = np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*car[5:7]/2@rotation.T+car[:2]
fig, ax = plt.subplots(figsize=(9,7), constrained_layout=True)
for line in lanes:
    line = np.asarray(line)[:,:2]
    ax.plot(line[:,0],line[:,1],color='#9aa9b5',lw=1)
ax.add_patch(Polygon(corners,facecolor='#eabf85',edgecolor='#4b3c2d',alpha=.8))
values = []
for corner in corners:
    best_distance, nearest = float('inf'), None
    for line in lanes:
        line = np.asarray(line)[:,:2]
        delta = np.diff(line,axis=0)
        length2 = np.sum(delta*delta,axis=1)
        valid = length2>1e-10
        start, delta, length2 = line[:-1][valid], delta[valid], length2[valid]
        t = np.clip(np.sum((corner-start)*delta,axis=1)/length2,0,1)
        p = start+t[:,None]*delta
        d = np.linalg.norm(p-corner,axis=1)
        if len(d) and d.min()<best_distance:
            best_distance, nearest = float(d.min()), p[np.argmin(d)]
    values.append(best_distance)
    color = '#cf3434' if best_distance>1.8 else '#249574'
    ax.scatter(*corner,c=color,s=55,zorder=5)
    ax.plot([corner[0],nearest[0]],[corner[1],nearest[1]],'--',color=color)
    ax.annotate(f'{best_distance:.4f} m',corner,xytext=(9,9),textcoords='offset points',color=color,fontsize=11)
ax.text(car[0],car[1],'BG 1',ha='center',fontsize=12)
ax.set_xlim(car[0]-6,car[0]+6); ax.set_ylim(car[1]-6,car[1]+6)
ax.set_aspect('equal'); ax.set_xlabel('World X (m)'); ax.set_ylabel('World Y (m)')
ax.set_title('Actual state at rejection: step 10\nBG 1 corner-to-lane distance; limit = 1.8000 m')
ax.text(.02,.02,'Red corners violate the original road gate.\nFuture trajectory repair cannot change this true starting state.',transform=ax.transAxes,
        fontsize=10,bbox=dict(facecolor='white',alpha=.9,edgecolor='#cccccc'))
out = root/'experiments/rejection_audit_20260917_r1/figures/road_initial_violation.png'
fig.savefig(out,dpi=160)
print(json.dumps(dict(figure=str(out),corner_distances_m=values,threshold_m=1.8)))
