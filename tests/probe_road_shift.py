"""冻结候选的只读道路修正可行性网格；不修改仿真或安全阈值。"""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from policies.rejection_audit import joint_metrics
from policies.trajectory_projection import project_positions

root = Path('/home2/zhaoyx/scenario-dreamer')
batch = root/'experiments/riskweaver_deferred_samples_1789670505498901664'
pack = next((batch/'joint_samples').glob('joint_20_*.npz'))
data = np.load(pack)
trace = next((batch/'smoke/single/movies').glob('scenario_*/execution_trace.jsonl'))
state = next(row for row in map(json.loads, trace.read_text().splitlines())
             if row.get('kind')=='state' and row['step']==20)
road_file = next((batch/'smoke/single/carla/initial_data').glob('*.json'))
lanes = json.loads(road_file.read_text())['road_network']
limits = dict(max_speed=20., max_acceleration=6., max_jerk=12., max_step_distance=2.)
env = SimpleNamespace(data_dict={'agent':[np.asarray(state['agents'])]},
                      scenario_dict={'lanes':lanes}, dt=float(data['dt']))
pipeline = SimpleNamespace(dynamic_limits=limits,builder=SimpleNamespace(half_width=1.8))
out=[]
diagnostics=[]
chosen = [int(sys.argv[1])] if len(sys.argv)>1 else range(1,len(data['positions']))
for sample in chosen:
    positions,yaws=[],[]
    for slot,agent_id in enumerate(data['agent_ids']):
        p,v,y,_=project_positions(data['positions'][sample,slot],data['states'][agent_id],
                                  env.dt,limits,max_deviation=.25)
        positions.append(p); yaws.append(y[:,0])
    original=np.asarray(positions)
    bg1=int(np.flatnonzero(data['agent_ids']==1)[0])
    for shift in np.arange(-.6,.601,.05):
        for ramp_seconds in (.5,1.,1.5,2.,3.):
            p=original.copy()
            n=len(p[bg1]); t=np.arange(1,n+1)*env.dt
            smooth=(1-np.cos(np.pi*np.minimum(t/ramp_seconds,1)))/2
            p[bg1,:,0]+=shift*smooth
            v=np.diff(np.vstack((data['states'][1,:2],p[bg1])),axis=0)/env.dt
            yy=np.arctan2(v[:,1],v[:,0])
            joint=SimpleNamespace(agent_ids=data['agent_ids'],positions_global=p,
                yaws_global=np.asarray([yy if k==bg1 else yaws[k] for k in range(len(yaws))]))
            metrics=joint_metrics(env,pipeline,joint)
            road_pass=all(not m['road']['violated'] for m in metrics)
            dynamics_pass=all(not any(d['violated'] for d in m['dynamics'].values()) for m in metrics)
            diagnostics.append(dict(sample=sample,shift_x_m=round(float(shift),3),ramp_s=ramp_seconds,
                                    road_pass=road_pass,dynamics_pass=dynamics_pass,
                                    max_road_m=round(max(m['road']['max_corner_distance_m'] for m in metrics),3),
                                    max_jerk=round(max(m['dynamics']['max_jerk']['value'] for m in metrics),3)))
            if road_pass and dynamics_pass:
                deviation=float(np.linalg.norm(p-np.asarray(data['positions'][sample]),axis=-1).max())
                out.append(dict(sample=sample,shift_x_m=round(float(shift),3),ramp_s=ramp_seconds,
                                max_deviation_from_raw_m=round(deviation,3),
                                max_road_m=round(max(m['road']['max_corner_distance_m'] for m in metrics),3)))
print(json.dumps(dict(feasible_count=len(out),smallest=sorted(out,key=lambda r:r['max_deviation_from_raw_m'])[:12],
                      road_pass_count=sum(r['road_pass'] for r in diagnostics),
                      dynamics_pass_count=sum(r['dynamics_pass'] for r in diagnostics),
                      best_road=sorted(diagnostics,key=lambda r:r['max_road_m'])[:8])))
