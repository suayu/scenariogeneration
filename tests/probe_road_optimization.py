"""冻结第 20 帧样本的离线软约束搜索；最终只按原硬门槛判定。"""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch

from policies.rejection_audit import joint_metrics
from policies.trajectory_projection import project_positions

root = Path('/home2/zhaoyx/scenario-dreamer')
batch = root/'experiments/riskweaver_deferred_samples_1789670505498901664'
data = np.load(next((batch/'joint_samples').glob('joint_20_*.npz')))
trace = next((batch/'smoke/single/movies').glob('scenario_*/execution_trace.jsonl'))
state = next(r for r in map(json.loads,trace.read_text().splitlines()) if r.get('kind')=='state' and r['step']==20)
road_file = next((batch/'smoke/single/carla/initial_data').glob('*.json'))
lanes = np.asarray(json.loads(road_file.read_text())['road_network'],float)
starts = lanes[:,:-1,:].reshape(-1,2)
ends = lanes[:,1:,:].reshape(-1,2)
valid = np.linalg.norm(ends-starts,axis=1)>1e-6
starts,ends=starts[valid],ends[valid]
limits=dict(max_speed=20.,max_acceleration=6.,max_jerk=12.,max_step_distance=2.)
env=SimpleNamespace(data_dict={'agent':[np.asarray(state['agents'])]},scenario_dict={'lanes':lanes},dt=float(data['dt']))
pipeline=SimpleNamespace(dynamic_limits=limits,builder=SimpleNamespace(half_width=1.8))
sample=5
projected=[]; yaws=[]
for row,agent_id in enumerate(data['agent_ids']):
    p,v,y,a=project_positions(data['positions'][sample,row],data['states'][agent_id],env.dt,limits,max_deviation=.25)
    projected.append(p); yaws.append(y[:,0])
base=np.asarray(projected)
slot=int(np.flatnonzero(data['agent_ids']==1)[0])
initial=np.asarray(data['states'][1],float)
raw=np.asarray(data['positions'][sample,slot],float)
# 只保留轨迹周围的原始道路线段；最终仍由完整道路数据硬校验。
near=np.min(np.linalg.norm(((starts+ends)/2)[:,None,:]-base[slot][None,:,:],axis=-1),axis=1)<8
starts,ends=starts[near],ends[near]
dt=env.dt
dtype=torch.float64
start=torch.tensor(initial[:2],dtype=dtype)
v0=torch.tensor(initial[2:4],dtype=dtype)
reference=torch.tensor(raw,dtype=dtype)
velocity=torch.nn.Parameter(torch.tensor(np.diff(np.vstack((initial[:2],base[slot])),axis=0)/dt,dtype=dtype))
segments_a=torch.tensor(starts,dtype=dtype)
segments_d=torch.tensor(ends-starts,dtype=dtype)
segment_norm=(segments_d*segments_d).sum(-1)
length,width=initial[5:7]


def geometry(v):
    positions=start+torch.cumsum(v*dt,dim=0)
    yaw=torch.atan2(v[:,1],v[:,0])
    all_pos=torch.cat((start[None],positions),dim=0)
    all_yaw=torch.cat((torch.tensor([initial[4]],dtype=dtype),yaw),dim=0)
    mid_pos=(all_pos[1:]+all_pos[:-1])/2
    mid_yaw=(all_yaw[1:]+all_yaw[:-1])/2
    points=torch.cat((all_pos,mid_pos),dim=0)
    angle=torch.cat((all_yaw,mid_yaw),dim=0)
    ahead=torch.stack((torch.cos(angle),torch.sin(angle)),dim=1)
    side=torch.stack((-torch.sin(angle),torch.cos(angle)),dim=1)
    corners=torch.stack([points+a*length/2*ahead+b*width/2*side
                         for a,b in ((1,1),(1,-1),(-1,1),(-1,-1))]).reshape(-1,2)
    return positions,yaw,corners


def road_distance(corners):
    delta=corners[:,None,:]-segments_a[None,:,:]
    fraction=torch.clamp((delta*segments_d[None,:,:]).sum(-1)/segment_norm[None,:],0,1)
    residual=delta-fraction[:,:,None]*segments_d[None,:,:]
    return torch.sqrt((residual*residual).sum(-1)+1e-12).min(dim=1).values


road_weight=float(sys.argv[1]) if len(sys.argv)>1 else 1000.
optimizer=torch.optim.Adam([velocity],lr=.006)
best=None
for iteration in range(1500):
    optimizer.zero_grad()
    p,y,c=geometry(velocity)
    distances=road_distance(c)
    a=torch.diff(torch.cat((v0[None],velocity)),dim=0)/dt
    j=torch.diff(a,dim=0)/dt
    speed=torch.linalg.vector_norm(velocity,dim=1)
    accel=torch.linalg.vector_norm(a,dim=1)
    jerk=torch.linalg.vector_norm(j,dim=1)
    deviation=torch.linalg.vector_norm(p-reference,dim=1)
    loss=2*((p-reference)**2).mean()+road_weight*torch.relu(distances-1.79).square().mean()
    loss=loss+1000*torch.relu(speed-19.9).square().mean()
    loss=loss+1000*torch.relu(accel-5.9).square().mean()+1000*torch.relu(jerk-11.9).square().mean()
    loss=loss+1000*torch.relu(deviation-.249).square().mean()
    loss.backward()
    optimizer.step()
    if iteration%25==0 or iteration==1499:
        p_np=p.detach().numpy(); y_np=y.detach().numpy()
        positions=base.copy(); yaw_all=np.asarray(yaws).copy()
        positions[slot]=p_np; yaw_all[slot]=y_np
        joint=SimpleNamespace(agent_ids=data['agent_ids'],positions_global=positions,yaws_global=yaw_all)
        metrics=joint_metrics(env,pipeline,joint)
        road=max(m['road']['max_corner_distance_m'] for m in metrics)
        dynamic=max(m['dynamics']['max_jerk']['value'] for m in metrics)
        shift=np.linalg.norm(p_np-raw,axis=1).max()
        score=(max(0,road-1.8)+max(0,dynamic-12)+max(0,shift-.25),road)
        if best is None or score<best[0]:
            best=(score,dict(iteration=iteration,road_max_m=road,jerk_max=dynamic,
                            raw_deviation_m=shift,per_vehicle=metrics))
print(json.dumps(best[1],allow_nan=False))
