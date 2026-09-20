"""只读验证低速减速模板，候选仍经现有完整安全筛查。"""
import json
from pathlib import Path
from types import SimpleNamespace
import math
import pickle
import numpy as np

from policies.ego_profile import CandidateBuilder

root=Path('/home2/zhaoyx/scenario-dreamer')
batch=root/'experiments/riskweaver_route_weight_5_1789671420336890397'
trace=next((batch/'smoke/single/movies').glob('scenario_*/execution_trace.jsonl'))
records=[json.loads(line) for line in trace.read_text().splitlines()]
road_file=next((batch/'smoke/single/carla/initial_data').glob('*.json'))
lanes=json.loads(road_file.read_text())['road_network']
with (root/'metadata/simulation_environment_datasets/scenario_dreamer_waymo_200m_pickles/6_5.pkl').open('rb') as handle:
    route=pickle.load(handle)['route']
builder=CandidateBuilder(SimpleNamespace())
rows=[]
for query in (r for r in records if r.get('kind')=='profile_ranking_input'):
    state=next(r for r in records if r.get('kind')=='state' and r['step']==query['step'])
    ego=np.asarray(state['ego'],float); agents=np.asarray(state['agents'],float)
    env=SimpleNamespace(ego_state=ego,data_dict={'agent':[agents]},
                        agent_active=np.asarray(state['active'],bool),
                        scenario_dict={'route':route, 'lanes':lanes},
                        get_static_obstacles=lambda: [])
    target=2
    forward=np.array([math.cos(ego[4]),math.sin(ego[4])])
    speed=float(np.linalg.norm(agents[target,2:4]))
    t=np.arange(4,dtype=float)
    for decel in (.25,.5,.75):
        if speed-decel*3<0:
            continue
        anchors=agents[target,:2]+(speed*t-.5*decel*t*t)[:,None]*forward
        info,reason=builder.screen(env,target,anchors,lanes)
        rows.append(dict(step=query['step'],speed=speed,deceleration=decel,reason=reason,
                         ttc_s=info.get('estimated_ttc_s') if info else None))
print(json.dumps(rows))
