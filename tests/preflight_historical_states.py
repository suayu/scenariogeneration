"""只读扫描冻结 20 场景的历史状态，寻找同刻可行画像候选与合法道路初始几何。"""
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np

from policies.ego_profile import CandidateBuilder, project


class EmptyKnowledge:
    def retrieve(self, condition, family):
        return dict(attempts=0, measured_executions=0, failure_count=0,
                    vulnerability_estimate=.5, uncertainty_interval=[0, 1])


root = Path('/home2/zhaoyx/scenario-dreamer')
base = root/'experiments/riskweaver_target_formal_20260913'
manifest = json.loads((base/'manifest.json').read_text())
builder = CandidateBuilder(SimpleNamespace())
rows = []
for index, source in enumerate(manifest['scenario_files']):
    with open(source, 'rb') as handle:
        scenario = pickle.load(handle)
    lanes = [np.asarray(l, float) for l in scenario['lanes'] if len(l) > 1]
    trace = base/'movies'/f'scenario_{index:03d}'/'attempt_0'/'execution_trace.jsonl'
    if not trace.is_file():
        continue
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    for state in records:
        if state.get('kind') != 'state' or state['step'] > 80 or state['step'] % 5:
            continue
        agents = np.asarray(state['agents'], float)
        active = np.asarray(state['active'], bool)
        env = SimpleNamespace(ego_state=np.asarray(state['ego'], float),
                              data_dict={'agent': [agents]}, agent_active=active,
                              scenario_dict={'route': scenario['route'], 'lanes': lanes},
                              get_static_obstacles=lambda: [])
        candidates, rejected = builder.build(env, {}, EmptyKnowledge())
        if not candidates:
            continue
        worst = 0.
        for agent in agents[active]:
            x,y,yaw,length,width = agent[0],agent[1],agent[4],agent[5],agent[6]
            ahead = np.array([np.cos(yaw),np.sin(yaw)])
            side = np.array([-ahead[1],ahead[0]])
            for forward in (-1,1):
                for lateral in (-1,1):
                    corner = np.array([x,y])+forward*length/2*ahead+lateral*width/2*side
                    worst = max(worst,float(min(project(corner,lane) for lane in lanes)))
        rows.append(dict(index=index, scene=Path(source).name, step=state['step'],
                         candidates=len(candidates), targets=[c['target_id'] for c in candidates],
                         max_initial_corner_m=round(worst,4), initial_road_pass=worst<=1.8,
                         active=int(active.sum()), rejected=rejected))
print(json.dumps(rows,ensure_ascii=False))
