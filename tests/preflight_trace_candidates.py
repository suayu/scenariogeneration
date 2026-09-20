"""从既有轨迹只读筛查后续帧的画像候选与拒绝原因。"""
import glob
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from policies.ego_profile import CandidateBuilder

class EmptyKnowledge:
    def retrieve(self,*args):
        return dict(attempts=0,measured_executions=0,failure_count=0,vulnerability_estimate=.5,uncertainty_interval=[0,1])

root=Path('/home2/zhaoyx/scenario-dreamer')
manifest=json.loads((root/'experiments/riskweaver_target_formal_20260913/manifest.json').read_text())
builder=CandidateBuilder(SimpleNamespace(low_speed_decelerations_mps2=[.25,.5]))
rows=[]
for index,source in enumerate(manifest['scenario_files']):
    with open(source,'rb') as handle: data=pickle.load(handle)
    path=root/f'experiments/riskweaver_target_formal_20260913/movies/scenario_{index:03d}/attempt_0/execution_trace.jsonl'
    if not path.exists(): continue
    states=[r for r in map(json.loads,path.open()) if r.get('kind')=='state' and r['step']>=10 and r['step']%10==0]
    for r in states:
        agents=np.asarray(r['agents'],float)
        env=SimpleNamespace(ego_state=np.asarray(r['ego'],float),data_dict={'agent':[agents]},agent_active=np.asarray(r['active'],bool),
                            scenario_dict={'route':data['route'],'lanes':data['lanes']},get_static_obstacles=lambda:[])
        try:
            candidates,rejected=builder.build(env,{},EmptyKnowledge())
            rows.append(dict(scene=index,source=Path(source).name,step=r['step'],active=int(env.agent_active.sum()),candidates=len(candidates),rejected=rejected))
        except Exception as error:
            rows.append(dict(scene=index,step=r['step'],error=type(error).__name__))
print(json.dumps(rows,ensure_ascii=False))
