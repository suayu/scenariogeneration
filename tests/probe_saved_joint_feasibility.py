"""逐个检查冻结 Diffusion 样本修正后的动力学和道路硬约束。"""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from policies.rejection_audit import joint_metrics
from policies.trajectory_projection import project_positions
from policies.joint_safety import NoSafeJointCandidate

root = Path('/home2/zhaoyx/scenario-dreamer')
batch = Path(sys.argv[1]) if len(sys.argv)>1 else root/'experiments/riskweaver_joint_diagnostic_1789627892542948329'
packs = list((batch/'joint_samples').glob('joint_*.npz'))
pack = max(packs, key=lambda path: int(path.name.split('_')[1]))
data = np.load(pack)
trace = next((batch/'smoke/single/movies').glob('scenario_*/execution_trace.jsonl'))
records = [json.loads(line) for line in trace.read_text().splitlines()]
sample_step = int(pack.name.split('_')[1])
state = next(row for row in records if row.get('kind') == 'state' and row['step'] == sample_step)
road_file = next((batch/'smoke/single/carla/initial_data').glob('*.json'))
lanes = json.loads(road_file.read_text())['road_network']
env = SimpleNamespace(data_dict={'agent': [np.asarray(state['agents'])]},
                      scenario_dict={'lanes': lanes}, dt=float(data['dt']))
limits = dict(max_speed=20., max_acceleration=6., max_jerk=12., max_step_distance=2.)
pipeline = SimpleNamespace(dynamic_limits=limits,builder=SimpleNamespace(half_width=1.8))
rows = []
for sample in range(1, len(data['positions'])):
    projected, yaws, failure = [], [], None
    for row, agent_id in enumerate(data['agent_ids']):
        try:
            p, v, y, audit = project_positions(data['positions'][sample,row],
                data['states'][agent_id], float(data['dt']), limits, max_deviation=.25)
            projected.append(p)
            yaws.append(y)
        except NoSafeJointCandidate as error:
            failure = str(error)
            break
    if failure:
        rows.append(dict(sample=sample, projection_failure=failure))
        continue
    joint = SimpleNamespace(agent_ids=data['agent_ids'], positions_global=np.asarray(projected),
                            yaws_global=np.asarray(yaws))
    metrics = joint_metrics(env,pipeline,joint)
    rows.append(dict(sample=sample, road_pass=all(not m['road']['violated'] for m in metrics),
                     dynamics_pass=all(not any(d['violated'] for d in m['dynamics'].values()) for m in metrics),
                     road_max=[round(m['road']['max_corner_distance_m'],3) for m in metrics],
                     initial_road_max=[round(m['road']['initial_max_corner_distance_m'],3) for m in metrics]))
print(json.dumps(rows))
