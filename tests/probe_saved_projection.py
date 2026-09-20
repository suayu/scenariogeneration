"""在冻结的失败样本上检验投影，不调用 LLM，也不执行仿真控制。"""
import json
from pathlib import Path
import sys

import numpy as np

from policies.trajectory_projection import project_positions
from policies.joint_safety import NoSafeJointCandidate

source = Path(sys.argv[1])
data = np.load(source)
limits = dict(max_speed=20., max_acceleration=6., max_jerk=12., max_step_distance=2.)
results = []
for index in range(1, len(data['positions'])):
    audits = []
    failure = None
    for row, agent_id in enumerate(data['agent_ids']):
        try:
            _, _, _, audit = project_positions(data['positions'][index, row],
                data['states'][agent_id], float(data['dt']), limits)
            audits.append(dict(agent_id=int(agent_id), **audit))
        except NoSafeJointCandidate as error:
            failure = str(error)
            break
    results.append(dict(sample=index, dynamics_passed=failure is None, failure=failure, vehicles=audits))
print(json.dumps(dict(source=str(source), results=results,
                     limitation='dynamics only; road, background safety and avoidability remain unverified'), indent=2))
