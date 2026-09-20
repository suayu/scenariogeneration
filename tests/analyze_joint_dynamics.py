"""严格复用画像门禁差分公式，报告全部样本的失败位置与可行比例。"""
import json
from pathlib import Path
import sys

import numpy as np


directory = Path(sys.argv[1])
rows = []
for path in sorted(directory.glob('joint_*.npz')):
    data = np.load(path)
    states = np.asarray(data['states'][data['agent_ids']], dtype=float)
    positions = np.asarray(data['positions'], dtype=float)
    dt = float(data['dt'])
    velocity = np.diff(np.concatenate((np.broadcast_to(states[None, :, None, :2],
                       (len(positions), len(states), 1, 2)), positions), axis=2), axis=2)/dt
    acceleration = np.diff(np.concatenate((np.broadcast_to(states[None, :, None, 2:4],
                          (len(positions), len(states), 1, 2)), velocity), axis=2), axis=2)/dt
    jerk = np.diff(acceleration, axis=2)/dt
    valid = np.ones(len(positions), dtype=bool)
    diagnostics = {}
    for key, values, limit in [('speed', velocity, 20.), ('acceleration', acceleration, 6.), ('jerk', jerk, 12.)]:
        norms = np.linalg.norm(values, axis=-1)
        peaks = norms.max(axis=(1, 2))
        valid &= peaks <= limit
        index = np.unravel_index(np.argmax(norms), norms.shape)
        diagnostics[key] = dict(peaks=peaks.tolist(), worst_sample=int(index[0]),
                                worst_agent_id=int(data['agent_ids'][index[1]]),
                                worst_future_index=int(index[2]), limit=limit)
    rows.append(dict(file=path.name, step=int(data['source_step']), selected=int(data['selected_index']),
                     dynamics_feasible_samples=np.flatnonzero(valid).tolist(), samples=len(valid), **diagnostics))
print(json.dumps(rows, indent=2, allow_nan=False))
