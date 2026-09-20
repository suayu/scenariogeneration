"""从既有冻结实验读取第十帧，筛选值得独立冒烟的场景。"""
import glob
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
BASE = ROOT/'experiments/riskweaver_target_formal_20260913/movies'
for index in range(20):
    paths = sorted(glob.glob(str(BASE/f'scenario_{index:03d}'/'attempt_0'/'execution_trace.jsonl')))
    if not paths:
        print(index, 'missing trace')
        continue
    records = [json.loads(line) for line in open(paths[0])]
    state = next((r for r in records if r.get('kind') == 'state' and r['step'] == 10), None)
    if state is None:
        print(index, 'ended before step10')
        continue
    ego = np.asarray(state['ego'])
    agents = np.asarray(state['agents'])
    active = np.asarray(state['active'])
    forward = np.array([math.cos(ego[4]), math.sin(ego[4])])
    side = np.array([-forward[1], forward[0]])
    picks = []
    for j in np.flatnonzero(active):
        bg = agents[j]
        rel = bg[:2]-ego[:2]
        front, lateral = rel@forward, rel@side
        yaw = abs(math.atan2(math.sin(bg[4]-ego[4]), math.cos(bg[4]-ego[4])))
        speed = np.linalg.norm(bg[2:4])
        if 5 < front < 45 and abs(lateral) <= 7 and yaw <= .7 and speed >= 1:
            picks.append((int(j), round(float(front), 1), round(float(lateral), 1)))
    print(index, Path(paths[0]).parents[1].name, 'active', int(active.sum()), 'prelim', picks)
