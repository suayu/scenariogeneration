"""使用冻结道路与联合样本定位道路门禁，区分真实起点和预测点越界。"""
import json
from pathlib import Path

import numpy as np

from policies.ego_profile import project
from policies.trajectory_projection import project_positions

root = Path('/home2/zhaoyx/scenario-dreamer')
data = np.load(root/'experiments/riskweaver_joint_diagnostic_1789627892542948329/joint_samples/joint_10_1789627946995714299.npz')
initial = json.loads((root/'experiments/riskweaver_projection_smoke_20260917_r1/smoke/single/carla/initial_data/scenario_0000_initial.json').read_text())
lanes = [np.asarray(l)[:, :2] for l in initial['road_network'] if len(l)>1 and project(np.asarray(l)[0,:2], l) is not None]
limits = dict(max_speed=20., max_acceleration=6., max_jerk=12., max_step_distance=2.)
rows = []
for row, agent_id in enumerate(data['agent_ids']):
    state = np.asarray(data['states'][agent_id], float)
    positions, _, headings, _ = project_positions(data['positions'][int(data['selected_index']), row], state, float(data['dt']), limits)
    points = np.vstack((state[:2], positions))
    yaw = np.unwrap(np.r_[state[4], headings[:,0]])
    points = np.vstack((points, (points[1:]+points[:-1])/2))
    yaw = np.r_[yaw, (yaw[1:]+yaw[:-1])/2]
    corners = []
    for along, across in ((1,1),(1,-1),(-1,1),(-1,-1)):
        corner = points+along*state[5]/2*np.column_stack((np.cos(yaw),np.sin(yaw)))+across*state[6]/2*np.column_stack((-np.sin(yaw),np.cos(yaw)))
        distances = np.min(np.stack([project(corner, line) for line in lanes]), axis=0)
        corners.append(dict(along=along, across=across, initial_m=float(distances[0]),
                            future_max_m=float(distances[1:].max()), worst_index=int(np.argmax(distances))))
    rows.append(dict(agent_id=int(agent_id), width=float(state[6]), length=float(state[5]),
                     lane_half_width_m=1.8, corners=corners))
print(json.dumps(rows, indent=2))
