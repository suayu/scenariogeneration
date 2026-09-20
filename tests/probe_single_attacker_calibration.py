"""另存原场景的单攻击车校准副本，验证完整闭环但不替代真实场景对比。"""
import copy
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np

import launch_multiagent_matrix as matrix

source=matrix.ROOT/'metadata/simulation_environment_datasets/scenario_dreamer_waymo_200m_pickles/6_5.pkl'
matrix.BASE=matrix.ROOT/'experiments'/('riskweaver_single_attacker_calibration_'+str(time.time_ns()))
matrix.BASE.mkdir(parents=True,exist_ok=False)
with source.open('rb') as handle:
    scene=pickle.load(handle)
derived=copy.deepcopy(scene)
kept=[2,len(scene['agents'])-1]
derived['agents']=np.asarray(scene['agents'])[kept].copy()
derived['agent_types']=np.asarray(scene['agent_types'])[kept].copy()
derived['num_agents']=2
target=matrix.BASE/'single_attacker_6_5.pkl'
with target.open('xb') as handle:
    pickle.dump(derived,handle,protocol=pickle.HIGHEST_PROTOCOL)
matrix.write(matrix.BASE/'derivation.json',dict(source=str(source),source_sha256=matrix.digest(source),
    derived=str(target),derived_sha256=matrix.digest(target),kept_original_agent_indices=kept,
    purpose='isolated pipeline calibration; not formal comparison'))
manifest=matrix.BASE/'smoke_scenario_manifest.json'
matrix.write(manifest,dict(scenario_files=[str(target)],scenario_sha256=[matrix.digest(target)]))
matrix.SMOKE_STEPS=80
matrix.SMOKE_MAX_PLANNING_CALLS=6
matrix.EXTRA_OVERRIDES=[
    'sim.traffic_model.dynamics_projection.enabled=true',
    '+sim.llm.min_planning_step=20',
    '+sim.llm.planning_interval_frames=10',
    'sim.ego_profile.low_speed_decelerations_mps2=[0.25,0.5]',
]
matrix.SOURCE += ['policies/trajectory_projection.py','tests/probe_single_attacker_calibration.py']
(matrix.BASE/'common_instructions.txt').write_text(
    'Generate a physically executable, avoidable interaction that probes the ego driving strategy. Follow all safety validators.\n')
hashes={name:matrix.digest(matrix.ROOT/name) for name in matrix.SOURCE}
matrix.write(matrix.BASE/'source_hashes.json',hashes)
os.environ['RISKWEAVER_JOINT_DIAGNOSTIC_DIR']=str(matrix.BASE/'joint_samples')
summary=matrix.run_arm('single','trajectory_only',True,'smoke',manifest,hashes)
print(json.dumps(summary,ensure_ascii=False),flush=True)
