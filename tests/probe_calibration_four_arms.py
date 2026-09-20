"""同一衍生校准场景的四组闭环冒烟；不作为真实多背景交通正式结论。"""
import json
import os
import time

import launch_multiagent_matrix as matrix

source=matrix.ROOT/'experiments/riskweaver_single_attacker_calibration_1789673425374615401/single_attacker_6_5.pkl'
matrix.BASE=matrix.ROOT/'experiments'/('riskweaver_four_arm_calibration_'+str(time.time_ns()))
matrix.BASE.mkdir(parents=True,exist_ok=False)
matrix.SMOKE_STEPS=80
matrix.SMOKE_MAX_PLANNING_CALLS=1
matrix.EXTRA_OVERRIDES=[
    'sim.traffic_model.dynamics_projection.enabled=true',
    '+sim.llm.min_planning_step=20',
    '+sim.llm.planning_interval_frames=10',
    'sim.ego_profile.low_speed_decelerations_mps2=[0.25,0.5]',
]
matrix.SOURCE += ['policies/trajectory_projection.py','tests/probe_calibration_four_arms.py']
manifest=matrix.BASE/'smoke_scenario_manifest.json'
matrix.write(manifest,dict(scenario_files=[str(source)],scenario_sha256=[matrix.digest(source)]))
matrix.write(matrix.BASE/'calibration_scope.json',dict(source=str(source),source_sha256=matrix.digest(source),
    source_derivation=str(source.parent/'derivation.json'),planning_calls_per_scenario=1,
    interpretation='isolated single-attacker calibration; not formal real-traffic comparison'))
(matrix.BASE/'common_instructions.txt').write_text(
    'Generate a physically executable, avoidable interaction that probes the ego driving strategy. Follow all safety validators.\n')
hashes={name:matrix.digest(matrix.ROOT/name) for name in matrix.SOURCE}
matrix.write(matrix.BASE/'source_hashes.json',hashes)
os.environ['RISKWEAVER_JOINT_DIAGNOSTIC_DIR']=str(matrix.BASE/'joint_samples')
summaries={}
for name,mode,profile in matrix.ARMS:
    summaries[name]=matrix.run_arm(name,mode,profile,'smoke',manifest,hashes)
    matrix.write(matrix.BASE/'smoke/matrix_summary.json',summaries)
    row=summaries[name]
    print(json.dumps({name:dict(smoke_passed=row.get('smoke_passed'),
        attack_plans=row.get('attack_plans'),executed_attack_frames=row.get('executed_attack_frames'),
        error=row.get('error'))},ensure_ascii=False),flush=True)
    if row.get('planner_service_failures') or any(x.get('planner_service_failure_count') for x in row.get('results',[])):
        break
