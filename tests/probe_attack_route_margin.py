"""原始多背景场景：诊断攻击窗口道路软引导的边界裕度。"""
import os
import sys
import time
import launch_multiagent_matrix as matrix

matrix.BASE=matrix.ROOT/'experiments'/('riskweaver_attack_margin_smoke_'+str(time.time_ns()))
matrix.SMOKE_SCENE_INDEX=3
matrix.SMOKE_STEPS=80
matrix.SMOKE_MAX_PLANNING_CALLS=1
matrix.ARMS=[('single','trajectory_only',True)]
matrix.SOURCE += ['policies/trajectory_projection.py','tests/probe_attack_route_margin.py']
matrix.EXTRA_OVERRIDES=[
    'sim.traffic_model.dynamics_projection.enabled=true',
    'sim.traffic_model.attack_route_weight=5.0',
    'sim.traffic_model.attack_route_lane_margin=0.55',
    '+sim.llm.min_planning_step=20',
    '+sim.llm.planning_interval_frames=10',
    'sim.ego_profile.low_speed_decelerations_mps2=[0.25,0.5]',
]
os.environ['RISKWEAVER_JOINT_DIAGNOSTIC_DIR']=str(matrix.BASE/'joint_samples')
sys.argv=[__file__,'--phase','smoke']
matrix.main()
