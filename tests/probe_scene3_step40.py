"""原始双背景场景：延迟到第 40 帧诊断一次攻击规划。"""
import os
import sys
import time
import launch_multiagent_matrix as matrix

matrix.BASE=matrix.ROOT/'experiments'/('riskweaver_scene3_step40_'+str(time.time_ns()))
matrix.SMOKE_SCENE_INDEX=3
matrix.SMOKE_STEPS=100
matrix.SMOKE_MAX_PLANNING_CALLS=1
matrix.ARMS=[('single','trajectory_only',True)]
matrix.SOURCE += ['policies/trajectory_projection.py','tests/probe_scene3_step40.py']
matrix.EXTRA_OVERRIDES=[
    'sim.traffic_model.dynamics_projection.enabled=true',
    'sim.traffic_model.attack_route_weight=5.0',
    'sim.traffic_model.attack_route_lane_margin=0.55',
    '+sim.llm.min_planning_step=40',
    '+sim.llm.planning_interval_frames=10',
    'sim.ego_profile.low_speed_decelerations_mps2=[0.25,0.5]',
]
os.environ['RISKWEAVER_JOINT_DIAGNOSTIC_DIR']=str(matrix.BASE/'joint_samples')
sys.argv=[__file__,'--phase','smoke']
matrix.main()
