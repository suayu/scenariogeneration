"""晚出现交互车的单场景冒烟：增加规划机会与观察帧数。"""
import os
import sys
import time
import launch_multiagent_matrix as matrix

matrix.BASE=matrix.ROOT/'experiments'/('riskweaver_scene7_budget_'+str(time.time_ns()))
matrix.SMOKE_SCENE_INDEX=7
matrix.SMOKE_STEPS=120
matrix.SMOKE_MAX_PLANNING_CALLS=30
matrix.ARMS=[('single','trajectory_only',True)]
matrix.SOURCE += ['policies/trajectory_projection.py','tests/probe_scene7_budget.py']
matrix.EXTRA_OVERRIDES=[
    'sim.traffic_model.dynamics_projection.enabled=true',
    'sim.ego_profile.low_speed_decelerations_mps2=[0.25,0.5]',
]
os.environ['RISKWEAVER_JOINT_DIAGNOSTIC_DIR']=str(matrix.BASE/'joint_samples')
sys.argv=[__file__,'--phase','smoke']
matrix.main()
