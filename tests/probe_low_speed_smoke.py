"""低速候选独立闭环冒烟：只启用共有参数，不进入四组正式比较。"""
import os
import sys
import time
import launch_multiagent_matrix as matrix

matrix.BASE=matrix.ROOT/'experiments'/('riskweaver_low_speed_smoke_'+str(time.time_ns()))
matrix.SMOKE_SCENE_INDEX=3
matrix.ARMS=[('single','trajectory_only',True)]
matrix.SOURCE += ['policies/trajectory_projection.py','tests/probe_low_speed_smoke.py']
matrix.EXTRA_OVERRIDES=[
    'sim.traffic_model.dynamics_projection.enabled=true',
    '+sim.llm.min_planning_step=20',
    '+sim.llm.planning_interval_frames=10',
    'sim.traffic_model.guidance.llm_joint.weight_variables.route=5.0',
    'sim.ego_profile.low_speed_decelerations_mps2=[0.25,0.5]',
]
os.environ['RISKWEAVER_JOINT_DIAGNOSTIC_DIR']=str(matrix.BASE/'joint_samples')
sys.argv=[__file__,'--phase','smoke']
matrix.main()
