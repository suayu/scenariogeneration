"""画像关闭的原始多背景规划链路冒烟，独立于画像对照。"""
import os
import sys
import time
import launch_multiagent_matrix as matrix

matrix.BASE=matrix.ROOT/'experiments'/('riskweaver_no_profile_multibg_'+str(time.time_ns()))
matrix.SMOKE_SCENE_INDEX=1
matrix.SMOKE_STEPS=80
matrix.SMOKE_MAX_PLANNING_CALLS=1
matrix.ARMS=[('single','trajectory_only',False)]
matrix.SOURCE += ['policies/trajectory_projection.py','tests/probe_multiagent_no_profile.py']
matrix.EXTRA_OVERRIDES=[]
os.environ['RISKWEAVER_JOINT_DIAGNOSTIC_DIR']=str(matrix.BASE/'joint_samples')
sys.argv=[__file__,'--phase','smoke']
matrix.main()
