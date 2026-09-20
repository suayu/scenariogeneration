"""显式可行性否决规则后的第四组同场景复验。"""
import sys
import time
import launch_multiagent_matrix as matrix

matrix.BASE=matrix.ROOT/'experiments'/('riskweaver_memory_multibg_r2_'+str(time.time_ns()))
matrix.MEMORY=matrix.ROOT/'experiments/riskweaver_memory_train_1789703744058152089/unprofiled_memory_verified.json'
matrix.SMOKE_SCENE_INDEX=1
matrix.SMOKE_STEPS=80
matrix.SMOKE_MAX_PLANNING_CALLS=1
matrix.ARMS=[('parallel_memory','trajectory_only',False)]
matrix.SOURCE += ['tests/probe_memory_multibg_r2.py',str(matrix.MEMORY)]
matrix.EXTRA_OVERRIDES=[]
sys.argv=[__file__,'--phase','smoke']
matrix.main()
