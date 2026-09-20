"""第二个冻结多背景场景：同配置四组诊断，训练记忆严格分离。"""
import sys
import time
import launch_multiagent_matrix as matrix

matrix.BASE=matrix.ROOT/'experiments'/('riskweaver_four_arm_second_scene_'+str(time.time_ns()))
matrix.MEMORY=matrix.ROOT/'experiments/riskweaver_memory_train_1789703744058152089/unprofiled_memory_verified.json'
matrix.SMOKE_SCENE_INDEX=2
matrix.SMOKE_STEPS=80
matrix.SMOKE_MAX_PLANNING_CALLS=1
matrix.ARMS=[(mode,'trajectory_only',False) for mode in ('single','conditional_critic','parallel_arbiter','parallel_memory')]
matrix.SOURCE += ['tests/probe_four_arm_second_scene.py',str(matrix.MEMORY)]
matrix.EXTRA_OVERRIDES=[]
sys.argv=[__file__,'--phase','smoke']
matrix.main()
