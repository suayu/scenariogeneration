"""无画像离线记忆训练源：冻结场景 3，不与评估场景 1 重叠。"""
import sys
import time
import launch_multiagent_matrix as matrix

matrix.BASE=matrix.ROOT/'experiments'/('riskweaver_memory_train_'+str(time.time_ns()))
matrix.SMOKE_SCENE_INDEX=3
matrix.SMOKE_STEPS=80
matrix.SMOKE_MAX_PLANNING_CALLS=1
matrix.ARMS=[('single','trajectory_only',False)]
matrix.SOURCE += ['tests/probe_memory_training.py']
matrix.EXTRA_OVERRIDES=[]
sys.argv=[__file__,'--phase','smoke']
matrix.main()
