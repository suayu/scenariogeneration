"""同一冻结多背景场景比较四种 LLM 规划模式。"""
import sys
import time
import launch_multiagent_matrix as matrix

matrix.BASE=matrix.ROOT/'experiments'/('riskweaver_four_arm_multibg_'+str(time.time_ns()))
matrix.SMOKE_SCENE_INDEX=1
matrix.SMOKE_STEPS=80
matrix.SMOKE_MAX_PLANNING_CALLS=1
matrix.ARMS=[(mode,'trajectory_only',False) for mode in ('single','conditional_critic','parallel_arbiter','parallel_memory')]
matrix.SOURCE += ['tests/probe_four_arm_multibg.py']
matrix.EXTRA_OVERRIDES=[]
sys.argv=[__file__,'--phase','smoke']
matrix.main()
