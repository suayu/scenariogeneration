"""第三个冻结多背景场景：保持四组相同预算并保留全部拒绝结果。"""
import sys
import time

import launch_multiagent_matrix as matrix


# 仅切换冻结场景，画像、模型、种子和调用预算沿用前两轮诊断设置。
matrix.BASE = matrix.ROOT / 'experiments' / ('riskweaver_four_arm_scene14_' + str(time.time_ns()))
matrix.MEMORY = matrix.ROOT / 'experiments/riskweaver_memory_train_1789703744058152089/unprofiled_memory_verified.json'
matrix.SMOKE_SCENE_INDEX = 14
matrix.SMOKE_STEPS = 80
matrix.SMOKE_MAX_PLANNING_CALLS = 1
matrix.ARMS = [(mode, 'trajectory_only', False) for mode in
               ('single', 'conditional_critic', 'parallel_arbiter', 'parallel_memory')]
matrix.SOURCE += ['tests/probe_four_arm_scene14.py', str(matrix.MEMORY)]
matrix.EXTRA_OVERRIDES = []
sys.argv = [__file__, '--phase', 'smoke']
matrix.main()
