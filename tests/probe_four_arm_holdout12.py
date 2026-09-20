"""用冻结的四场景离线记忆评估独立的第 12 个场景。"""
import sys
import time

import launch_multiagent_matrix as matrix


# 留出场景不在离线复盘库中；其余条件与前期四组诊断一致。
matrix.BASE = matrix.ROOT / 'experiments' / ('riskweaver_four_arm_holdout12_' + str(time.time_ns()))
matrix.MEMORY = matrix.ROOT / 'experiments/riskweaver_unprofiled_memory_v2_20260918/memory_verified.json'
matrix.SMOKE_SCENE_INDEX = 12
matrix.SMOKE_STEPS = 80
matrix.SMOKE_MAX_PLANNING_CALLS = 1
matrix.ARMS = [(mode, 'trajectory_only', False) for mode in
               ('single', 'conditional_critic', 'parallel_arbiter', 'parallel_memory')]
matrix.SOURCE += ['tests/probe_four_arm_holdout12.py', str(matrix.MEMORY)]
matrix.EXTRA_OVERRIDES = []
sys.argv = [__file__, '--phase', 'smoke']
matrix.main()
