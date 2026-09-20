"""验证放宽道路/规避见证门控后的单场景真实闭环。"""
import sys
import time

import launch_multiagent_matrix as matrix


# 使用曾有画像候选的冻结场景；单 Agent 保持原规划路径，只改变本次明确要求的门控。
matrix.BASE = matrix.ROOT / 'experiments' / ('riskweaver_relaxed_gate_smoke_' + str(time.time_ns()))
matrix.SMOKE_SCENE_INDEX = 3
matrix.SMOKE_STEPS = 80
matrix.SMOKE_MAX_PLANNING_CALLS = 1
matrix.ARMS = [('single', 'trajectory_only', True)]
matrix.SOURCE += ['tests/probe_relaxed_gate_smoke.py']
matrix.EXTRA_OVERRIDES = []
sys.argv = [__file__, '--phase', 'smoke']
matrix.main()
