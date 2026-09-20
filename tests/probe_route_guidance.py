"""固定求解器与安全阈值，仅扫描非目标背景车的道路引导权重。"""
import os
import sys
import time
import launch_multiagent_matrix as matrix

matrix.BASE = matrix.ROOT/'experiments'/('riskweaver_route_weight_5_'+str(time.time_ns()))
matrix.SMOKE_SCENE_INDEX = 3
matrix.ARMS = [('single', 'trajectory_only', True)]
matrix.SOURCE += ['policies/trajectory_projection.py', 'tests/probe_route_guidance.py']
matrix.EXTRA_OVERRIDES = [
    'sim.traffic_model.dynamics_projection.enabled=true',
    '+sim.llm.min_planning_step=20',
    '+sim.llm.planning_interval_frames=10',
    'sim.traffic_model.guidance.llm_joint.weight_variables.route=5.0',
]
os.environ['RISKWEAVER_JOINT_DIAGNOSTIC_DIR'] = str(matrix.BASE/'joint_samples')
sys.argv = [__file__, '--phase', 'smoke']
matrix.main()
