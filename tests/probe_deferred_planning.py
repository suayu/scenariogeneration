"""画像预热 20 帧后首次规划；仅验证默认关闭的共同实验配置。"""
import sys
import time
import launch_multiagent_matrix as matrix

matrix.BASE = matrix.ROOT/'experiments'/('riskweaver_deferred_smoke_'+str(time.time_ns()))
matrix.SMOKE_SCENE_INDEX = 3
matrix.ARMS = [('single', 'trajectory_only', True)]
matrix.SOURCE += ['policies/trajectory_projection.py', 'tests/probe_deferred_planning.py']
matrix.EXTRA_OVERRIDES = [
    'sim.traffic_model.dynamics_projection.enabled=true',
    '+sim.llm.min_planning_step=20',
    '+sim.llm.planning_interval_frames=10',
]
sys.argv = [__file__, '--phase', 'smoke']
matrix.main()
