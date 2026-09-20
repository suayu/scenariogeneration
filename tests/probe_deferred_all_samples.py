"""保存延后规划时的所有候选供离线道路与动力学审计。"""
import os
import sys
import time
import launch_multiagent_matrix as matrix

matrix.BASE = matrix.ROOT/'experiments'/('riskweaver_deferred_samples_'+str(time.time_ns()))
matrix.SMOKE_SCENE_INDEX = 3
matrix.ARMS = [('single', 'trajectory_only', True)]
matrix.SOURCE += ['policies/trajectory_projection.py', 'tests/probe_deferred_all_samples.py']
matrix.EXTRA_OVERRIDES = [
    'sim.traffic_model.dynamics_projection.enabled=true',
    '+sim.llm.min_planning_step=20',
    '+sim.llm.planning_interval_frames=10',
]
os.environ['RISKWEAVER_JOINT_DIAGNOSTIC_DIR'] = str(matrix.BASE/'joint_samples')
sys.argv = [__file__, '--phase', 'smoke']
matrix.main()
