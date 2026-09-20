"""公共动力学投影的独立闭环验证，通过后才开展四组新对照。"""
import sys
import argparse
import time
import launch_multiagent_matrix as matrix

parser = argparse.ArgumentParser()
parser.add_argument('--scene-index', type=int, default=3)
args = parser.parse_args()
if not 0 <= args.scene_index < 20:
    raise ValueError('scene index out of frozen manifest')
matrix.BASE = matrix.ROOT/'experiments'/('riskweaver_projection_smoke_'+str(time.time_ns()))
matrix.SMOKE_SCENE_INDEX = args.scene_index
matrix.ARMS = [('single', 'trajectory_only', True)]
matrix.SOURCE += ['policies/trajectory_projection.py', 'tests/probe_projection_smoke.py']
matrix.EXTRA_OVERRIDES = ['sim.traffic_model.dynamics_projection.enabled=true']
sys.argv = [__file__, '--phase', 'smoke']
matrix.main()
