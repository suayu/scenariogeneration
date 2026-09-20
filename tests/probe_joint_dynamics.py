"""独立诊断批次：冻结相同单场景，保留全部联合样本用于离线分析。"""
import os
from pathlib import Path
import sys
import time
import argparse

import launch_multiagent_matrix as matrix

parser = argparse.ArgumentParser()
parser.add_argument('--kinematics-scale', type=float, default=1.0)
args = parser.parse_args()
if not 0 < args.kinematics_scale <= 100:
    raise ValueError('invalid diagnostic kinematics scale')
matrix.BASE = matrix.ROOT/'experiments'/('riskweaver_joint_diagnostic_'+str(time.time_ns()))
matrix.EXTRA_OVERRIDES = [f'sim.traffic_model.guidance.loss_configs.scenario_collision.kinematics_loss_scale={args.kinematics_scale}']
matrix.ARMS = [('single', 'trajectory_only', True)]
os.environ['RISKWEAVER_JOINT_DIAGNOSTIC_DIR'] = str(matrix.BASE/'joint_samples')
sys.argv = [__file__, '--phase', 'smoke']
matrix.main()
print(matrix.BASE, flush=True)
