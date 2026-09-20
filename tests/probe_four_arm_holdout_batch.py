"""三个互不重叠的冻结留出场景，逐组保留通过和失败的完整闭环结果。"""
import json
import sys
import time

import launch_multiagent_matrix as matrix


ROOT = matrix.ROOT
MEMORY = ROOT / 'experiments/riskweaver_unprofiled_memory_v2_20260918/memory_verified.json'
BASE = ROOT / 'experiments' / ('riskweaver_four_arm_holdout_batch_' + str(time.time_ns()))
SCENES = (13, 15, 17)


def main():
    BASE.mkdir()
    matrix.MEMORY = MEMORY
    matrix.SMOKE_STEPS = 80
    matrix.SMOKE_MAX_PLANNING_CALLS = 1
    matrix.ARMS = [(mode, 'trajectory_only', False) for mode in
                   ('single', 'conditional_critic', 'parallel_arbiter', 'parallel_memory')]
    matrix.SOURCE += ['tests/probe_four_arm_holdout_batch.py', str(MEMORY)]
    matrix.EXTRA_OVERRIDES = []
    sys.argv = [__file__, '--phase', 'smoke']
    for index in SCENES:
        # 每个场景独立冻结源代码、模型配置及视频，服务失败仍由底层入口立即终止。
        matrix.BASE = BASE / f'scene_{index:02d}'
        matrix.SMOKE_SCENE_INDEX = index
        print(json.dumps({'scene_index': index, 'state': 'started', 'base': str(matrix.BASE)}), flush=True)
        matrix.main()
        print(json.dumps({'scene_index': index, 'state': 'finished'}), flush=True)


if __name__ == '__main__':
    main()
