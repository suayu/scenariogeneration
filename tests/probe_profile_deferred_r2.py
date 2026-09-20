"""画像预热后验证放宽门控、真实百炼规划和实际攻击执行。"""
import sys
import time

import launch_multiagent_matrix as matrix


matrix.BASE = matrix.ROOT / 'experiments' / ('riskweaver_profile_deferred_r2_' + str(time.time_ns()))
matrix.SMOKE_SCENE_INDEX = 3
matrix.SMOKE_STEPS = 100
matrix.SMOKE_MAX_PLANNING_CALLS = 6
matrix.ARMS = [('single', 'trajectory_only', True)]
matrix.SOURCE += ['policies/trajectory_projection.py', 'tests/probe_profile_deferred_r2.py']
matrix.EXTRA_OVERRIDES = [
    'sim.traffic_model.dynamics_projection.enabled=true',
    '+sim.llm.min_planning_step=20',
    '+sim.llm.planning_interval_frames=10',
]
matrix.BASE.mkdir(parents=True, exist_ok=True)
(matrix.BASE / 'common_instructions.txt').write_text(
    'Generate a physically executable adversarial interaction that probes the ego driving strategy. '
    'The attacker may leave the road topology. Preserve non-target background safety and a positive '
    'theoretical drivable region for the ego.\n', encoding='utf-8')
sys.argv = [__file__, '--phase', 'smoke']
matrix.main()
