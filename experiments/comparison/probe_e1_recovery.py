"""重放已保存状态的组件诊断；不启动额外正式场景、不覆盖R1产物。"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('PROJECT_ROOT', str(ROOT))
os.environ.setdefault('SCRATCH_ROOT', str(ROOT))
os.environ.setdefault('DATASET_ROOT', str(ROOT/'metadata'))

import numpy as np
from hydra import compose, initialize

from experiments.comparison.ctrl_sim_forecast import CtrlSimJointForecaster, validate_forecast
from policies.idm_policy import IDMPolicy
from policies.strive_adapter import StriveController
from simulator import Simulator


def restore_frame(env, states, step):
    """仅从已发生的trace重建历史，不能使用trace中未到源帧的未来状态。"""
    prefix = [states[k] for k in range(step + 1)]
    env.t = step
    env.ego_state = np.asarray(prefix[-1]['ego']).copy()
    env.agent_active = np.asarray(prefix[-1]['active'], bool)
    env.data_dict['ego'] = [np.asarray(row['ego'])[None] for row in prefix]
    env.data_dict['agent'] = [np.asarray(row['agents']) for row in prefix]
    env.data_dict['agent_active'] = env.agent_active.copy()
    env.data_dict['agent_active_history'] = [np.asarray(row['active'], bool) for row in prefix]
    env.data_dict['agent_action'] = [np.full(len(env.agent_active), -1) for _ in range(step)]
    env.data_dict['agent_rtg'] = [np.zeros(len(env.agent_active)) for _ in range(step)]
    env.data_dict['ego_action'] = [np.array(-1) for _ in range(step)]
    env.data_dict['ego_rtg'] = [np.array([[349.]]) for _ in range(step)]
    env.local_frame = {'center': env.ego_state[:2].copy(), 'yaw': env.ego_state[4].copy()}
    env.last_active_agent_position = env.data_dict['agent'][-1].copy()
    env.current_state = env._get_observation()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['forecast', 'strive'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir/'component.log').open('w') as log, contextlib.redirect_stdout(log):
        with initialize(version_base=None, config_path='../../cfgs'):
            cfg = compose(config_name='config', overrides=['sim.traffic_model.backend=external_joint',
                'sim.visualize=false', 'sim.policy=idm', 'sim.seed=42', 'sim.llm.obstacles.enabled=false'])
        env = Simulator(cfg)
        manifest = json.loads((ROOT/'experiments/scenario_dreamer_ldm_expand250_20260928_r1/riskweaver_frozen_250_scenes.json').read_text())
        env.test_files = manifest['scenario_files']
        obs = env.reset(0)
        policy = IDMPolicy(cfg, env)
        policy.reset(obs)
        results = {'diagnostic_only': True, 'real_executed_frames': 0, 'mode': args.mode}
        if args.mode == 'forecast':
            source = ROOT/'experiments/e1_three_baseline_single_scene_smoke_20261008_r1/safe_sim/movies/scenario_000/execution_trace.jsonl'
            states = {r['step']: r for r in map(json.loads, source.read_text().splitlines()) if r['kind'] == 'state'}
            forecasts = []
            forecaster = CtrlSimJointForecaster(env)
            for step in (32, 50, 100):
                restore_frame(env, states, step)
                policy.t = step
                for tracking in ('observation', 'persistent'):
                    forecaster.tracking = tracking
                    result = forecaster.predict(env, policy)
                    error = None
                    try:
                        validate_forecast(result, env, 3.)
                    except ValueError as exc:
                        error = str(exc)
                    forecasts.append({'step': step, 'tracking': tracking, 'validation_error': error,
                                      'metadata': result.metadata})
            results['forecasts'] = forecasts
        else:
            # 小预算验证真实权重、CUDA反向传播和真实策略回调；不当作正式性能结果。
            controller = StriveController('/home2/zhaoyx/STRIVE-riskweaver',
                '/home2/zhaoyx/STRIVE-riskweaver/.venv/bin/python',
                ROOT/'checkpoints/strive_official_20261008/traffic_model.pth',
                args.output_dir/'planning', init_iters=2, adv_iters=1, timeout=600)
            joint = controller.predict(env, policy)
            results['prediction'] = joint.metadata
            results['joint_shape'] = list(joint.positions_global.shape)
            assert np.isfinite(joint.positions_global).all()
            assert joint.metadata['model_loaded_strict']
            assert joint.metadata['planner_rollout_calls'] > 0
        (args.output_dir/'result.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(results, ensure_ascii=False))


if __name__ == '__main__':
    main()
