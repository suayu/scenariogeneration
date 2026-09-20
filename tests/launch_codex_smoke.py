"""固定预算单场景闭环冒烟；不启动正式评估，不覆盖历史产物。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home2/zhaoyx/scenario-dreamer')


def save(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # 按冻结顺序选择首个初始规划输入有车辆的场景，不使用攻击结果。
    previous = json.loads((ROOT / 'experiments/riskweaver_trajectory_smoke_20260914/manifest.json').read_text())
    initial_counts = []
    for index, name in enumerate(previous['scenario_files']):
        trace = ROOT / f'experiments/riskweaver_trajectory_formal_20260914_r2/movies/scenario_{index:03d}/attempt_0/execution_trace.jsonl'
        with trace.open() as handle:
            record = next(json.loads(line) for line in handle if '"kind": "llm_input"' in line)
        count = len(record['state']['agents'])
        initial_counts.append({'index': index, 'source': name, 'initial_planning_agents': count})
        if count:
            break
    else:
        raise RuntimeError('冻结清单中没有初始交互车辆')
    source = Path(previous['scenario_files'][index])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == previous['scenario_sha256'][index]
    manifest = {'scenario_files': [str(source)], 'scenario_sha256': [digest], 'seed': 42,
                'steps': 40, 'max_replays': 0, 'purpose': 'planner_diffusion_execution_smoke'}
    save(args.output / 'manifest.json', manifest)
    save(args.output / 'selection_basis.json', initial_counts)
    files = ['run_simulation.py', 'scenario_generator.py', 'simulator.py', 'cfgs/sim/base.yaml',
             'policies/codex_planner_bridge.py', 'policies/llm_adversarial_planner.py',
             'policies/difficulty_control.py', 'policies/diffusion_model_wrapper.py',
             'policies/evaluation_trace.py', 'policies/planner_smoke_gate.py',
             'policies/joint_safety.py', 'scripts/codex_queue.py']
    save(args.output / 'source_hashes.json', {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files})
    overrides = ['sim.seed=42', '+sim.continue_after_off_route=true', 'sim.evaluation.max_scenarios=1',
                 'sim.steps=40', f'+sim.scenario_manifest={args.output}/manifest.json',
                 f'+sim.attack_request_file={ROOT}/tests/riskweaver_trajectory_instructions.txt',
                 'sim.llm.provider=codex', 'sim.llm.attack_mode=trajectory_only', 'sim.llm.obstacles.enabled=false',
                 'sim.traffic_model.device=cuda:2', 'sim.traffic_model.guidance.mode=llm_joint',
                 'sim.traffic_model.iterative_adversarial.profile_enabled=true',
                 'sim.traffic_model.iterative_adversarial.escalation_enabled=true',
                 'sim.traffic_model.iterative_adversarial.full_method_enabled=true',
                 'sim.evaluation.difficulty_control.mode=target',
                 'sim.evaluation.difficulty_control.target_difficulty=0.5',
                 'sim.evaluation.difficulty_control.tolerance=0.1',
                 'sim.evaluation.difficulty_control.max_replays=0',
                 'sim.evaluation.difficulty_control.smoke_acceptance=true',
                 'sim.visualize=true', 'sim.lightweight=true', 'sim.visualization.show_diffusion_candidates=true',
                 f'sim.movie_path={args.output}/movies', f'sim.scenario_data_output_path={args.output}/carla',
                 f'hydra.run.dir={args.output}/hydra']
    env = dict(os.environ, PROJECT_ROOT=str(ROOT), SCRATCH_ROOT=str(ROOT), DATASET_ROOT=str(ROOT/'metadata'),
               PYTHONPATH=f'{ROOT}:{ROOT}/safe-sim:{ROOT}/safe-sim/trajdata/src', MPLBACKEND='Agg',
               PYTHONUNBUFFERED='1', HYDRA_FULL_ERROR='1')
    command = ['bash', 'scripts/run_with_llm_provider.sh', 'codex', sys.executable, 'run_simulation.py', *overrides]
    save(args.output / 'launch.json', {'command': command, 'pid': os.getpid(), 'started_at': time.time()})
    with (args.output / 'resolved_config.yaml').open('x') as handle:
        subprocess.run(command + ['--cfg', 'job', '--resolve'], cwd=ROOT, env=env, stdout=handle, check=True)
    if args.prepare_only:
        return 0
    with (args.output / 'run.log').open('x') as handle:
        code = subprocess.call(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    save(args.output / 'exit_status.json', {'exit_code': code, 'finished_at': time.time()})
    return code


if __name__ == '__main__':
    raise SystemExit(main())
