"""冻结 RiskWeaver 实验输入；完整冒烟验收后才允许正式测试。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
SOURCE_FILES = [
    'run_simulation.py', 'simulator.py', 'scenario_generator.py', 'cfgs/sim/base.yaml',
    'policies/llm_adversarial_planner.py', 'policies/diffusion_model_wrapper.py',
    'policies/scenario_guidance.py', 'policies/joint_safety.py', 'policies/evaluation_trace.py',
    'policies/difficulty_control.py', 'policies/difficulty_audit.py', 'policies/risk_metrics.py',
    'safe-sim/tbsim/models/diffusion.py', 'safe-sim/tbsim/utils/guidance_utils.py',
    'tests/riskweaver_trajectory_instructions.txt',
]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def freeze(path):
    # 沿用既有数据枚举顺序，不能通过挑选容易通过的场景改变正式样本。
    dataset = ROOT / 'metadata/simulation_environment_datasets/scenario_dreamer_waymo_200m_pickles'
    files = [dataset / name for name in os.listdir(dataset)][:20]
    if len(files) != 20 or any(not file.is_file() for file in files):
        raise RuntimeError('正式清单必须包含 20 个真实场景')
    manifest = {'scenario_files': [str(file) for file in files],
                'scenario_sha256': [digest(file) for file in files], 'seed': 42,
                'target': .5, 'tolerance': .1, 'max_replays': 2, 'steps': 400,
                'instruction_file': str(ROOT / 'tests/riskweaver_trajectory_instructions.txt')}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def validate_smoke(directory, hashes):
    # 逐尝试检查闭环、真实测量、背景安全与视频；目标未达也须保留为未命中。
    if json.loads((directory / 'exit_status.json').read_text())['exit_code'] != 0:
        raise RuntimeError('冒烟进程未成功完成')
    previous = json.loads((directory / 'source_hashes.json').read_text())
    if previous != hashes:
        raise RuntimeError('源码已不同于通过冒烟的版本，禁止启动正式实验')
    attempts = list((directory / 'movies').glob('scenario_*/attempt_*/attempt_result.json'))
    if not attempts:
        raise RuntimeError('冒烟没有完整尝试记录')
    for path in attempts:
        result = json.loads(path.read_text())
        feasible = result['feasibility']
        if result['generation_failure'] or not result['danger_valid']:
            raise RuntimeError('冒烟存在生成失败或缺少测量')
        if feasible['background_collision_frames'] or feasible['background_static_collision_frames']:
            raise RuntimeError('冒烟存在背景执行碰撞')
        terminated = result['collision'] or result['off_route'] or result['completed']
        if not terminated and result['executed_steps'] < result['configured_steps']:
            raise RuntimeError('冒烟未执行至终止或完整帧数')
        if not Path(result['video_path']).is_file() or not Path(feasible['execution_trace_path']).is_file():
            raise RuntimeError('冒烟缺少完整视频或执行证据')
    return {'attempt_count': len(attempts), 'passed': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['smoke', 'formal'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--smoke-dir', type=Path)
    parser.add_argument('--after', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'launcher.json', {'pid': os.getpid(), 'phase': 'queued' if args.after else 'starting',
                                        'after': str(args.after) if args.after else None})
    if args.after:
        # 排队等待前一独立实验结束，不中止正在运行的尝试。
        while not (args.after / 'exit_status.json').is_file():
            time.sleep(30)
    if not args.manifest.exists():
        freeze(args.manifest)
    manifest = json.loads(args.manifest.read_text())
    for file, expected in zip(manifest['scenario_files'], manifest['scenario_sha256']):
        if digest(Path(file)) != expected:
            raise RuntimeError('冻结场景文件发生变化')
    hashes = {file: digest(ROOT / file) for file in SOURCE_FILES}
    if args.kind == 'formal':
        if args.smoke_dir is None:
            raise RuntimeError('正式测试必须指定通过的冒烟目录')
        save(args.output / 'smoke_gate.json', validate_smoke(args.smoke_dir, hashes))
        if digest(args.manifest) != digest(args.smoke_dir / 'manifest.json'):
            raise RuntimeError('正式测试与冒烟清单不一致')
    save(args.output / 'source_hashes.json', hashes)
    save(args.output / 'manifest.json', manifest)
    count = 1 if args.kind == 'smoke' else 20
    overrides = ['sim.seed=42', '+sim.continue_after_off_route=true', f'sim.evaluation.max_scenarios={count}', 'sim.steps=400',
        f'+sim.scenario_manifest={args.manifest}',
        f'+sim.attack_request_file={ROOT}/tests/riskweaver_trajectory_instructions.txt',
        'sim.llm.provider=qwen', 'sim.llm.attack_mode=trajectory_only', 'sim.llm.obstacles.enabled=false',
        'sim.traffic_model.guidance.mode=llm_joint',
        'sim.traffic_model.iterative_adversarial.profile_enabled=true',
        'sim.traffic_model.iterative_adversarial.escalation_enabled=true',
        'sim.traffic_model.iterative_adversarial.full_method_enabled=true',
        'sim.evaluation.difficulty_control.mode=target',
        'sim.evaluation.difficulty_control.target_difficulty=0.5',
        'sim.evaluation.difficulty_control.tolerance=0.1',
        'sim.evaluation.difficulty_control.max_replays=2',
        f'sim.evaluation.difficulty_control.smoke_acceptance={str(args.kind == "smoke").lower()}',
        'sim.visualize=true', 'sim.lightweight=true',
        'sim.visualization.show_diffusion_candidates=true',
        f'sim.movie_path={args.output}/movies', f'sim.scenario_data_output_path={args.output}/carla',
        f'hydra.run.dir={args.output}/hydra']
    env = dict(os.environ, PROJECT_ROOT=str(ROOT), SCRATCH_ROOT=str(ROOT), DATASET_ROOT=str(ROOT / 'metadata'),
               PYTHONPATH=f'{ROOT}:{ROOT}/safe-sim:{ROOT}/safe-sim/trajdata/src', MPLBACKEND='Agg',
               PYTHONUNBUFFERED='1', HYDRA_FULL_ERROR='1', RISKWEAVER_DIAGNOSTIC_DIR=str(args.output / 'diagnostics'))
    cmd = ['bash', 'scripts/run_with_llm_provider.sh', 'qwen', sys.executable, 'run_simulation.py', *overrides]
    save(args.output / 'launch.json', {'command': cmd, 'pid': os.getpid(), 'started_at': time.time()})
    with (args.output / 'resolved_config.yaml').open('w') as handle:
        subprocess.run(cmd + ['--cfg', 'job', '--resolve'], cwd=ROOT, env=env, stdout=handle, check=True)
    with (args.output / 'run.log').open('w') as handle:
        code = subprocess.call(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    save(args.output / 'exit_status.json', {'exit_code': code, 'finished_at': time.time()})
    return code


if __name__ == '__main__':
    raise SystemExit(main())
