"""E1统一冒烟：每种基线仅一个冻结场景，不自动启动正式250场。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def smoke_acceptance(method, report):
    """进程退出成功不足以通过；零执行或生成/预测失败必须保留并拒绝。"""
    episodes = report.get('episodes', [])
    errors = []
    if len(episodes) != 1:
        return ['expected_exactly_one_episode']
    row = episodes[0]
    if row.get('executed_steps', 0) <= 0:
        errors.append('no_actual_execution')
    if row.get('runtime_failure'):
        errors.append('runtime_failure')
    if method == 'strive' and (row.get('attack_plan_count', 0) <= 0 or row.get('attack_executed_frames', 0) <= 0):
        errors.append('no_actual_strive_attack')
    if row.get('generation_failure'):
        errors.append('generation_failure')
    if row.get('forecast_failure_count', 0):
        errors.append('forecast_failure')
    if method == 'safe_sim' and report.get('reachability_source') == 'ctrl_sim_rollout':
        if not row.get('danger_valid'):
            errors.append('no_valid_ea_and_reachability')
    if not row.get('video_path') or not Path(row['video_path']).is_file():
        errors.append('missing_video')
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--scenario-index', type=int, default=0)
    parser.add_argument('--gpu-scenario-dreamer', default='1')
    parser.add_argument('--gpu-safe-sim', default='2')
    parser.add_argument('--safe-sim-reachability-source', choices=['native', 'ctrl_sim_rollout'], default='ctrl_sim_rollout')
    parser.add_argument('--methods', nargs='+', choices=['scenario_dreamer', 'safe_sim', 'strive'],
                        default=['scenario_dreamer', 'safe_sim', 'strive'])
    parser.add_argument('--forecast-tracking', choices=['observation', 'persistent'], default='observation')
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods):
        parser.error('duplicate method')
    root = Path(__file__).resolve().parents[2]
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    manifest = json.loads(args.manifest.read_text())
    index = args.scenario_index
    if not 0 <= index < len(manifest['scenario_files']):
        raise ValueError('invalid frozen scene index')
    scene = Path(manifest['scenario_files'][index])
    if hashlib.sha256(scene.read_bytes()).hexdigest() != manifest['scenario_sha256'][index]:
        raise ValueError('frozen scene hash mismatch')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    one = dict(manifest, scenario_files=[str(scene)],
               scenario_sha256=[manifest['scenario_sha256'][index]], scenario_count=1,
               parent_manifest=str(args.manifest), parent_scene_index=index)
    frozen = args.output_dir / 'single_scene_manifest.json'
    frozen.write_text(json.dumps(one, indent=2), encoding='utf-8')
    py = '/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python'
    strive_py = '/home2/zhaoyx/STRIVE-riskweaver/.venv/bin/python'
    commands = {
        method: [py, str(root/'experiments/comparison/run_passive_baseline.py'),
                 '--method', method, '--manifest', str(frozen), '--steps', '400',
                 '--max-scenarios', '1', '--output-dir', str(args.output_dir/method),
                 '--reachability-source', args.safe_sim_reachability_source if method == 'safe_sim' else 'native',
                 '--forecast-tracking', args.forecast_tracking]
        for method in ('scenario_dreamer', 'safe_sim')
    }
    commands['strive'] = [strive_py, str(root/'experiments/comparison/run_strive_baseline.py'),
        '--manifest', str(frozen), '--indices', '0', '--steps', '400', '--policy', 'idm',
        '--checkpoint', str(args.checkpoint), '--output-dir', str(args.output_dir/'strive')]
    launch = {'smoke_scenarios_per_method': 1, 'scenario_source': str(scene),
              'source_manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'commands': {m: commands[m] for m in args.methods}, 'methods': args.methods,
              'metric_protocol': 'ea_empty_set_zero_v2', 'forecast_tracking': args.forecast_tracking,
              'formal_started': False,
              'strive_cross_domain_audit_required': True}
    (args.output_dir/'launch.json').write_text(json.dumps(launch, indent=2))
    running, logs, results = {}, {}, {}

    def start(method, gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                   PYTHONPATH='.:safe-sim:safe-sim/trajdata/src', PYTHONUNBUFFERED='1')
        logs[method] = (args.output_dir/f'{method}.log').open('w')
        running[method] = subprocess.Popen(commands[method], cwd=root, env=env,
            stdout=logs[method], stderr=subprocess.STDOUT, start_new_session=True)
        (args.output_dir/f'{method}.pid').write_text(str(running[method].pid))

    # 两张空闲GPU运行不同方法；STRIVE复用Scenario Dreamer结束后的GPU。
    if 'scenario_dreamer' in args.methods:
        start('scenario_dreamer', args.gpu_scenario_dreamer)
    if 'safe_sim' in args.methods:
        start('safe_sim', args.gpu_safe_sim)
    if 'strive' in args.methods and 'scenario_dreamer' not in args.methods:
        start('strive', args.gpu_scenario_dreamer)
    while running:
        for method, process in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            logs[method].close()
            result_file = args.output_dir/method/'results.json'
            report = json.loads(result_file.read_text()) if result_file.exists() else {}
            errors = ['process_failure'] if code else smoke_acceptance(method, report)
            trace = list((args.output_dir/method).glob('movies/*/execution_trace.jsonl'))
            if len(trace) != 1 or trace[0].stat().st_size == 0:
                errors.append('missing_trace')
            results[method] = {'exit_code': code, 'smoke_passed': not errors,
                               'acceptance_errors': errors, 'aggregate': report.get('aggregate')}
            del running[method]
            if method == 'scenario_dreamer' and 'strive' in args.methods:
                start('strive', args.gpu_scenario_dreamer)
            (args.output_dir/'launcher_result.partial.json').write_text(json.dumps(results, indent=2))
        time.sleep(2)
    completed = {'methods': results, 'all_smokes_passed': all(r['smoke_passed'] for r in results.values()),
                 'formal_ready': False, 'formal_started': False,
                 'formal_blocker': 'STRIVE跨域地图和初始化差异仍需审计；不由冒烟自动放行250场'}
    (args.output_dir/'launcher_result.json').write_text(json.dumps(completed, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
