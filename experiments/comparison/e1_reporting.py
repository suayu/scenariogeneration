"""五组实验完整结果的 CSV 和配对统计；缺失观测不替换成零。"""
import csv
from pathlib import Path
import numpy as np
from experiments.comparison.e1_protocol import dump_json, GROUPS


def paired_summary(a, b, field, seed=42):
    """只对同一场景的实际有效观测配对，并披露排除数和失败状态。"""
    left = {r['scenario_index']: r for r in a}
    right = {r['scenario_index']: r for r in b}
    differences = []
    for index in sorted(left.keys() & right.keys()):
        x, y = left[index], right[index]
        # 条件结果包含部分执行，不能把它误称完整场景效果。
        if x.get('executed_steps', 0) <= 0 or y.get('executed_steps', 0) <= 0:
            continue
        if x.get(field) is None or y.get(field) is None:
            continue
        if np.isfinite(x[field]) and np.isfinite(y[field]):
            differences.append(float(x[field]) - float(y[field]))
    n = len(differences)
    result = {'paired_observed_count': n, 'difference_direction': 'RiskWeaver_minus_comparator',
              'matched_attempt_count': len(left.keys() & right.keys()),
              'basis': 'observed_execution_pairs_including_partial_failures',
              'mean_difference': None, 'bootstrap_95ci': None}
    if n:
        values = np.asarray(differences)
        bootstrap = values[np.random.default_rng(seed).integers(0, n, (5000, n))].mean(axis=1)
        result.update(mean_difference=float(values.mean()),
                      bootstrap_95ci=np.quantile(bootstrap, [.025, .975]).tolist())
    return result


def write_tables(results, directory):
    """产物保留所有尝试，D 因预测来源不同不做跨方法的配对排名。"""
    directory = Path(directory)
    fields = ['group', 'scenario_index', 'status', 'executed_steps', 'collision', 'off_route',
              'completed', 'progress', 'min_ttc_s', 'min_ttc_capped_7s',
              'mean_evasive_acceleration_mps2', 'max_evasive_acceleration_mps2',
              'ego_collision_speed_mps', 'relative_collision_speed_mps',
              'mean_reachable_area_m2', 'scenario_danger_score', 'danger_coverage',
              'reachability_source', 'attack_plan_count', 'attack_executed_frames',
              'generation_failure_reason', 'runtime_failure_reason', 'video_path']
    with (directory / 'all_attempts.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for group in GROUPS:
            writer.writerows(results[group]['episodes'])
    pairs = {}
    for group in GROUPS[:2]:
        pairs[group] = {}
        for other in GROUPS:
            if other == group:
                continue
            pairs[group][other] = {field: paired_summary(results[group]['episodes'],
                results[other]['episodes'], field) for field in ['collision', 'min_ttc_capped_7s',
                'mean_evasive_acceleration_mps2', 'off_route', 'progress']}
    dump_json(directory / 'paired_statistics.json', pairs)
