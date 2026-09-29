"""逐场计算 SAFE-SIM 兼容真实性，并对固定场景做配对统计。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.audit_realism_from_traces import audit_group


METRICS = ("lon_accel", "lat_accel", "jerk")


def _read_json(path):
    """读取可选审计文件；缺失时返回空字典。"""
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _scenario_key(trace_path, root, attempt):
    """优先使用冻结清单索引，路径仅作为旧产物的兼容后备。"""
    if attempt.get("scenario_index") is not None:
        return str(int(attempt["scenario_index"]))
    return str(trace_path.parent.relative_to(root))


def bootstrap_mean_interval(values, iterations=10000, seed=42):
    """使用固定随机种子计算场景等权均值和 percentile 置信区间。"""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(int(seed))
    samples = rng.choice(array, size=(int(iterations), len(array)), replace=True)
    boot_means = samples.mean(axis=1)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "ci95": [
            float(np.quantile(boot_means, 0.025)),
            float(np.quantile(boot_means, 0.975)),
        ],
    }


def collect_group(root, reference, dt=0.1, minimum_samples=8, bootstrap_iterations=10000):
    """逐个 trace 计算真实性，避免长场景在全局直方图中获得额外权重。"""
    root = Path(root)
    traces = sorted(root.rglob("execution_trace.jsonl")) if root.exists() else []
    attempts = sorted(root.rglob("attempt_result.json")) if root.exists() else []
    scenes = {}
    for trace_path in traces:
        attempt = _read_json(trace_path.parent / "attempt_result.json")
        key = _scenario_key(trace_path, root, attempt)
        if key in scenes:
            raise ValueError(f"组内场景键重复：{key}")
        audit = audit_group(
            trace_path,
            reference,
            dt=float(dt),
            minimum_samples=int(minimum_samples),
        )
        scenes[key] = {
            "scenario_key": key,
            "scenario_index": attempt.get("scenario_index"),
            "scenario_source": attempt.get("scenario_source"),
            "trace_path": str(trace_path),
            "valid": bool(audit["valid"]),
            "reason": audit.get("reason"),
            "realism_deviation": audit.get("realism_deviation"),
            "wasserstein": audit.get("wasserstein"),
            "sample_counts": audit.get("sample_counts"),
            "range_audit": audit.get("range_audit"),
        }

    valid = [row for row in scenes.values() if row["valid"]]
    attempted_count = len(attempts)
    denominator = attempted_count if attempted_count else len(scenes)
    metric_stats = {}
    for metric in METRICS:
        metric_stats[metric] = bootstrap_mean_interval(
            [row["wasserstein"][metric] for row in valid],
            iterations=bootstrap_iterations,
        )
    overall = bootstrap_mean_interval(
        [row["realism_deviation"] for row in valid],
        iterations=bootstrap_iterations,
    )
    return {
        "experiment_root": str(root),
        "attempt_count": attempted_count,
        "trace_count": len(traces),
        "valid_scene_count": len(valid),
        "valid_scene_coverage": len(valid) / denominator if denominator else 0.0,
        "aggregation": "equal_weight_mean_of_per_scene_SAFE_SIM_histogram_Wasserstein",
        "realism_deviation": overall,
        "wasserstein": metric_stats,
        "scenes": scenes,
    }


def paired_comparison(baseline, candidate, bootstrap_iterations=10000):
    """只在双方同一冻结场景都有有效测量时计算配对差。"""
    baseline_scenes = baseline["scenes"]
    candidate_scenes = candidate["scenes"]
    common_keys = sorted(set(baseline_scenes) & set(candidate_scenes))
    source_mismatch_keys = []
    source_matched_keys = []
    for key in common_keys:
        baseline_source = baseline_scenes[key].get("scenario_source")
        candidate_source = candidate_scenes[key].get("scenario_source")
        if baseline_source and candidate_source and baseline_source != candidate_source:
            source_mismatch_keys.append(key)
        else:
            source_matched_keys.append(key)
    valid_keys = [
        key for key in source_matched_keys
        if baseline_scenes[key]["valid"] and candidate_scenes[key]["valid"]
    ]
    deltas = [
        candidate_scenes[key]["realism_deviation"]
        - baseline_scenes[key]["realism_deviation"]
        for key in valid_keys
    ]
    stats = bootstrap_mean_interval(deltas, iterations=bootstrap_iterations)
    return {
        "common_scene_count": len(common_keys),
        "source_matched_scene_count": len(source_matched_keys),
        "source_mismatch_scene_keys": source_mismatch_keys,
        "paired_valid_scene_count": len(valid_keys),
        "candidate_minus_baseline": stats,
        "interpretation": "negative_favors_candidate",
        "paired_scene_keys": valid_keys,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", action="append", required=True, help="格式：显示名称=实验目录")
    parser.add_argument("--baseline", required=True, help="用于配对差的基线显示名称")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--minimum-samples", type=int, default=8)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()

    groups = []
    for item in args.group:
        if "=" not in item:
            parser.error("--group 必须使用 显示名称=实验目录 格式")
        label, path = item.split("=", 1)
        if not label or any(label == old_label for old_label, _ in groups):
            parser.error("--group 显示名称不能为空或重复")
        groups.append((label, Path(path)))
    labels = {label for label, _ in groups}
    if args.baseline not in labels:
        parser.error("--baseline 必须与一个 --group 显示名称一致")

    results = {
        label: collect_group(
            path,
            args.reference,
            dt=args.dt,
            minimum_samples=args.minimum_samples,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        for label, path in groups
    }
    comparisons = {
        label: paired_comparison(
            results[args.baseline], result, args.bootstrap_iterations
        )
        for label, result in results.items()
        if label != args.baseline
    }
    payload = {
        "schema_version": 1,
        "reference": str(args.reference),
        "dt_seconds": args.dt,
        "minimum_samples": args.minimum_samples,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": 42,
        "baseline": args.baseline,
        "groups": results,
        "paired_comparisons": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per_scene_realism_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with (args.output_dir / "per_scene_realism_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = [
            "group", "scenario_key", "scenario_index", "scenario_source", "valid",
            "reason", "realism_deviation", "wasserstein_lon_accel",
            "wasserstein_lat_accel", "wasserstein_jerk", "trace_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, result in results.items():
            for row in result["scenes"].values():
                writer.writerow(
                    {
                        "group": label,
                        "scenario_key": row["scenario_key"],
                        "scenario_index": row["scenario_index"],
                        "scenario_source": row["scenario_source"],
                        "valid": row["valid"],
                        "reason": row["reason"],
                        "realism_deviation": row["realism_deviation"],
                        "wasserstein_lon_accel": (row["wasserstein"] or {}).get("lon_accel"),
                        "wasserstein_lat_accel": (row["wasserstein"] or {}).get("lat_accel"),
                        "wasserstein_jerk": (row["wasserstein"] or {}).get("jerk"),
                        "trace_path": row["trace_path"],
                    }
                )


if __name__ == "__main__":
    main()
