"""只读汇总 RiskWeaver 正式实验，生成可追溯的论文统计表和中英文图。"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _finite(value):
    return value is not None and np.isfinite(value)


def _mean(values):
    values = [float(value) for value in values if _finite(value)]
    return float(np.mean(values)) if values else None


def _failure_reason(value):
    """保留失败类别；兼容字符串和结构化失败记录。"""
    if not value:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("reason") or value.get("type") or "unspecified")
    return str(value)


def summarize_attempts(attempts):
    """按固定分母汇总；任何无效危险度都保持为空而不是补零。"""
    failures = Counter()
    valid_danger = []
    valid_window_weight = 0
    weighted_danger = 0.0
    ttc_values = []
    ea_values = []
    ability_values = []
    executed = 0
    attack_scenes = 0
    background_events = 0
    background_static_events = 0

    for row in attempts:
        reason = _failure_reason(row.get("generation_failure"))
        if reason:
            failures[reason] += 1
        if int(row.get("executed_steps") or 0) > 0:
            executed += 1
        if int(row.get("attack_executed_frames") or 0) > 0:
            attack_scenes += 1
        if row.get("danger_valid") and _finite(row.get("scenario_danger_score")):
            valid_danger.append(float(row["scenario_danger_score"]))
        if _finite(row.get("autonomous_driving_ability_score")):
            ability_values.append(float(row["autonomous_driving_ability_score"]))

        feasibility = row.get("feasibility") or {}
        background_events += int(feasibility.get("background_collision_events") or 0)
        background_static_events += int(feasibility.get("background_static_collision_events") or 0)
        for window in row.get("attack_window_difficulties") or []:
            metrics = window.get("metrics") or {}
            if _finite(metrics.get("avg_min_ttc")):
                ttc_values.append(float(metrics["avg_min_ttc"]))
            if _finite(metrics.get("max_evasive_acceleration_mps2")):
                ea_values.append(float(metrics["max_evasive_acceleration_mps2"]))
            if window.get("valid") and _finite(window.get("difficulty")):
                weight = int(window.get("weight") or 0)
                if weight > 0:
                    valid_window_weight += weight
                    weighted_danger += float(window["difficulty"]) * weight

    count = len(attempts)
    return {
        "attempt_count": count,
        "executed_count": executed,
        "generation_failure_count": sum(failures.values()),
        "generation_failure_reasons": dict(sorted(failures.items())),
        "planner_service_failure_count": sum(int(x.get("planner_service_failure_count") or 0) for x in attempts),
        "attack_plan_count": sum(int(x.get("attack_plan_count") or 0) for x in attempts),
        "attack_scene_count": attack_scenes,
        "attack_scene_rate": attack_scenes / count if count else None,
        "attack_executed_frames": sum(int(x.get("attack_executed_frames") or 0) for x in attempts),
        "valid_danger_count": len(valid_danger),
        "valid_danger_coverage": len(valid_danger) / count if count else None,
        "danger_scene_mean": _mean(valid_danger),
        "danger_valid_frame_count": valid_window_weight,
        "danger_frame_weighted_mean": weighted_danger / valid_window_weight if valid_window_weight else None,
        "min_ttc_s": min(ttc_values) if ttc_values else None,
        "mean_max_ea_mps2": _mean(ea_values),
        "collision_count": sum(bool(x.get("collision")) for x in attempts),
        "completion_count": sum(bool(x.get("completed")) for x in attempts),
        "mean_progress": _mean(x.get("progress") for x in attempts),
        "ability_score_mean": _mean(ability_values),
        "background_collision_events": background_events,
        "background_static_collision_events": background_static_events,
    }


def summarize_group(root):
    paths = sorted(root.rglob("attempt_result.json"))
    attempts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    result = summarize_attempts(attempts)
    result["root"] = str(root.resolve())
    result["attempt_files"] = [str(path.resolve()) for path in paths]
    return result


def _scenario_key(row):
    source = row.get("scenario_source")
    return str(Path(source).name) if source else f"index:{row.get('scenario_index')}"


def _attempt_metric(row, name):
    if name == "attack_scene":
        return float(int(row.get("attack_executed_frames") or 0) > 0)
    if name == "attack_frames":
        return float(row.get("attack_executed_frames") or 0)
    if name == "danger":
        return float(row["scenario_danger_score"]) if row.get("danger_valid") and _finite(row.get("scenario_danger_score")) else None
    if name == "ability":
        return float(row["autonomous_driving_ability_score"]) if _finite(row.get("autonomous_driving_ability_score")) else None
    windows = row.get("attack_window_difficulties") or []
    if name == "ttc":
        values = [(window.get("metrics") or {}).get("avg_min_ttc") for window in windows]
        values = [float(value) for value in values if _finite(value)]
        return min(values) if values else None
    if name == "ea":
        values = [(window.get("metrics") or {}).get("max_evasive_acceleration_mps2") for window in windows]
        values = [float(value) for value in values if _finite(value)]
        return max(values) if values else None
    raise KeyError(name)


def _bootstrap_mean(values, samples=2000, seed=42):
    """以场景为抽样单位；没有有效配对时区间保持为空。"""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"mean": None, "ci95": [None, None], "pair_count": 0}
    generator = np.random.default_rng(seed)
    means = np.mean(generator.choice(values, size=(samples, len(values)), replace=True), axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "pair_count": int(len(values)),
    }


def compare_paired_roots(off_root, on_root, samples=2000, seed=42):
    """计算画像开启减去关闭的逐场差，禁止用未配对总体均值替代。"""
    def load(root):
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.rglob("attempt_result.json"))]
        return {_scenario_key(row): row for row in rows}

    off, on = load(off_root), load(on_root)
    common = sorted(set(off) & set(on))
    result = {
        "off_root": str(off_root.resolve()),
        "on_root": str(on_root.resolve()),
        "off_scenario_count": len(off),
        "on_scenario_count": len(on),
        "common_scenario_count": len(common),
        "missing_from_off": sorted(set(on) - set(off)),
        "missing_from_on": sorted(set(off) - set(on)),
        "difference_definition": "profile_on_minus_profile_off",
        "metrics": {},
    }
    for metric in ("attack_scene", "attack_frames", "danger", "ttc", "ea", "ability"):
        differences = []
        for key in common:
            before = _attempt_metric(off[key], metric)
            after = _attempt_metric(on[key], metric)
            if before is not None and after is not None:
                differences.append(after - before)
        result["metrics"][metric] = _bootstrap_mean(differences, samples=samples, seed=seed)
    return result


def _configure_chinese_font():
    candidates = [Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"), Path("C:/Windows/Fonts/msyh.ttc")]
    for path in candidates:
        if path.exists():
            from matplotlib import font_manager
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = [font_manager.FontProperties(fname=str(path)).get_name(), "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot(summaries, output, language):
    if language == "zh":
        _configure_chinese_font()
    labels = list(summaries)
    coverage = [100 * (summaries[x]["attack_scene_rate"] or 0) for x in labels]
    danger = [summaries[x]["danger_scene_mean"] or 0 for x in labels]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].bar(labels, coverage, color="#4472C4")
    axes[0].set_ylabel("实际攻击场景覆盖率（%）" if language == "zh" else "Executed attack coverage (%)")
    axes[1].bar(labels, danger, color="#C55A11")
    axes[1].set_ylabel("有效危险度均值" if language == "zh" else "Mean valid danger")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("正式实验核心指标" if language == "zh" else "Core formal-experiment metrics")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", action="append", required=True, help="显示名称=实验目录")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair", action="append", default=[], help="显示名称=画像关目录,画像开目录")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    groups = []
    for item in args.group:
        label, path = item.split("=", 1)
        groups.append((label, Path(path)))
    summaries = {label: summarize_group(path) for label, path in groups}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "formal_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    fields = [key for key in next(iter(summaries.values())) if key not in {"attempt_files", "generation_failure_reasons"}]
    with (args.output_dir / "formal_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", *fields])
        writer.writeheader()
        for label, row in summaries.items():
            writer.writerow({"group": label, **{key: row[key] for key in fields}})
    plot(summaries, args.output_dir / "正式实验核心指标.png", "zh")
    plot(summaries, args.output_dir / "formal_experiment_core_metrics.png", "en")
    paired = {}
    for item in args.pair:
        label, roots = item.split("=", 1)
        off_root, on_root = roots.split(",", 1)
        paired[label] = compare_paired_roots(
            Path(off_root), Path(on_root), samples=args.bootstrap_samples, seed=args.bootstrap_seed
        )
    if paired:
        (args.output_dir / "paired_profile_summary.json").write_text(
            json.dumps(paired, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        with (args.output_dir / "paired_profile_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["pair", "metric", "mean_difference", "ci95_low", "ci95_high", "pair_count"])
            writer.writeheader()
            for label, result in paired.items():
                for metric, value in result["metrics"].items():
                    writer.writerow({
                        "pair": label,
                        "metric": metric,
                        "mean_difference": value["mean"],
                        "ci95_low": value["ci95"][0],
                        "ci95_high": value["ci95"][1],
                        "pair_count": value["pair_count"],
                    })


if __name__ == "__main__":
    main()
