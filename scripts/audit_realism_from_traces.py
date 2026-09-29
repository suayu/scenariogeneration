"""从既有闭环执行 trace 离线计算 SAFE-SIM 口径的真实性指标。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from policies.realism_audit import _DomainAccumulator, TrajectoryRealismAudit
from scripts.summarize_realism_audit import METRICS, plot


def _trace_paths(root):
    """返回稳定排序的 trace 文件列表，输入也可以是单个 trace 文件。"""
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(root.rglob("execution_trace.jsonl"))


def audit_group(root, reference_path, dt=0.1, minimum_samples=8):
    """聚合一个实验目录中的背景车轨迹，并与冻结参考分布比较。"""
    audit = TrajectoryRealismAudit(
        {
            "enabled": True,
            "reference_histogram_path": str(reference_path),
            "minimum_samples": int(minimum_samples),
        },
        dt=float(dt),
    )
    accumulator = _DomainAccumulator()
    trace_paths = _trace_paths(root)
    state_row_count = 0
    observed_agent_state_count = 0

    for trace_path in trace_paths:
        # 文件路径进入轨迹键，防止不同场景中复用的 agent_id 被错误拼接。
        trace_key = str(trace_path.resolve())
        with trace_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{trace_path}:{line_number} 不是有效 JSON") from exc
                if row.get("kind") != "state":
                    continue
                step = int(row["step"])
                agents = row.get("agents", [])
                active = row.get("active", [])
                state_row_count += 1
                for agent_id, state in enumerate(agents[: len(active)]):
                    if not bool(active[agent_id]) or len(state) < 5:
                        continue
                    values = np.asarray(state[:5], dtype=np.float64)
                    if not np.isfinite(values).all():
                        continue
                    accumulator.observe((trace_key, agent_id), step, values, audit.dt)
                    observed_agent_state_count += 1

    audit._simulated = accumulator
    # finish_episode 只使用该集合的长度；离线模式用全局行号表达聚合帧数。
    audit._observed_steps = set(range(state_row_count))
    result = audit.finish_episode()
    result.update(
        {
            "trace_count": len(trace_paths),
            "state_row_count": state_row_count,
            "observed_agent_state_count": observed_agent_state_count,
            "experiment_root": str(Path(root)),
        }
    )
    return result


def _plot_summary(results):
    """转换为现有中英文画图函数使用的聚合结构。"""
    summaries = {}
    for label, audit in results.items():
        available = [
            name for name in METRICS
            if name in audit["histograms"]["reference_probability"]
            and sum(audit["histograms"]["simulated_counts"].get(name, [])) > 0
        ]
        summaries[label] = {
            "simulated_counts": {
                name: audit["histograms"]["simulated_counts"][name] for name in available
            },
            "reference_probability": {
                name: audit["histograms"]["reference_probability"][name] for name in available
            },
            "histogram_edges": {name: audit["histogram_edges"][name] for name in available},
        }
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", action="append", required=True, help="格式：显示名称=实验目录")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--minimum-samples", type=int, default=8)
    args = parser.parse_args()

    groups = []
    for item in args.group:
        if "=" not in item:
            parser.error("--group 必须使用 显示名称=实验目录 格式")
        label, path = item.split("=", 1)
        if not label:
            parser.error("--group 显示名称不能为空")
        groups.append((label, Path(path)))

    results = {
        label: audit_group(path, args.reference, args.dt, args.minimum_samples)
        for label, path in groups
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "realism_trace_audit.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "group", "trace_count", "state_row_count", "observed_agent_state_count",
        "valid", "reason", "realism_deviation", *[f"wasserstein_{name}" for name in METRICS],
    ]
    with (args.output_dir / "realism_trace_audit.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, result in results.items():
            row = {key: result.get(key) for key in fields}
            row["group"] = label
            for name in METRICS:
                row[f"wasserstein_{name}"] = result["wasserstein"].get(name)
            writer.writerow(row)

    if results:
        summaries = _plot_summary(results)
        plot(summaries, args.output_dir / "真实性分布对比.png", "zh")
        plot(summaries, args.output_dir / "realism_distribution_comparison.png", "en")


if __name__ == "__main__":
    main()
