"""汇总 Scenario-Dreamer E1 闭环评估的逐场景工件与稳健统计。"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def finite(values):
    """过滤无有限 TTC 等不可比较值，避免无穷值污染分位数。"""
    return np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)


def bootstrap_ci(values, seed=20260902, iterations=2000):
    """对场景均值做非参数 bootstrap 95% 置信区间。"""
    values = finite(values)
    if len(values) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def describe(values):
    values = finite(values)
    if len(values) == 0:
        return {"mean": None, "p50": None, "p95": None, "bootstrap_ci95": [None, None]}
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "bootstrap_ci95": bootstrap_ci(values),
    }


def write_csv(records, path):
    """将嵌套攻击原因序列编码为 JSON，保留 CSV 的逐场景可读性。"""
    fields = [
        "scenario_index", "scenario_id", "executed_steps", "llm_call_count",
        "attack_target_id", "attack_accepted", "static_obstacle_count",
        "guidance_active_frames", "mean_diffusion_seconds", "p95_diffusion_seconds",
        "min_center_distance_m", "min_ttc_s", "collision", "off_route", "completed",
        "background_collision_frames", "background_colliding_agent_count", "video_path",
        "attack_reasons_json",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in fields}
            row["attack_reasons_json"] = json.dumps(record.get("attack_reasons", []), ensure_ascii=False)
            writer.writerow(row)


def draw_ascii_attack_audit(records, output_path, service_failed):
    """用 ASCII 文本生成跨机器可读的攻击—原因审计图。"""
    figure, (axis, text_axis) = plt.subplots(
        1, 2, figsize=(15, max(7, 0.37 * len(records) + 2)),
        gridspec_kw={"width_ratios": [1.0, 2.2]},
    )
    positions = np.arange(len(records))
    colors = ["tab:red" if record["collision"] else "0.6" for record in records]
    axis.barh(positions, np.ones(len(records)), color=colors)
    axis.set_yticks(positions, [str(record["scenario_index"]) for record in records])
    axis.set_xticks([])
    axis.set_xlim(0, 1.1)
    axis.set_xlabel("run result")
    axis.set_ylabel("scenario index")
    axis.set_title("red = ego collision; gray = no ego collision")
    lines = []
    for record in records:
        reason = "LLM quota exhausted; no attack plan returned" if service_failed else "No attack plan returned"
        if record["collision"]:
            result = "ego collision"
        else:
            result = "no ego collision"
        lines.append(f"#{record['scenario_index']:03d} | {result} | {reason}")
    text_axis.axis("off")
    text_axis.set_title("attack-reason audit (latest requested plan)", loc="left")
    text_axis.text(0.0, 0.99, "\n".join(lines), va="top", fontsize=8.5)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--runner-log", type=Path, default=None)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    records = summary["episodes"]
    output_dir = args.summary.parent
    write_csv(records, output_dir / "episodes.csv")

    attack_reason_count = sum(len(record.get("attack_reasons", [])) for record in records)
    accepted_reason_count = sum(
        sum(bool(reason.get("accepted")) for reason in record.get("attack_reasons", []))
        for record in records
    )
    # 当 LLM 服务在返回 JSON 前失败时，单独统计服务失败原因，不能伪装为攻击原因。
    failure_counts = {}
    if args.runner_log and args.runner_log.exists():
        log_text = args.runner_log.read_text(encoding="utf-8", errors="replace")
        for message in re.findall(r"高级攻击规划失败，已跳过且仿真继续：([^\n]+)", log_text):
            normalized = "LLM 请求失败：配额不足" if "insufficient_quota" in message else message[:240]
            failure_counts[normalized] = failure_counts.get(normalized, 0) + 1
        (output_dir / "llm_request_failures.json").write_text(
            json.dumps({"count": sum(failure_counts.values()), "by_reason": failure_counts}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    draw_ascii_attack_audit(records, output_dir / "attack_reason_audit_ascii.png", bool(failure_counts))
    metrics = {
        "scenario_count": len(records),
        "attack_reason_count": attack_reason_count,
        "accepted_attack_reason_count": accepted_reason_count,
        "llm_request_failure_count": sum(failure_counts.values()),
        "rates": {
            "attack_accepted": describe([float(record["attack_accepted"]) for record in records]),
            "static_obstacle_deployed": describe([float(record["static_obstacle_count"] > 0) for record in records]),
            "ego_collision": describe([float(record["collision"]) for record in records]),
            "off_route": describe([float(record["off_route"]) for record in records]),
            "completed": describe([float(record["completed"]) for record in records]),
            "background_collision_episode": describe([
                float(record["background_collision_frames"] > 0) for record in records
            ]),
        },
        "continuous_metrics": {
            "min_center_distance_m": describe([record["min_center_distance_m"] for record in records]),
            "min_ttc_s": describe([record["min_ttc_s"] for record in records]),
            "mean_diffusion_seconds": describe([record["mean_diffusion_seconds"] for record in records]),
            "background_collision_frames": describe([record["background_collision_frames"] for record in records]),
            "background_colliding_agent_count": describe([
                record["background_colliding_agent_count"] for record in records
            ]),
        },
    }
    (output_dir / "e1_metrics_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 代表案例覆盖：动态/联合攻击导致碰撞、非目标交通冲突、可行联合攻击与拒绝计划。
    collisions = [record for record in records if record["collision"]]
    selected = {
        "collision_with_accepted_attack": next(
            (record for record in collisions if record["attack_accepted"]),
            collisions[0] if collisions else None,
        ),
        "max_background_collision": max(
            records, key=lambda record: record["background_collision_frames"], default=None
        ),
        "joint_attack_without_ego_collision": next(
            (record for record in records if record["attack_accepted"] and record["static_obstacle_count"] and not record["collision"]),
            None,
        ),
        "rejected_or_no_attack": next(
            (record for record in records if not record["attack_accepted"]), None
        ),
    }
    compact = {
        key: None if record is None else {
            field: record.get(field) for field in (
                "scenario_index", "scenario_id", "collision", "attack_accepted",
                "attack_target_id", "static_obstacle_count", "min_ttc_s",
                "background_collision_frames", "background_colliding_agent_count",
                "video_path", "attack_reasons",
            )
        }
        for key, record in selected.items()
    }
    (output_dir / "representative_cases.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
