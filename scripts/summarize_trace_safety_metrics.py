"""从统一 execution trace 补算 TTC、碰撞相对速度和结果率。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from policies.collision_ttc import minimum_obb_ttc
from policies.joint_safety import obb_clearance


def _finite_mean(values):
    """空集合保持缺失，禁止用零替代无效测量。"""
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _state_boxes(states):
    """把统一状态 `[x,y,vx,vy,yaw,length,width,...]` 转为 OBB。"""
    states = np.asarray(states, dtype=np.float64)
    return states[..., [0, 1, 4, 5, 6]]


def trace_metrics(trace_path, attempt):
    """计算单场最小 OBB-TTC 与碰撞瞬间接触车辆相对速度。"""
    minimum_ttc = float("inf")
    collision_speeds = []
    terminal_collision_state_found = False
    with Path(trace_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{trace_path}:{line_number} 不是有效 JSON") from exc
            if row.get("kind") != "state" or not row.get("ego"):
                continue
            ego = np.asarray(row["ego"], dtype=np.float64)
            agents = np.asarray(row.get("agents") or [], dtype=np.float64)
            active = np.asarray(row.get("active") or [], dtype=bool)
            if agents.ndim != 2 or len(active) != len(agents):
                continue
            current_ttc = minimum_obb_ttc(ego, agents, active)
            if np.isfinite(current_ttc):
                minimum_ttc = min(minimum_ttc, float(current_ttc))

            info = row.get("info") or {}
            if not bool(info.get("collision")):
                continue
            terminal_collision_state_found = True
            if info.get("obstacle_collision_ids"):
                # 静态障碍没有速度，车辆相对碰撞速度不适用。
                continue
            active_agents = agents[active]
            if not len(active_agents):
                continue
            ego_box = _state_boxes(ego)
            clearances = obb_clearance(
                np.broadcast_to(ego_box, (len(active_agents), 5)),
                _state_boxes(active_agents),
            )
            contact_indices = np.flatnonzero(clearances <= 1e-6)
            for index in contact_indices:
                collision_speeds.append(
                    float(np.linalg.norm(active_agents[index, 2:4] - ego[2:4]))
                )

    return {
        "min_obb_ttc_s": float(minimum_ttc) if np.isfinite(minimum_ttc) else None,
        "collision_relative_speed_mps": (
            max(collision_speeds) if collision_speeds else None
        ),
        "collision_contact_count": len(collision_speeds),
        "terminal_collision_state_found": terminal_collision_state_found,
        "collision_speed_reason": (
            None
            if collision_speeds
            else (
                "not_applicable_no_collision"
                if not attempt.get("collision")
                else "no_dynamic_obb_contact_in_terminal_trace"
            )
        ),
    }


def summarize_group(root):
    """汇总全部尝试；生成失败保留在 attempted 分母中。"""
    root = Path(root)
    rows = []
    for attempt_path in sorted(root.rglob("attempt_result.json")):
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        trace_path = attempt_path.parent / "execution_trace.jsonl"
        trace = trace_metrics(trace_path, attempt) if trace_path.is_file() else {
            "min_obb_ttc_s": None,
            "collision_relative_speed_mps": None,
            "collision_contact_count": 0,
            "terminal_collision_state_found": False,
            "collision_speed_reason": "missing_execution_trace",
        }
        rows.append(
            {
                "scenario_index": attempt.get("scenario_index"),
                "scenario_source": attempt.get("scenario_source"),
                "generation_failure": bool(attempt.get("generation_failure")),
                "collision": attempt.get("collision"),
                "off_route": attempt.get("off_route"),
                "completed": attempt.get("completed"),
                "progress": attempt.get("progress"),
                "trace_path": str(trace_path) if trace_path.is_file() else None,
                **trace,
            }
        )
    executed = [row for row in rows if not row["generation_failure"]]
    collisions = [row for row in executed if row["collision"]]
    attempted_count = len(rows)
    executed_count = len(executed)
    return {
        "experiment_root": str(root),
        "attempted_scenario_count": attempted_count,
        "executed_scenario_count": executed_count,
        "generation_failure_count": attempted_count - executed_count,
        "collision_count": len(collisions),
        "collision_rate_attempted": len(collisions) / attempted_count if attempted_count else None,
        "collision_rate_executed": len(collisions) / executed_count if executed_count else None,
        "off_route_count": sum(bool(row["off_route"]) for row in executed),
        "off_route_rate_executed": (
            sum(bool(row["off_route"]) for row in executed) / executed_count
            if executed_count else None
        ),
        "mean_min_obb_ttc_s": _finite_mean([row["min_obb_ttc_s"] for row in executed]),
        "valid_min_obb_ttc_count": sum(row["min_obb_ttc_s"] is not None for row in executed),
        "mean_collision_relative_speed_mps": _finite_mean(
            [row["collision_relative_speed_mps"] for row in collisions]
        ),
        "valid_collision_speed_count": sum(
            row["collision_relative_speed_mps"] is not None for row in collisions
        ),
        "episodes": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", action="append", required=True, help="格式：显示名称=实验目录")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    groups = {}
    for item in args.group:
        if "=" not in item:
            parser.error("--group 必须使用 显示名称=实验目录 格式")
        label, path = item.split("=", 1)
        if not label or label in groups:
            parser.error("--group 显示名称不能为空或重复")
        groups[label] = summarize_group(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "trace_safety_metrics.json").write_text(
        json.dumps(groups, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    fields = [
        "group", "scenario_index", "scenario_source", "generation_failure", "collision",
        "off_route", "completed", "progress", "min_obb_ttc_s",
        "collision_relative_speed_mps", "collision_contact_count",
        "collision_speed_reason", "trace_path",
    ]
    with (args.output_dir / "trace_safety_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, result in groups.items():
            for row in result["episodes"]:
                writer.writerow({"group": label, **{key: row.get(key) for key in fields if key != "group"}})


if __name__ == "__main__":
    main()
