"""验证单场闭环规划冒烟，并只在全部门槛通过时写出验收文件。"""

import argparse
import json
import os
from pathlib import Path
import tempfile


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_smoke(root, expected_model):
    root = Path(root)
    result_paths = sorted(root.rglob("attempt_result.json"))
    reasons = []
    totals = {
        "attack_plan_count": 0,
        "attack_executed_frames": 0,
        "obstacle_plan_count": 0,
        "planner_service_failure_count": 0,
        "background_collision_count": 0,
        "background_static_collision_count": 0,
    }
    trace_count = 0
    video_count = 0
    successful_models = set()
    accepted_llm_plan_count = 0

    launch_path = root / "launch.json"
    launch_model = None
    launch_command = []
    if launch_path.is_file():
        launch = _load_json(launch_path)
        launch_model = launch.get("model_name")
        launch_command = [str(item) for item in launch.get("command", [])]
    else:
        reasons.append("missing_launch_config")

    if not result_paths:
        reasons.append("missing_attempt_result")

    for result_path in result_paths:
        result = _load_json(result_path)
        for key in (
            "attack_plan_count", "attack_executed_frames", "obstacle_plan_count",
            "planner_service_failure_count",
        ):
            totals[key] += int(result.get(key, 0))

        video_path = result.get("video_path")
        if video_path and Path(video_path).is_file():
            video_count += 1
        else:
            reasons.append(f"missing_video:{result_path}")

        trace_path = result_path.with_name("execution_trace.jsonl")
        if not trace_path.is_file():
            reasons.append(f"missing_execution_trace:{result_path}")
            continue
        trace_count += 1
        with trace_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("kind") == "state":
                    if row.get("static_obstacles"):
                        reasons.append(f"static_obstacles_present:{trace_path}:{line_number}")
                    totals["background_collision_count"] += len(row.get("background_pairs", []))
                    totals["background_static_collision_count"] += len(
                        row.get("background_static_pairs", [])
                    )
                elif row.get("kind") == "llm_output":
                    model = row.get("planner_trace", {}).get("successful_model")
                    if model:
                        successful_models.add(str(model))
                    plan = row.get("validated_plan") or {}
                    if (
                        bool(plan.get("attack"))
                        and not bool(row.get("request_failed"))
                        and row.get("planner_trace", {}).get("plan_validation") == "accepted"
                    ):
                        accepted_llm_plan_count += 1

    if totals["attack_plan_count"] <= 0:
        reasons.append("missing_attack_plan")
    if totals["attack_executed_frames"] <= 0:
        reasons.append("missing_attack_execution")
    if totals["obstacle_plan_count"] != 0:
        reasons.append("unexpected_obstacle_plan")
    if totals["planner_service_failure_count"] != 0:
        reasons.append("planner_service_failure")
    if totals["background_collision_count"] != 0:
        reasons.append("background_collision")
    if totals["background_static_collision_count"] != 0:
        reasons.append("background_static_collision")
    fixed_model_flags = {
        f"sim.llm.model_name={expected_model}",
        f"sim.llm.model_names=[{expected_model}]",
    }
    if launch_model != expected_model or not fixed_model_flags.issubset(set(launch_command)):
        reasons.append("launch_model_not_fixed_to_expected_model")
    if accepted_llm_plan_count <= 0:
        reasons.append("missing_accepted_llm_attack_plan")
    if successful_models and successful_models != {expected_model}:
        reasons.append(
            "unexpected_successful_models:" + ",".join(sorted(successful_models))
        )

    return {
        "accepted": not reasons,
        "model_name": expected_model,
        "attempt_count": len(result_paths),
        "trace_count": trace_count,
        "video_count": video_count,
        "accepted_llm_plan_count": accepted_llm_plan_count,
        **totals,
        "successful_models": sorted(successful_models),
        "model_identity_source": "launch_config_and_accepted_trace",
        "rejection_reasons": reasons,
        "smoke_root": str(root.resolve()),
    }


def _write_atomic(path, payload):
    """同一目录临时文件替换，避免正式 launcher 读到半写入通行证。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = validate_smoke(args.smoke_dir, args.expected_model)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["accepted"]:
        return 2
    _write_atomic(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
