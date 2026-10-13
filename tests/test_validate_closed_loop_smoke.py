import json
from pathlib import Path

from scripts.validate_closed_loop_smoke import validate_smoke


def _write_attempt(root, *, plans=1, frames=3, service_failures=0, model="model-a", collision=False):
    launch = {
        "model_name": "model-a",
        "command": ["sim.llm.model_name=model-a", "sim.llm.model_names=[model-a]"],
    }
    (root / "launch.json").write_text(json.dumps(launch), encoding="utf-8")
    attempt = root / "movies/scenario_000/attempt_0"
    attempt.mkdir(parents=True)
    video = attempt / "scenario_000.mp4"
    video.write_bytes(b"video")
    result = {
        "attack_plan_count": plans,
        "attack_executed_frames": frames,
        "obstacle_plan_count": 0,
        "planner_service_failure_count": service_failures,
        "video_path": str(video),
    }
    (attempt / "attempt_result.json").write_text(json.dumps(result), encoding="utf-8")
    rows = [
        {
            "kind": "llm_output",
            "planner_trace": {"successful_model": model, "plan_validation": "accepted"},
            "validated_plan": {"attack": plans > 0},
            "request_failed": False,
        },
        {
            "kind": "state",
            "static_obstacles": [],
            "background_pairs": [[1, 2]] if collision else [],
            "background_static_pairs": [],
        },
    ]
    (attempt / "execution_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_validate_smoke_accepts_complete_safe_attack(tmp_path):
    _write_attempt(tmp_path)
    result = validate_smoke(tmp_path, "model-a")
    assert result["accepted"] is True
    assert result["attack_plan_count"] == 1
    assert result["attack_executed_frames"] == 3
    assert result["rejection_reasons"] == []


def test_validate_smoke_accepts_fixed_launch_when_profile_trace_omits_model(tmp_path):
    _write_attempt(tmp_path, model=None)
    result = validate_smoke(tmp_path, "model-a")
    assert result["accepted"] is True
    assert result["successful_models"] == []
    assert result["accepted_llm_plan_count"] == 1


def test_validate_smoke_rejects_no_attack_and_wrong_model(tmp_path):
    _write_attempt(tmp_path, plans=0, frames=0, model="model-b")
    result = validate_smoke(tmp_path, "model-a")
    assert result["accepted"] is False
    assert "missing_attack_plan" in result["rejection_reasons"]
    assert "missing_attack_execution" in result["rejection_reasons"]
    assert "unexpected_successful_models:model-b" in result["rejection_reasons"]


def test_validate_smoke_rejects_service_and_background_collision(tmp_path):
    _write_attempt(tmp_path, service_failures=1, collision=True)
    result = validate_smoke(tmp_path, "model-a")
    assert result["accepted"] is False
    assert result["planner_service_failure_count"] == 1
    assert result["background_collision_count"] == 1


def test_validate_smoke_rejects_missing_artifacts(tmp_path):
    (tmp_path / "launch.json").write_text(
        json.dumps({
            "model_name": "model-a",
            "command": ["sim.llm.model_name=model-a", "sim.llm.model_names=[model-a]"],
        }),
        encoding="utf-8",
    )
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / "attempt_result.json").write_text(
        json.dumps({"attack_plan_count": 1, "attack_executed_frames": 1}), encoding="utf-8"
    )
    result = validate_smoke(tmp_path, "model-a")
    assert result["accepted"] is False
    assert any(reason.startswith("missing_video:") for reason in result["rejection_reasons"])
    assert any(reason.startswith("missing_execution_trace:") for reason in result["rejection_reasons"])
