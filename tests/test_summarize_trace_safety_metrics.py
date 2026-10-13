import json

from scripts.summarize_trace_safety_metrics import summarize_group, trace_metrics


def _state(step, ego_x, agent_x, collision=False):
    return {
        "kind": "state",
        "step": step,
        "ego": [ego_x, 0.0, 2.0, 0.0, 0.0, 4.0, 2.0, 1.0],
        "agents": [[agent_x, 0.0, 0.0, 0.0, 0.0, 4.0, 2.0, 1.0]],
        "active": [True],
        "info": {"collision": collision, "obstacle_collision_ids": []},
    }


def _write_attempt(root, index, collision, with_trace=True):
    scene = root / f"scenario_{index:03d}"
    scene.mkdir(parents=True)
    attempt = {
        "scenario_index": index,
        "scenario_source": f"scene_{index}.pkl",
        "generation_failure": not with_trace,
        "collision": collision if with_trace else None,
        "off_route": False if with_trace else None,
        "completed": not collision if with_trace else None,
        "progress": 1.0 if with_trace else None,
    }
    (scene / "attempt_result.json").write_text(json.dumps(attempt), encoding="utf-8")
    if with_trace:
        rows = [_state(0, 0.0, 8.0), _state(1, 0.2, 3.9, collision=collision)]
        (scene / "execution_trace.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return scene, attempt


def test_trace_metrics_reports_contact_relative_speed(tmp_path):
    scene, attempt = _write_attempt(tmp_path, 0, collision=True)
    result = trace_metrics(scene / "execution_trace.jsonl", attempt)
    assert result["min_obb_ttc_s"] == 0.0
    assert result["collision_relative_speed_mps"] == 2.0
    assert result["collision_contact_count"] == 1


def test_group_preserves_failure_denominator_and_missing_speed(tmp_path):
    _write_attempt(tmp_path, 0, collision=True)
    _write_attempt(tmp_path, 1, collision=False)
    _write_attempt(tmp_path, 2, collision=False, with_trace=False)
    result = summarize_group(tmp_path)
    assert result["attempted_scenario_count"] == 3
    assert result["executed_scenario_count"] == 2
    assert result["generation_failure_count"] == 1
    assert result["collision_rate_attempted"] == 1 / 3
    assert result["collision_rate_executed"] == 1 / 2
    assert result["mean_collision_relative_speed_mps"] == 2.0
