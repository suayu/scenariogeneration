import json
import math

from scripts.summarize_formal_experiments import compare_paired_roots, summarize_attempts


def test_summary_preserves_invalid_danger_and_fixed_denominators():
    rows = [
        {
            "executed_steps": 100,
            "attack_plan_count": 2,
            "attack_executed_frames": 10,
            "danger_valid": True,
            "scenario_danger_score": 0.4,
            "autonomous_driving_ability_score": 30.0,
            "completed": True,
            "progress": 1.0,
            "feasibility": {"background_collision_events": 0, "background_static_collision_events": 0},
            "attack_window_difficulties": [{"valid": True, "difficulty": 0.4, "weight": 10, "metrics": {"avg_min_ttc": 2.0, "max_evasive_acceleration_mps2": 3.0}}],
        },
        {
            "executed_steps": 50,
            "attack_plan_count": 0,
            "attack_executed_frames": 0,
            "danger_valid": False,
            "scenario_danger_score": None,
            "completed": False,
            "progress": 0.5,
            "attack_window_difficulties": [{"valid": False, "difficulty": None, "weight": 50, "metrics": {"avg_min_ttc": 5.0, "max_evasive_acceleration_mps2": 1.0}}],
        },
    ]
    result = summarize_attempts(rows)
    assert result["attempt_count"] == 2
    assert result["attack_scene_count"] == 1
    assert result["attack_scene_rate"] == 0.5
    assert result["valid_danger_count"] == 1
    assert result["valid_danger_coverage"] == 0.5
    assert result["danger_scene_mean"] == 0.4
    assert result["danger_valid_frame_count"] == 10
    assert result["danger_frame_weighted_mean"] == 0.4
    assert result["min_ttc_s"] == 2.0
    assert result["mean_max_ea_mps2"] == 2.0
    assert result["ability_score_mean"] == 30.0


def test_empty_and_failures_are_explicit():
    empty = summarize_attempts([])
    assert empty["danger_scene_mean"] is None
    assert empty["danger_frame_weighted_mean"] is None
    assert empty["attack_scene_rate"] is None

    result = summarize_attempts([
        {"generation_failure": {"reason": "no_safe_joint_candidate"}, "planner_service_failure_count": 0},
        {"generation_failure": "planner_service_failure", "planner_service_failure_count": 1},
    ])
    assert result["generation_failure_count"] == 2
    assert result["generation_failure_reasons"] == {"no_safe_joint_candidate": 1, "planner_service_failure": 1}
    assert result["planner_service_failure_count"] == 1
    assert math.isclose(result["valid_danger_coverage"], 0.0)


def test_paired_comparison_uses_common_scenes_and_valid_pairs(tmp_path):
    off, on = tmp_path / "off", tmp_path / "on"
    for root in (off, on):
        root.mkdir()

    def write(root, name, payload):
        path = root / name
        path.mkdir()
        (path / "attempt_result.json").write_text(json.dumps(payload), encoding="utf-8")

    write(off, "a", {"scenario_source": "/x/a.pkl", "attack_executed_frames": 0, "danger_valid": True, "scenario_danger_score": 0.1})
    write(on, "a", {"scenario_source": "/y/a.pkl", "attack_executed_frames": 10, "danger_valid": True, "scenario_danger_score": 0.3})
    write(off, "b", {"scenario_source": "/x/b.pkl", "attack_executed_frames": 5, "danger_valid": False, "scenario_danger_score": None})
    write(on, "b", {"scenario_source": "/y/b.pkl", "attack_executed_frames": 5, "danger_valid": True, "scenario_danger_score": 0.8})
    write(on, "c", {"scenario_source": "/y/c.pkl", "attack_executed_frames": 1})

    result = compare_paired_roots(off, on, samples=200, seed=7)
    assert result["common_scenario_count"] == 2
    assert result["missing_from_off"] == ["c.pkl"]
    assert result["metrics"]["attack_scene"]["mean"] == 0.5
    assert result["metrics"]["attack_frames"]["mean"] == 5.0
    assert result["metrics"]["danger"]["pair_count"] == 1
    assert math.isclose(result["metrics"]["danger"]["mean"], 0.2)
