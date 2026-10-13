"""正式被动基线运行器的清单和汇总口径回归。"""

import hashlib
import json
import os

import pytest

from experiments.comparison.run_passive_baseline import (
    aggregate, csv_fieldnames, rejected_episode, run_scenarios, sanitize_nonfinite,
    validate_manifest,
)
from policies.joint_safety import NoSafeJointCandidate


def test_standalone_runner_defines_project_root():
    assert os.environ["PROJECT_ROOT"]
    assert os.environ["SCRATCH_ROOT"]
    assert os.environ["DATASET_ROOT"].endswith("metadata")



def test_csv_fieldnames_covers_mixed_success_and_generation_failure_records():
    rows = [
        {"scenario_index": 0, "status": "completed"},
        {
            "scenario_index": 1,
            "status": "generation_failure",
            "generation_failure_reason": "no_safe_joint_candidate",
            "execution_trace_path": "trace.jsonl",
        },
    ]
    assert csv_fieldnames(rows) == [
        "scenario_index",
        "status",
        "generation_failure_reason",
        "execution_trace_path",
    ]


def test_validate_manifest_checks_frozen_hashes(tmp_path):
    scene = tmp_path / "scene.pkl"
    scene.write_bytes(b"frozen-scene")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "scenario_files": [str(scene)],
        "scenario_sha256": [hashlib.sha256(scene.read_bytes()).hexdigest()],
    }), encoding="utf-8")
    loaded, files = validate_manifest(manifest)
    assert loaded["scenario_files"] == [str(scene)]
    assert files == [scene]

    scene.write_bytes(b"changed")
    with pytest.raises(ValueError, match="场景哈希不一致"):
        validate_manifest(manifest)


def test_aggregate_preserves_invalid_danger_as_missing():
    records = [
        {
            "collision": False, "off_route": False, "completed": True,
            "progress": 1.0, "min_ttc_s": 2.0,
            "max_evasive_acceleration_mps2": 3.0,
            "mean_evasive_acceleration_mps2": 2.0,
            "danger_valid": True, "scenario_danger_score": 0.4,
            "ability_valid": True, "autonomous_driving_ability_score": 40.0,
            "feasibility": {"background_collision_frames": 0,
                            "background_static_collision_frames": 0},
        },
        {
            "collision": True, "off_route": False, "completed": False,
            "progress": 0.3, "min_ttc_s": None,
            "max_evasive_acceleration_mps2": None,
            "mean_evasive_acceleration_mps2": None,
            "danger_valid": False, "scenario_danger_score": None,
            "ability_valid": False, "autonomous_driving_ability_score": None,
            "feasibility": {"background_collision_frames": 0,
                            "background_static_collision_frames": 0},
        },
    ]
    result = aggregate(records)
    assert result["danger_valid_count"] == 1
    assert result["danger_valid_coverage"] == 0.5
    assert result["mean_valid_danger"] == 0.4
    assert result["ability_valid_count"] == 1


def test_sanitize_nonfinite_recurses_without_fabricating_zero():
    raw = {
        "finite": 1.25,
        "nested": [{"ttc": float("inf")}, float("-inf"), float("nan")],
        "tuple": (2.0, float("inf")),
    }
    cleaned = sanitize_nonfinite(raw)
    assert cleaned == {
        "finite": 1.25,
        "nested": [{"ttc": None}, None, None],
        "tuple": [2.0, None],
    }
    # 严格 JSON 模式应能直接序列化，且无效测量不得被改写成 0。
    json.dumps(cleaned, allow_nan=False)


def test_aggregate_counts_generation_failure_without_fabricating_outcome():
    records = [
        {
            "generation_failure": False,
            "collision": True, "off_route": False, "completed": False,
            "progress": 0.2, "min_ttc_s": 1.0,
            "max_evasive_acceleration_mps2": 4.0,
            "mean_evasive_acceleration_mps2": 2.0,
            "danger_valid": True, "scenario_danger_score": 0.7,
            "ability_valid": True, "autonomous_driving_ability_score": 14.0,
            "feasibility": {"background_collision_frames": 0,
                            "background_static_collision_frames": 0},
        },
        {
            "generation_failure": True,
            "collision": None, "off_route": None, "completed": None,
            "progress": None, "min_ttc_s": None,
            "max_evasive_acceleration_mps2": None,
            "mean_evasive_acceleration_mps2": None,
            "danger_valid": False, "scenario_danger_score": None,
            "ability_valid": False, "autonomous_driving_ability_score": None,
            "feasibility": {"background_collision_frames": 0,
                            "background_static_collision_frames": 0},
        },
    ]
    result = aggregate(records)
    assert result["attempted_scenario_count"] == 2
    assert result["executed_scenario_count"] == 1
    assert result["generation_failure_count"] == 1
    assert result["collision_rate"] == 1.0
    assert result["danger_valid_coverage"] == 0.5


def test_run_scenarios_continues_only_after_joint_candidate_rejection(
    monkeypatch, tmp_path
):
    class Env:
        test_files = ["a.json", "b.json"]

    calls = []

    def fake_run_episode(env, policy, index, steps, output_dir, expect_joint):
        calls.append(index)
        if index == 0:
            raise NoSafeJointCandidate("几何安全=0/7，有限评分=7/7")
        return {"scenario_index": index}

    monkeypatch.setattr(
        "experiments.comparison.run_passive_baseline.run_episode",
        fake_run_episode,
    )
    records = run_scenarios(Env(), object(), 2, 400, tmp_path, True)
    assert calls == [0, 1]
    assert records[0]["generation_failure"] is True
    assert records[0]["collision"] is None
    assert "几何安全=0/7" in records[0]["generation_failure_reason"]
    assert records[1]["generation_failure"] is False
    assert (tmp_path / "movies/scenario_000/attempt_result.json").exists()
