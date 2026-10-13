import json

import numpy as np

from scripts.audit_realism_per_scene import (
    bootstrap_mean_interval,
    collect_group,
    paired_comparison,
)


def _write_reference(path):
    edges = {
        "velocity": np.linspace(0.0, 30.0, 21).tolist(),
        "lon_accel": np.linspace(0.0, 10.0, 21).tolist(),
        "lat_accel": np.linspace(0.0, 10.0, 21).tolist(),
        "jerk": np.linspace(0.0, 20.0, 21).tolist(),
    }
    path.write_text(
        json.dumps(
            {
                "dt_seconds": 0.1,
                "ticks": edges,
                "stats": {name: [1.0] * 20 for name in edges},
            }
        ),
        encoding="utf-8",
    )


def _state(step, acceleration):
    time = step * 0.1
    x = 2.0 * time + 0.5 * acceleration * time * time
    return {
        "kind": "state",
        "step": step,
        "agents": [[x, 0.0, 0.0, 0.0, 0.0]],
        "active": [True],
    }


def _write_scene(root, index, acceleration=0.5, frames=12, with_trace=True):
    scene = root / f"scenario_{index:03d}"
    scene.mkdir(parents=True)
    (scene / "attempt_result.json").write_text(
        json.dumps({"scenario_index": index, "scenario_source": f"scene_{index}.pkl"}),
        encoding="utf-8",
    )
    if with_trace:
        rows = [_state(step, acceleration) for step in range(frames)]
        (scene / "execution_trace.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )


def test_collect_group_uses_equal_scene_weights_and_reports_coverage(tmp_path):
    reference = tmp_path / "reference.json"
    _write_reference(reference)
    root = tmp_path / "group"
    _write_scene(root, 0, acceleration=0.5)
    _write_scene(root, 1, acceleration=1.0)
    _write_scene(root, 2, with_trace=False)

    result = collect_group(
        root, reference, minimum_samples=1, bootstrap_iterations=100
    )

    assert result["attempt_count"] == 3
    assert result["trace_count"] == 2
    assert result["valid_scene_count"] == 2
    assert result["valid_scene_coverage"] == 2 / 3
    expected = np.mean(
        [row["realism_deviation"] for row in result["scenes"].values()]
    )
    assert result["realism_deviation"]["mean"] == expected


def test_paired_comparison_uses_only_matching_valid_scenes(tmp_path):
    reference = tmp_path / "reference.json"
    _write_reference(reference)
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    for index in range(2):
        _write_scene(baseline_root, index, acceleration=0.5 + index)
        _write_scene(candidate_root, index, acceleration=0.5 + index)
    baseline = collect_group(
        baseline_root, reference, minimum_samples=1, bootstrap_iterations=100
    )
    candidate = collect_group(
        candidate_root, reference, minimum_samples=1, bootstrap_iterations=100
    )

    comparison = paired_comparison(baseline, candidate, bootstrap_iterations=100)

    assert comparison["common_scene_count"] == 2
    assert comparison["paired_valid_scene_count"] == 2
    assert comparison["candidate_minus_baseline"]["mean"] == 0.0
    assert comparison["candidate_minus_baseline"]["ci95"] == [0.0, 0.0]


def test_paired_comparison_rejects_same_index_with_different_source(tmp_path):
    reference = tmp_path / "reference.json"
    _write_reference(reference)
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    _write_scene(baseline_root, 0)
    _write_scene(candidate_root, 0)
    attempt = candidate_root / "scenario_000" / "attempt_result.json"
    attempt.write_text(
        json.dumps({"scenario_index": 0, "scenario_source": "different.pkl"}),
        encoding="utf-8",
    )
    baseline = collect_group(
        baseline_root, reference, minimum_samples=1, bootstrap_iterations=10
    )
    candidate = collect_group(
        candidate_root, reference, minimum_samples=1, bootstrap_iterations=10
    )

    comparison = paired_comparison(baseline, candidate, bootstrap_iterations=10)

    assert comparison["common_scene_count"] == 1
    assert comparison["source_matched_scene_count"] == 0
    assert comparison["source_mismatch_scene_keys"] == ["0"]
    assert comparison["paired_valid_scene_count"] == 0
    assert comparison["candidate_minus_baseline"]["mean"] is None


def test_bootstrap_empty_measurement_remains_missing():
    result = bootstrap_mean_interval([], iterations=10)
    assert result == {"count": 0, "mean": None, "ci95": [None, None]}
