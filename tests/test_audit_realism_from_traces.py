import json

import numpy as np

from scripts.audit_realism_from_traces import audit_group


def _write_reference(path):
    edges = {
        "velocity": np.linspace(0.0, 30.0, 21).tolist(),
        "lon_accel": np.linspace(0.0, 10.0, 21).tolist(),
        "lat_accel": np.linspace(0.0, 10.0, 21).tolist(),
        "jerk": np.linspace(0.0, 20.0, 21).tolist(),
    }
    payload = {
        "dt_seconds": 0.1,
        "ticks": edges,
        "stats": {name: [1.0] * 20 for name in edges},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _state(step, xs, active):
    agents = [[x, float(agent_id), 0.0, 0.0, 0.25] for agent_id, x in enumerate(xs)]
    return {"kind": "state", "step": step, "agents": agents, "active": active}


def test_audit_group_aggregates_traces_without_joining_agent_ids(tmp_path):
    reference = tmp_path / "reference.json"
    _write_reference(reference)
    for trace_index in range(2):
        trace_dir = tmp_path / f"scenario_{trace_index:03d}"
        trace_dir.mkdir()
        rows = [_state(step, [step * step * 0.01 + trace_index, 99.0], [True, False]) for step in range(7)]
        (trace_dir / "execution_trace.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    result = audit_group(tmp_path, reference, minimum_samples=1)

    assert result["trace_count"] == 2
    assert result["state_row_count"] == 14
    assert result["observed_agent_state_count"] == 14
    assert result["sample_counts"]["velocity"]["simulated"] == 12
    assert result["sample_counts"]["lon_accel"]["simulated"] == 10
    assert result["sample_counts"]["jerk"]["simulated"] == 8
    assert result["valid"] is True


def test_audit_group_breaks_trajectory_when_step_is_not_consecutive(tmp_path):
    reference = tmp_path / "reference.json"
    _write_reference(reference)
    trace = tmp_path / "execution_trace.jsonl"
    rows = [_state(step, [float(step)], [True]) for step in (0, 1, 3, 4)]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    result = audit_group(trace, reference, minimum_samples=1)

    assert result["sample_counts"]["velocity"]["simulated"] == 2
    assert result["sample_counts"]["lon_accel"]["simulated"] == 0
    assert result["valid"] is False
    assert result["reason"] == "insufficient_realism_samples"


def test_audit_group_reports_empty_input_instead_of_fabricating_values(tmp_path):
    reference = tmp_path / "reference.json"
    _write_reference(reference)

    result = audit_group(tmp_path / "missing", reference, minimum_samples=1)

    assert result["trace_count"] == 0
    assert result["state_row_count"] == 0
    assert result["realism_deviation"] is None
    assert result["valid"] is False
