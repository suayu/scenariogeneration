import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "launch_baseline_formal_120_multigpu_20260927.py"
SPEC = importlib.util.spec_from_file_location("baseline_multigpu", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_split_manifest_is_complete_disjoint_and_contiguous(tmp_path):
    """多 GPU 分片必须完整覆盖原清单，且不重复或按结果选场景。"""
    manifest = {
        "name": "frozen_test",
        "scenario_count": 5,
        "scenario_files": [f"scene_{index}.pkl" for index in range(5)],
        "scenario_sha256": [f"digest_{index}" for index in range(5)],
    }

    shards = MODULE.split_manifest(manifest, shard_count=2, output_dir=tmp_path)

    assert [indices for _, indices in shards] == [[0, 1, 2], [3, 4]]
    assert sorted(index for _, indices in shards for index in indices) == list(range(5))
    first = json.loads(shards[0][0].read_text(encoding="utf-8"))
    second = json.loads(shards[1][0].read_text(encoding="utf-8"))
    assert first["scenario_files"] == ["scene_0.pkl", "scene_1.pkl", "scene_2.pkl"]
    assert second["scenario_files"] == ["scene_3.pkl", "scene_4.pkl"]
    assert first["shard_rule"] == "contiguous_numpy_array_split_no_outcome_selection"


def test_array_split_keeps_empty_tail_shards_deterministic():
    """极小验证清单的空尾分片也必须保持确定性。"""
    assert MODULE.np_array_split(range(2), 4) == [[0], [1], [], []]
