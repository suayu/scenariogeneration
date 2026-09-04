import pickle
import numpy as np
from experiments.protocol import build_manifest, summarize_episodes


def test_manifest_is_mutually_exclusive_and_repeatable(tmp_path):
    for index in range(10):
        with (tmp_path / f"scene_{index}.pkl").open("wb") as handle:
            pickle.dump({"scenario_id": f"scene-{index}", "agents": np.zeros((2, 1, 5)), "agent_types": np.array([[0, 1], [0, 1]])}, handle)
    assert build_manifest(tmp_path, 7) == build_manifest(tmp_path, 7)
    assert {row["split"] for row in build_manifest(tmp_path, 7)} == {"development", "validation", "test"}


def test_summary_groups_by_split():
    rows = []
    for split, idm, rl in (("development", 0, 0), ("validation", 0, 1), ("test", 1, 0)):
        rows.extend([{"method": "full", "policy": "idm", "split": split, "scenario_id": f"{split}-idm", "seed": 1, "collision": idm}, {"method": "full", "policy": "rl", "split": split, "scenario_id": f"{split}-rl", "seed": 1, "collision": rl}])
    assert len(summarize_episodes(rows, 3, 100)) == 6
    assert {item["split"] for item in summarize_episodes(rows, 3, 100)} == {"development", "validation", "test"}
