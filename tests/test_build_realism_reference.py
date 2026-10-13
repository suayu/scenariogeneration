import io
import pickle
import tarfile

import numpy as np

from scripts.build_realism_reference import build_reference, build_reference_archive


def test_build_reference_excludes_ego_and_records_manifest(tmp_path):
    frames = 30
    states = np.zeros((3, frames, 8), dtype=float)
    states[..., 7] = 1.0
    time = np.arange(frames) * 0.1
    for agent_id in range(3):
        states[agent_id, :, 0] = (2.0 + agent_id) * time + 0.2 * (agent_id + 1) * time ** 2
        states[agent_id, :, 1] = agent_id * 3.0 + 0.1 * np.sin(time)
        states[agent_id, :, 4] = 0.02 * np.sin(time)
    path = tmp_path / "real.pkl"
    with path.open("wb") as handle:
        pickle.dump({"agents": states, "ego_index": 2}, handle)

    result = build_reference([path], 0.1, "synthetic-real")

    assert result["file_count"] == 1
    assert result["dataset_label"] == "synthetic-real"
    assert len(result["source_manifest_sha256"]) == 64
    assert result["sample_counts"]["velocity"] == 2 * (frames - 1)
    assert sum(result["stats"]["jerk"]) > 0


def test_build_reference_reads_only_waymo_log_replay_from_archive(tmp_path):
    frames = 30
    states = np.zeros((3, frames, 8), dtype=float)
    states[..., 7] = 1.0
    time = np.arange(frames) * 0.1
    for agent_id in range(3):
        states[agent_id, :, 0] = (2.0 + agent_id) * time + 0.1 * time ** 2
        states[agent_id, :, 1] = agent_id * 3.0
    archive_path = tmp_path / "simulation_environment_datasets.tar"
    with tarfile.open(archive_path, "w") as archive:
        for name, payload in (
            ("bundle/waymo_sim_test_pickles/real.pkl", {"agents": states, "ego_index": 2}),
            ("bundle/scenario_dreamer_waymo_100m_pickles/initial.pkl", {"agents": states[:, :1]}),
        ):
            raw = pickle.dumps(payload)
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))

    result = build_reference_archive(archive_path, 0.1, "official-archive-smoke")

    assert result["file_count"] == 1
    assert result["sources"][0]["path"].endswith("waymo_sim_test_pickles/real.pkl")
    assert result["sample_counts"]["velocity"] == 2 * (frames - 1)
