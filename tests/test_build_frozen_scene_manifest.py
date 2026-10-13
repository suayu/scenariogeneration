import hashlib
import json
import pytest
from scripts.build_frozen_scene_manifest import build_manifest, build_manifest_from_roots

def test_build_manifest_is_stable_and_hashes_files(tmp_path):
    (tmp_path / "b.pkl").write_bytes(b"second")
    (tmp_path / "a.pkl").write_bytes(b"first")
    manifest = build_manifest(tmp_path, count=2, seed=42, steps=400)
    assert [path.rsplit("/", 1)[-1] for path in manifest["scenario_files"]] == ["a.pkl", "b.pkl"]
    assert manifest["scenario_sha256"][0] == hashlib.sha256(b"first").hexdigest()
    assert manifest["scenario_count"] == 2
    assert json.loads(json.dumps(manifest))["seed"] == 42

def test_build_manifest_rejects_insufficient_source(tmp_path):
    (tmp_path / "one.pkl").write_bytes(b"one")
    with pytest.raises(ValueError, match="互异可用场景不足"):
        build_manifest(tmp_path, count=2, seed=42, steps=400)

def test_build_manifest_from_roots_preserves_source_order_and_deduplicates(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "b.pkl").write_bytes(b"existing-b")
    (first / "a.pkl").write_bytes(b"existing-a")
    (second / "a-duplicate.pkl").write_bytes(b"existing-a")
    (second / "c.pkl").write_bytes(b"generated-c")

    manifest = build_manifest_from_roots([first, second], count=3, seed=7, steps=400)

    assert [path.rsplit("/", 1)[-1] for path in manifest["scenario_files"]] == [
        "a.pkl",
        "b.pkl",
        "c.pkl",
    ]
    assert manifest["source_duplicate_count"] == 1
    assert manifest["source_unique_count"] == 3
    assert manifest["scene_root"] is None
    assert manifest["scene_roots"] == [str(first.resolve()), str(second.resolve())]
    assert manifest["selection_rule"] == "source_order_then_lexicographic_path_first_n_unique_sha256"

def test_build_manifest_from_roots_rejects_duplicates_as_extra_capacity(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.pkl").write_bytes(b"same")
    (second / "b.pkl").write_bytes(b"same")

    with pytest.raises(ValueError, match="实际仅 1 个"):
        build_manifest_from_roots([first, second], count=2, seed=42, steps=400)
