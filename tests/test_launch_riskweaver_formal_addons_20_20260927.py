import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "launch_riskweaver_formal_addons_20_20260927.py"
SPEC = importlib.util.spec_from_file_location("formal_addons", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_fixed_subset_spans_original_and_generated_portions():
    """等间隔 20 场必须覆盖 120 场全域且不重复。"""
    indices = MODULE.fixed_subset_indices()
    assert indices == list(range(0, 120, 6))
    assert len(indices) == len(set(indices)) == 20
    assert indices[0] == 0
    assert indices[-1] == 114
    assert sum(index >= 75 for index in indices) == 7


def test_subset_manifest_preserves_file_digest_alignment():
    """场景路径、哈希和原始索引必须保持一一对应。"""
    source = {
        "name": "source",
        "scenario_count": 120,
        "scenario_files": [f"scene_{index}.pkl" for index in range(120)],
        "scenario_sha256": [f"digest_{index}" for index in range(120)],
    }
    subset = MODULE.build_subset_manifest(source)
    indices = subset["original_scenario_indices"]
    assert subset["scenario_count"] == 20
    assert subset["scenario_files"] == [f"scene_{index}.pkl" for index in indices]
    assert subset["scenario_sha256"] == [f"digest_{index}" for index in indices]
    assert subset["selection_rule"] == "indices_0_to_114_stride_6_no_outcome_selection"


def test_groups_change_only_declared_ablation_dimensions():
    """五个补充组覆盖画像、锚点、静态障碍和 RL 完整方法。"""
    groups = {group["name"]: group for group in MODULE.GROUPS}
    assert set(groups) == {
        "idm_profile_off",
        "idm_no_anchor_drive",
        "idm_dynamic_only",
        "rl_profile_off",
        "rl_profile_on",
    }
    assert groups["idm_no_anchor_drive"]["anchor_drive_enabled"] is False
    assert groups["idm_dynamic_only"]["obstacles_enabled"] is False
    assert groups["rl_profile_off"]["profile_enabled"] is False
    assert groups["rl_profile_on"]["profile_enabled"] is True
