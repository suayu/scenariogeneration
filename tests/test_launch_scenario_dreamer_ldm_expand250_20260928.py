"""250 场 LDM 扩充启动器的静态协议测试。"""

import importlib.util
from pathlib import Path


MODULE_NAME = "launch_scenario_dreamer_ldm_expand250_20260928.py"
MODULE_PATH = next(
    path
    for path in (Path(__file__).parent.parent / MODULE_NAME, Path(__file__).with_name(MODULE_NAME))
    if path.is_file()
)
SPEC = importlib.util.spec_from_file_location("launcher", MODULE_PATH)
LAUNCHER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(LAUNCHER)


def test_generation_uses_new_seed_and_adequate_budget():
    generation, _, _ = LAUNCHER.commands()

    assert "ldm.eval.seed=43" in generation
    assert "ldm.eval.num_samples=140" in generation
    assert LAUNCHER.MINIMUM_NEW_VALID == 120
    assert LAUNCHER.EXPECTED_FROZEN == 250


def test_manifest_preserves_source_order_and_freezes_250():
    _, _, manifest = LAUNCHER.commands()
    roots = [
        manifest[index + 1]
        for index, value in enumerate(manifest)
        if value in {"--source-root", "--additional-source-root"}
    ]

    assert roots == [
        str(LAUNCHER.EXISTING_SCENES),
        str(LAUNCHER.PRIOR_GENERATED_SCENES),
        str(LAUNCHER.POSTPROCESSED),
    ]
    assert manifest[manifest.index("--count") + 1] == "250"


def test_selected_source_counts_rejects_unexpected_parent():
    files = [
        LAUNCHER.EXISTING_SCENES / "old.pkl",
        LAUNCHER.PRIOR_GENERATED_SCENES / "prior.pkl",
        LAUNCHER.POSTPROCESSED / "new.pkl",
        Path("/tmp/unexpected.pkl"),
    ]

    assert LAUNCHER.selected_source_counts(files) == {
        "existing": 1,
        "prior_generated": 1,
        "new_generated": 1,
        "unexpected": 1,
    }
