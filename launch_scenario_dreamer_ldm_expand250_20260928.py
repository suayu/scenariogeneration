"""继续生成 Scenario Dreamer 环境，并冻结 250 场统一评测清单。"""

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path("/home2/zhaoyx/scenario-dreamer")
PYTHON = Path("/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python")
CANONICAL_CHECKPOINT = ROOT / "checkpoints/scenario_dreamer_ldm_large_waymo/last.ckpt"
EXPECTED_SHA256 = "06a1a65e9949f55c3398aeadacde388b03a6705f2661bc273cf43e7319de4cd5"
PRIOR_RESULT = ROOT / "experiments/scenario_dreamer_ldm_full60_20260927_r2/stage_result.json"
PRIOR_GENERATED_SCENES = (
    ROOT
    / "experiments/scenario_dreamer_ldm_full60_20260927_r2/postprocessed_200m_pickles"
)
RUN_NAME = "scenario_dreamer_ldm_large_waymo_riskweaver_expand140_20260928_r1"
RUN_CHECKPOINT_DIR = ROOT / "checkpoints" / RUN_NAME
OUT = ROOT / "experiments/scenario_dreamer_ldm_expand250_20260928_r1"
POSTPROCESSED = OUT / "postprocessed_200m_pickles"
EXISTING_SCENES = ROOT / "metadata/simulation_environment_datasets/scenario_dreamer_waymo_200m_pickles"
FROZEN_MANIFEST = OUT / "riskweaver_frozen_250_scenes.json"
EXPECTED_GENERATED = 140
MINIMUM_NEW_VALID = 120
EXPECTED_FROZEN = 250


def write_json(path, value):
    """保存严格 JSON，便于监控器和后续正式实验读取。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def sha256(path):
    """流式校验大权重文件。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def commands():
    generation = [
        str(PYTHON),
        "eval.py",
        # 使用短且固定的 Hydra 输出路径，避免全部覆盖参数组成超长目录名。
        f"hydra.run.dir={OUT / 'hydra_simulation_environments'}",
        "dataset_name=waymo",
        "model_name=ldm",
        "ldm.model.num_l2l_blocks=3",
        f"ldm.eval.run_name={RUN_NAME}",
        "ldm.model.autoencoder_run_name=scenario_dreamer_autoencoder_waymo",
        "ldm.eval.seed=43",
        "ldm.eval.mode=simulation_environments",
        f"ldm.eval.num_samples={EXPECTED_GENERATED}",
        "ldm.eval.batch_size=8",
        "ldm.eval.visualize=false",
        "ldm.eval.sim_envs.route_length=200",
        "ldm.eval.sim_envs.overhead_factor=3",
        "ldm.eval.sim_envs.num_inpainting_candidates=8",
        "ldm.eval.sim_envs.nocturne_compatible_only=true",
    ]
    pre_path = RUN_CHECKPOINT_DIR / "complete_sim_envs"
    postprocess = [
        str(PYTHON),
        "data_processing/postprocess_simulation_environments.py",
        f"hydra.run.dir={OUT / 'hydra_postprocess_200m'}",
        "dataset_name=waymo",
        f"postprocess_sim_envs.run_name={RUN_NAME}",
        f"postprocess_sim_envs.pre_path={pre_path}",
        f"postprocess_sim_envs.post_path={POSTPROCESSED}",
        "postprocess_sim_envs.route_length=200",
        "postprocess_sim_envs.max_num_envs=-1",
    ]
    manifest = [
        str(PYTHON),
        "scripts/build_frozen_scene_manifest.py",
        "--source-root",
        str(EXISTING_SCENES),
        "--additional-source-root",
        str(PRIOR_GENERATED_SCENES),
        "--additional-source-root",
        str(POSTPROCESSED),
        "--output",
        str(FROZEN_MANIFEST),
        "--count",
        str(EXPECTED_FROZEN),
        "--seed",
        "42",
        "--steps",
        "400",
    ]
    return generation, postprocess, manifest


def run_stage(name, command, env, result):
    """串行执行阶段并保留终态，避免生成失败后继续冻结清单。"""
    log_path = OUT / f"{name}.log"
    started_at = time.time()
    with log_path.open("w", encoding="utf-8") as output:
        exit_code = subprocess.call(
            command,
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    stage = {
        "name": name,
        "command": [str(item) for item in command],
        "exit_code": exit_code,
        "started_at": started_at,
        "finished_at": time.time(),
        "log_path": str(log_path),
    }
    result["stages"].append(stage)
    write_json(OUT / "stage_result.partial.json", result)
    if exit_code != 0:
        raise RuntimeError(f"阶段 {name} 失败，退出码 {exit_code}")


def validate_scene(path):
    """验证闭环加载所需的最小字段和数组形状，不按危险度筛选场景。"""
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    required = {
        "agents",
        "agent_types",
        "lanes",
        "lane_graph",
        "route",
        "route_lane_indices",
        "num_agents",
        "num_lanes",
    }
    missing = sorted(required - set(payload))
    if missing:
        return False, f"missing_keys:{','.join(missing)}"
    agents = payload["agents"]
    lanes = payload["lanes"]
    route = payload["route"]
    if getattr(agents, "ndim", None) != 3 or agents.shape[-1] < 8:
        return False, f"invalid_agents_shape:{getattr(agents, 'shape', None)}"
    if getattr(lanes, "ndim", None) != 3 or lanes.shape[-1] != 2:
        return False, f"invalid_lanes_shape:{getattr(lanes, 'shape', None)}"
    if getattr(route, "ndim", None) != 2 or route.shape[-1] != 2 or len(route) < 2:
        return False, f"invalid_route_shape:{getattr(route, 'shape', None)}"
    return True, None


def selected_source_counts(scenario_files):
    """审计冻结清单中三个固定来源的实际入选数量。"""
    counts = {}
    for source in scenario_files:
        source_parent = Path(source).resolve().parent
        if source_parent == EXISTING_SCENES.resolve():
            key = "existing"
        elif source_parent == PRIOR_GENERATED_SCENES.resolve():
            key = "prior_generated"
        elif source_parent == POSTPROCESSED.resolve():
            key = "new_generated"
        else:
            key = "unexpected"
        counts[key] = counts.get(key, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只解析 Hydra 配置和清单工具参数，不创建输出或读取权重",
    )
    args = parser.parse_args()
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "2"),
        PROJECT_ROOT=str(ROOT),
        SCRATCH_ROOT=str(ROOT),
        DATASET_ROOT=str(ROOT),
        PYTHONPATH=f"{ROOT}:{ROOT / 'safe-sim'}:{ROOT / 'safe-sim/trajdata/src'}",
        MPLBACKEND="Agg",
        PYTHONUNBUFFERED="1",
        HYDRA_FULL_ERROR="1",
    )
    generation, postprocess, manifest = commands()
    if args.validate_only:
        for command in (generation, postprocess):
            subprocess.run(
                [*command, "--cfg", "job", "--resolve"],
                cwd=ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                check=True,
            )
        subprocess.run(
            [str(PYTHON), "scripts/build_frozen_scene_manifest.py", "--help"],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            check=True,
        )
        return

    # 扩充批次以已通过的 120 场冻结结果为前置，绝不改写上一批产物。
    if not PRIOR_RESULT.is_file():
        raise FileNotFoundError(f"缺少上一批 LDM 生成结果：{PRIOR_RESULT}")
    prior = json.loads(PRIOR_RESULT.read_text(encoding="utf-8"))
    if (
        not prior.get("accepted")
        or int(prior.get("frozen_scenario_count") or 0) != 120
        or not PRIOR_GENERATED_SCENES.is_dir()
    ):
        raise RuntimeError("上一批 120 场生成门禁未通过，拒绝扩充至 250 场")
    if not CANONICAL_CHECKPOINT.is_file():
        raise FileNotFoundError(f"缺少 LDM 权重：{CANONICAL_CHECKPOINT}")
    actual_sha256 = sha256(CANONICAL_CHECKPOINT)
    if actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError("LDM 权重 SHA-256 不匹配")
    if OUT.exists() or RUN_CHECKPOINT_DIR.exists():
        raise FileExistsError("正式生成输出或独立运行目录已存在，拒绝覆盖")

    OUT.mkdir(parents=True)
    RUN_CHECKPOINT_DIR.mkdir(parents=True)
    shutil.copy2(__file__, OUT / Path(__file__).name)
    (RUN_CHECKPOINT_DIR / "last.ckpt").symlink_to(CANONICAL_CHECKPOINT)
    result = {
        "run_name": RUN_NAME,
        "checkpoint_sha256": actual_sha256,
        "expected_generated": EXPECTED_GENERATED,
        "minimum_new_valid": MINIMUM_NEW_VALID,
        "expected_frozen": EXPECTED_FROZEN,
        "generation_seed": 43,
        "prior_generation_result": str(PRIOR_RESULT),
        "selection_rule": (
            "existing_then_prior_generated_then_new_generated_"
            "lexicographic_unique_sha256"
        ),
        "no_outcome_based_selection": True,
        "stages": [],
        "started_at": time.time(),
    }
    write_json(OUT / "launch.json", result)
    try:
        run_stage("simulation_environments", generation, env, result)
        complete_count = len(list((RUN_CHECKPOINT_DIR / "complete_sim_envs").glob("*.pkl")))
        if complete_count < MINIMUM_NEW_VALID:
            raise RuntimeError(
                f"完整环境不足：需要至少 {MINIMUM_NEW_VALID}，实际 {complete_count}"
            )

        run_stage("postprocess_200m", postprocess, env, result)
        valid_files = []
        invalid = {}
        for path in sorted(POSTPROCESSED.glob("*.pkl")):
            valid, reason = validate_scene(path)
            if valid:
                valid_files.append(path)
            else:
                invalid[path.name] = reason
        result.update(
            {
                "complete_sim_env_count": complete_count,
                "postprocessed_valid_count": len(valid_files),
                "postprocessed_invalid": invalid,
            }
        )
        write_json(OUT / "scene_validation.json", result)
        if len(valid_files) < MINIMUM_NEW_VALID:
            raise RuntimeError(
                f"有效后处理场景不足：需要至少 {MINIMUM_NEW_VALID}，实际 {len(valid_files)}"
            )

        run_stage("freeze_250_manifest", manifest, env, result)
        frozen = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
        if int(frozen.get("scenario_count") or 0) != EXPECTED_FROZEN:
            raise RuntimeError("冻结清单场景数不是 250")
        source_counts = selected_source_counts(frozen.get("scenario_files") or [])
        if source_counts.get("unexpected", 0):
            raise RuntimeError(f"冻结清单包含未知来源：{source_counts}")
        if source_counts.get("new_generated", 0) < MINIMUM_NEW_VALID:
            raise RuntimeError(f"冻结清单的新批次场景不足：{source_counts}")
        result.update(
            {
                "accepted": True,
                "frozen_manifest": str(FROZEN_MANIFEST),
                "frozen_manifest_sha256": sha256(FROZEN_MANIFEST),
                "frozen_scenario_count": EXPECTED_FROZEN,
                "source_duplicate_count": frozen.get("source_duplicate_count"),
                "selected_source_counts": source_counts,
            }
        )
    except Exception as exc:
        result.update({"accepted": False, "failure": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        result["finished_at"] = time.time()
        write_json(OUT / "stage_result.json", result)


if __name__ == "__main__":
    main()
