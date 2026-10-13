"""官方 Scenario Dreamer LDM 的分阶段生成冒烟。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path("/home2/zhaoyx/scenario-dreamer")
PYTHON = Path("/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python")
CANONICAL_CHECKPOINT = ROOT / "checkpoints/scenario_dreamer_ldm_large_waymo/last.ckpt"
EXPECTED_SHA256 = "06a1a65e9949f55c3398aeadacde388b03a6705f2661bc273cf43e7319de4cd5"
RUN_NAME = "scenario_dreamer_ldm_large_waymo_riskweaver_smoke_20260927_r1"
RUN_CHECKPOINT_DIR = ROOT / "checkpoints" / RUN_NAME
OUT = ROOT / "experiments/scenario_dreamer_ldm_generation_smoke_20260927_r1"
POSTPROCESSED = OUT / "postprocessed_200m_pickles"


def write_json(path, value):
    """保存严格 JSON，保证阶段结果可由后续门禁读取。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def sha256(path):
    """流式计算大权重哈希，避免把 13 GB 文件读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_stage(name, command, env, result):
    """逐阶段运行并落盘；失败立即终止，禁止跳过门禁继续生成。"""
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


def commands():
    """返回固定的小样本配置，供验证和实际执行共用。"""
    common = [
        str(PYTHON),
        "eval.py",
        "dataset_name=waymo",
        "model_name=ldm",
        "ldm.model.num_l2l_blocks=3",
        f"ldm.eval.run_name={RUN_NAME}",
        "ldm.model.autoencoder_run_name=scenario_dreamer_autoencoder_waymo",
        "ldm.eval.seed=42",
        "ldm.eval.batch_size=2",
    ]
    initial = [
        *common,
        "ldm.eval.mode=initial_scene",
        "ldm.eval.num_samples=4",
        "ldm.eval.cache_samples=true",
        "ldm.eval.visualize=true",
    ]
    simulation = [
        *common,
        "ldm.eval.mode=simulation_environments",
        "ldm.eval.num_samples=2",
        "ldm.eval.visualize=true",
        "ldm.eval.sim_envs.route_length=200",
        "ldm.eval.sim_envs.overhead_factor=3",
        "ldm.eval.sim_envs.num_inpainting_candidates=8",
        "ldm.eval.sim_envs.nocturne_compatible_only=true",
    ]
    pre_path = RUN_CHECKPOINT_DIR / "complete_sim_envs"
    postprocess = [
        str(PYTHON),
        "data_processing/postprocess_simulation_environments.py",
        "dataset_name=waymo",
        f"postprocess_sim_envs.run_name={RUN_NAME}",
        f"postprocess_sim_envs.pre_path={pre_path}",
        f"postprocess_sim_envs.post_path={POSTPROCESSED}",
        "postprocess_sim_envs.route_length=200",
        "postprocess_sim_envs.max_num_envs=2",
    ]
    return initial, simulation, postprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="只解析三阶段 Hydra 配置，不读取权重或创建实验目录",
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
    initial, simulation, postprocess = commands()
    if args.validate_only:
        for command in (initial, simulation, postprocess):
            subprocess.run(
                [*command, "--cfg", "job", "--resolve"],
                cwd=ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return

    # 权重与哈希在创建任何输出前验证，避免错误文件污染实验目录。
    if not CANONICAL_CHECKPOINT.is_file():
        raise FileNotFoundError(f"缺少已校验的 LDM 权重：{CANONICAL_CHECKPOINT}")
    actual_sha256 = sha256(CANONICAL_CHECKPOINT)
    if actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError(
            f"LDM 权重 SHA-256 不匹配：{actual_sha256} != {EXPECTED_SHA256}"
        )
    if OUT.exists() or RUN_CHECKPOINT_DIR.exists():
        raise FileExistsError("冒烟输出或独立运行目录已存在，拒绝覆盖")

    OUT.mkdir(parents=True)
    RUN_CHECKPOINT_DIR.mkdir(parents=True)
    shutil.copy2(__file__, OUT / Path(__file__).name)
    # 独立运行目录只链接已校验权重，生成缓存与官方权重目录相互隔离。
    (RUN_CHECKPOINT_DIR / "last.ckpt").symlink_to(CANONICAL_CHECKPOINT)
    result = {
        "run_name": RUN_NAME,
        "canonical_checkpoint": str(CANONICAL_CHECKPOINT),
        "checkpoint_sha256": actual_sha256,
        "expected_sha256": EXPECTED_SHA256,
        "stages": [],
        "started_at": time.time(),
    }
    write_json(OUT / "launch.json", result)
    try:
        run_stage("initial_scene", initial, env, result)
        initial_count = len(list((RUN_CHECKPOINT_DIR / "initial_scene_samples").glob("*.pkl")))
        if initial_count < 4:
            raise RuntimeError(f"initial_scene 缓存不足：{initial_count}/4")

        run_stage("simulation_environments", simulation, env, result)
        complete_count = len(list((RUN_CHECKPOINT_DIR / "complete_sim_envs").glob("*.pkl")))
        if complete_count < 1:
            raise RuntimeError("没有生成完整 simulation environment")

        run_stage("postprocess_200m", postprocess, env, result)
        postprocessed_files = sorted(POSTPROCESSED.glob("*.pkl"))
        if not postprocessed_files:
            raise RuntimeError("200 m 后处理没有产生有效场景")
        result.update(
            {
                "accepted": True,
                "initial_scene_count": initial_count,
                "complete_sim_env_count": complete_count,
                "postprocessed_count": len(postprocessed_files),
                "postprocessed_files": [str(path) for path in postprocessed_files],
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
