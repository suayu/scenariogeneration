"""构建可复现实验的冻结场景清单，不修改任何原始场景文件。"""
import argparse
import hashlib
import json
from pathlib import Path
import time

def sha256(path: Path) -> str:
    """流式计算哈希，避免将大型场景一次性读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def build_manifest(source_root: Path, count: int, seed: int, steps: int) -> dict:
    """按路径稳定排序选择场景，并绑定每个文件的内容哈希。"""
    return build_manifest_from_roots([source_root], count=count, seed=seed, steps=steps)

def build_manifest_from_roots(source_roots, count: int, seed: int, steps: int) -> dict:
    """按来源顺序和文件名选择互异场景，内容重复项只保留首次出现者。"""
    roots = [Path(root).resolve() for root in source_roots]
    if not roots:
        raise ValueError("至少需要一个场景来源目录")
    unique = []
    seen_hashes = set()
    duplicate_count = 0
    for root in roots:
        files = sorted(path.resolve() for path in root.glob("*.pkl") if path.is_file())
        for path in files:
            digest = sha256(path)
            if digest in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(digest)
            unique.append((path, digest))
    if len(unique) < count:
        raise ValueError(f"互异可用场景不足：需要至少 {count} 个，实际仅 {len(unique)} 个")
    selected = unique[:count]
    single_source = len(roots) == 1
    return {
        "name": f"riskweaver_frozen_{count}_scenes",
        "created_at_unix": time.time(),
        "selection_rule": (
            "lexicographic_path_order_first_n" if single_source
            else "source_order_then_lexicographic_path_first_n_unique_sha256"
        ),
        "scene_root": str(roots[0]) if single_source else None,
        "scene_roots": [str(root) for root in roots],
        "source_unique_count": len(unique),
        "source_duplicate_count": duplicate_count,
        "scenario_count": count,
        "scenario_files": [str(path) for path, _ in selected],
        "scenario_sha256": [digest for _, digest in selected],
        "seed": seed,
        "simulation_steps": steps,
        "statistical_unit": "frozen_unique_scenario",
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="构建 RiskWeaver 冻结场景清单")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument(
        "--additional-source-root",
        action="append",
        default=[],
        type=Path,
        help="可重复指定；排在主来源之后，并按 SHA-256 去除重复场景",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"为保护既有实验，输出清单已存在：{args.output}")
    manifest = build_manifest_from_roots(
        [args.source_root, *args.additional_source_root],
        args.count,
        args.seed,
        args.steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "scenario_count": args.count}, ensure_ascii=False))

if __name__ == "__main__": main()
