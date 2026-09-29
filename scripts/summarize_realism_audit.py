"""汇总真实性审计并生成中英文论文图。"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = ("velocity", "lon_accel", "lat_accel", "jerk")
LABELS = {
    "zh": {"velocity": "速度", "lon_accel": "纵向加速度", "lat_accel": "横向加速度", "jerk": "加加速度"},
    "en": {"velocity": "Speed", "lon_accel": "Longitudinal acceleration", "lat_accel": "Lateral acceleration", "jerk": "Jerk"},
}


def _load_group(path):
    rows = []
    for result_path in sorted(path.rglob("attempt_result.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        audit = payload.get("realism_audit")
        if audit:
            rows.append((result_path, audit))
    return rows


def summarize(groups):
    summaries = {}
    for label, root in groups:
        rows = _load_group(root)
        valid = [audit for _, audit in rows if audit.get("valid")]
        summary = {
            "group": label,
            "attempt_count": len(rows),
            "valid_count": len(valid),
            "valid_rate": len(valid) / len(rows) if rows else 0.0,
            "realism_deviation_mean": float(np.mean([x["realism_deviation"] for x in valid])) if valid else None,
            "wasserstein": {},
            "simulated_counts": {},
            "reference_probability": {},
            "histogram_edges": {},
        }
        for metric in METRICS:
            distances = [x["wasserstein"][metric] for x in valid if x["wasserstein"].get(metric) is not None]
            summary["wasserstein"][metric] = float(np.mean(distances)) if distances else None
            counts = [np.asarray(x["histograms"]["simulated_counts"][metric], dtype=float) for x in valid]
            if counts:
                summary["simulated_counts"][metric] = np.sum(counts, axis=0).tolist()
                summary["reference_probability"][metric] = valid[0]["histograms"]["reference_probability"][metric]
                summary["histogram_edges"][metric] = valid[0]["histogram_edges"][metric]
        summaries[label] = summary
    return summaries


def _configure_chinese_font():
    candidates = [
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for path in candidates:
        if path.exists():
            from matplotlib import font_manager
            font_manager.fontManager.addfont(str(path))
            # 中文字形与拉丁字符分别回退，避免单一 CJK 字体缺少英文和数字。
            plt.rcParams["font.family"] = [
                font_manager.FontProperties(fname=str(path)).get_name(), "DejaVu Sans"
            ]
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot(summaries, output, language):
    if language == "zh":
        _configure_chinese_font()
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for axis, metric in zip(axes.flat, METRICS):
        reference_drawn = False
        for label, summary in summaries.items():
            if metric not in summary["simulated_counts"]:
                continue
            edges = np.asarray(summary["histogram_edges"][metric])
            centers = 0.5 * (edges[:-1] + edges[1:])
            counts = np.asarray(summary["simulated_counts"][metric])
            probability = counts / counts.sum()
            axis.step(centers, probability, where="mid", linewidth=2, label=label)
            if not reference_drawn:
                reference = np.asarray(summary["reference_probability"][metric])
                axis.step(centers, reference, where="mid", linewidth=2.5, linestyle="--", color="black",
                          label="真实数据" if language == "zh" else "Real data")
                reference_drawn = True
        axis.set_title(LABELS[language][metric])
        axis.set_ylabel("归一化频率" if language == "zh" else "Normalized frequency")
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.subplots_adjust(top=0.84, hspace=0.34, wspace=0.18)
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=max(1, len(labels)))
    figure.suptitle(
        "生成轨迹与真实轨迹分布" if language == "zh" else "Generated versus real trajectory distributions",
        y=0.985,
    )
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", action="append", required=True, help="格式：显示名称=实验目录")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    groups = []
    for item in args.group:
        label, path = item.split("=", 1)
        groups.append((label, Path(path)))
    summaries = summarize(groups)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "realism_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "realism_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "attempt_count", "valid_count", "valid_rate", "realism_deviation_mean"])
        writer.writeheader()
        for summary in summaries.values():
            writer.writerow({key: summary[key] for key in writer.fieldnames})
    plot(summaries, args.output_dir / "真实性分布对比.png", "zh")
    plot(summaries, args.output_dir / "realism_distribution_comparison.png", "en")


if __name__ == "__main__":
    main()
