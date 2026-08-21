"""为大模型多模态推理生成简洁的当前帧鸟瞰图。"""

from io import BytesIO
import math

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np


def _lane_offsets(points, half_width):
    """根据中心线切向量计算左右可行驶区域边界。"""
    if len(points) < 2:
        return None, None
    tangent = np.gradient(points, axis=0)
    norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    tangent = tangent / np.maximum(norm, 1e-6)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
    return points + normal * half_width, points - normal * half_width


def _draw_vehicle(ax, state, color, label, zorder):
    """绘制带朝向箭头的交通参与者包围框。"""
    x, y, heading = float(state[0]), float(state[1]), float(state[4])
    length = max(float(state[5]) * 0.8, 1.0)
    width = max(float(state[6]) * 0.8, 0.6)
    rectangle = mpatches.FancyBboxPatch(
        (x - width / 2, y - length / 2),
        width,
        length,
        ec="black",
        fc=color,
        linewidth=0.8,
        boxstyle=mpatches.BoxStyle("Round", pad=0.15),
        zorder=zorder,
    )
    rectangle.set_transform(
        transforms.Affine2D().rotate_deg_around(x, y, math.degrees(heading) - 90)
        + ax.transData
    )
    ax.add_patch(rectangle)

    arrow_length = length / 2 + 1.5
    ax.annotate(
        "",
        xy=(x + arrow_length * math.cos(heading), y + arrow_length * math.sin(heading)),
        xytext=(x, y),
        arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.2},
        zorder=zorder + 1,
    )
    ax.text(
        x,
        y,
        str(label),
        color="black",
        fontsize=8,
        ha="center",
        va="center",
        zorder=zorder + 2,
    )


def render_llm_scene_png(
    ego_state,
    agent_states,
    agent_ids,
    lanes,
    lanes_mask,
    view_radius=40.0,
    drivable_half_width=2.0,
):
    """返回仅包含当前空间关系和道路信息的 PNG 字节。"""
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
    ax.set_xlim(-view_radius, view_radius)
    ax.set_ylim(-view_radius, view_radius)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_facecolor("white")

    for lane, lane_mask in zip(lanes, lanes_mask):
        points = np.asarray(lane, dtype=np.float32)[np.asarray(lane_mask, dtype=bool), :2]
        if len(points) < 2:
            continue
        left, right = _lane_offsets(points, drivable_half_width)
        polygon = np.concatenate([left, right[::-1]], axis=0)
        ax.fill(polygon[:, 0], polygon[:, 1], color="#e6e6e6", alpha=0.8, zorder=1)
        ax.plot(left[:, 0], left[:, 1], color="#6f6f6f", linewidth=1.0, zorder=2)
        ax.plot(right[:, 0], right[:, 1], color="#6f6f6f", linewidth=1.0, zorder=2)
        ax.plot(
            points[:, 0],
            points[:, 1],
            color="#8a8a8a",
            linewidth=0.9,
            linestyle="--",
            zorder=3,
        )

    for state, agent_id in zip(agent_states, agent_ids):
        if state[-1] != 0:
            _draw_vehicle(ax, state, "#87b3e6", int(agent_id), zorder=5)
    _draw_vehicle(ax, ego_state, "#de5959", "EGO", zorder=7)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return buffer.getvalue()
