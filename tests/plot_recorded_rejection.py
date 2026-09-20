"""把真实拒绝审计中的稀疏锚点和连续预测画在同一坐标系。"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(batch: Path, output: Path) -> None:
    root = batch/'smoke/single/movies/scenario_000'
    records = [json.loads(line) for line in (root/'execution_trace.jsonl').read_text().splitlines()]
    rejected = next(item for item in records if item.get('kind') == 'attack_rejected')
    state = next(item for item in records if item.get('kind') == 'state' and item['step'] == rejected['step'])
    road_file = next((batch/'smoke/single/carla/initial_data').glob('*.json'))
    lanes = json.loads(road_file.read_text())['road_network']
    fig, (ax, detail) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1.7, 1]})
    for lane in lanes:
        lane = np.asarray(lane)
        ax.plot(lane[:, 0], lane[:, 1], color='#b9c3ce', linewidth=.7)
    ego = np.asarray(state['ego'])
    ax.scatter(ego[0], ego[1], color='#24a174', s=90, label='Ego at rejection')
    agents = np.asarray(state['agents'])
    active = np.flatnonzero(state['active'])
    for agent_id in active:
        ax.scatter(*agents[agent_id, :2], color='#ed9945', s=55)
        ax.annotate(f'BG {agent_id}', agents[agent_id, :2])
    intent = rejected['attack_intent']
    anchors = np.asarray(intent['anchors'])
    ax.plot(anchors[:, 0], anchors[:, 1], 's--', color='#a44dc4', label='LLM sparse anchors')
    for name, key, color, style in [('Raw Diffusion', 'original_joint', '#2375bc', '-'),
                                     ('Projected, rejected', 'rejected_joint', '#d34d54', '--')]:
        joint = rejected.get(key)
        if joint is None:
            continue
        for agent_id, positions in zip(joint['agent_ids'], joint['positions']):
            points = np.asarray(positions)
            ax.plot(points[:, 0], points[:, 1], style, color=color, linewidth=1.8,
                    label=name if agent_id == joint['agent_ids'][0] else None)
    all_points = np.vstack([ego[None, :2], agents[active, :2], anchors,
                            *[np.asarray(p) for p in rejected['original_joint']['positions']]])
    low, high = all_points.min(axis=0)-5, all_points.max(axis=0)+5
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel='World X (m)',
           ylabel='World Y (m)', title=f"Step {rejected['step']}: rejected before execution")
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    detail.axis('off')
    lines = [f"Reason: {rejected['reason']}", f"Target BG {intent['target_id']}, strategy {intent['strategy']}",
             'Planned trajectories were never executed.', '', 'Road corner distance / 1.8 m limit:']
    for metric in rejected['rejected_metrics']:
        road = metric['road']
        dyn = metric['dynamics']
        lines += [f"BG {metric['agent_id']}: {road['max_corner_distance_m']:.3f} m"
                  f" {'FAIL' if road['violated'] else 'pass'}",
                  f"  initial {road['initial_max_corner_distance_m']:.3f} m",
                  f"  max jerk {dyn['max_jerk']['value']:.3f} / 12 m/s³"]
    detail.text(.02, .98, '\n'.join(lines), va='top', fontsize=11, family='monospace')
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(json.dumps({'output': str(output), 'reason': rejected['reason'],
                      'raw_agents': len(rejected['original_joint']['agent_ids']),
                      'projected_agents': len(rejected['rejected_joint']['agent_ids'])}))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
