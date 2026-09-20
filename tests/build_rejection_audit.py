"""只读生成逐次拒绝图册；缺失轨迹明确标注，不伪造执行结果。"""
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
OUT = ROOT/'experiments/rejection_audit_20260917_r1'
OUT.mkdir(exist_ok=False)
(OUT/'figures').mkdir()
(OUT/'videos').mkdir()
LOCAL = 'C:/Users/赵宇轩/Documents/Codex/2026-09-15/riskweaver-api-github-riskweaver-scenario-dreamer/outputs/rejection_audit_20260917_r1'
PREFIXES = ('riskweaver_multiagent_closed_', 'riskweaver_projection_smoke_', 'riskweaver_joint_diagnostic_')


def vehicle(ax, state, color, label):
    x, y, _, _, yaw, length, width = state[:7]
    corners = np.array([[1,1],[1,-1],[-1,-1],[-1,1]])*np.array([length, width])/2
    rotation = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    ax.add_patch(Polygon(corners@rotation.T+[x,y], facecolor=color, edgecolor='black', alpha=.6))
    ax.text(x+1, y+1, label, fontsize=8)


index = []
videos = []
for experiment in sorted((ROOT/'experiments').iterdir()):
    if not experiment.is_dir() or not experiment.name.startswith(PREFIXES):
        continue
    for path in sorted(experiment.glob('smoke/*/movies/scenario_*/attempt_result.json')):
        result = json.loads(path.read_text())
        records = [json.loads(line) for line in path.with_name('execution_trace.jsonl').read_text().splitlines()]
        outputs = [r for r in records if r.get('kind') == 'llm_output']
        arm = path.parents[2].name
        video = Path(result['video_path'])
        video_name = experiment.name+'__'+arm+'.mp4'
        if video.is_file():
            shutil.copy2(video, OUT/'videos'/video_name)
            videos.append(video_name)
        initial_paths = list((path.parents[2]/'carla/initial_data').glob('*.json'))
        lanes = json.loads(initial_paths[0].read_text())['road_network'] if initial_paths else []
        for output in outputs:
            step = output['step']
            plan = output.get('validated_plan') or {}
            rejected = not plan.get('attack') and not plan.get('obstacle_plan')
            late_failure = bool(result.get('generation_failure')) and output is outputs[-1]
            if not rejected and not late_failure:
                continue
            reason = result['generation_failure'] if late_failure else plan.get('reason', 'plan_validation_or_service_failure')
            states = [r for r in records if r.get('kind') == 'state' and r['step'] <= step]
            if not states:
                continue
            state = states[-1]
            context_rows = [r['context'] for r in records if r.get('kind') == 'profile_ranking_input' and r['step'] == step]
            context = context_rows[-1] if context_rows else {}
            diagnostics = dict(candidate_rejections=context.get('candidate_rejections'),
                               candidate_count=len(context.get('candidates', [])),
                               planner_trace=output.get('planner_trace'))
            name = experiment.name+'__'+arm+'__step'+str(step)
            fig = plt.figure(figsize=(13, 8), constrained_layout=True)
            grid = fig.add_gridspec(3, 2, width_ratios=(1.5, 1))
            ax = fig.add_subplot(grid[:, 0])
            for line in lanes:
                line = np.asarray(line)
                ax.plot(line[:,0], line[:,1], color='#b9c3ce', linewidth=.65, zorder=0)
            ego = np.asarray(state['ego'])
            vehicle(ax, ego, '#24a174', 'Ego')
            active = np.flatnonzero(state['active'])
            agents = np.asarray(state['agents'])
            target = plan.get('attack_target_id')
            for agent_id in active:
                vehicle(ax, agents[agent_id], '#ed9945' if agent_id == target else '#7292b2', f'BG {agent_id}')
            anchors = np.asarray(plan.get('anchors', []))
            if anchors.size:
                ax.plot(anchors[:,0], anchors[:,1], 's--', color='#a44dc4', label='LLM sparse anchors (unexecuted)')
                for i, point in enumerate(anchors):
                    ax.annotate(f'{i}s', point, fontsize=8)
            packs = list((experiment/'joint_samples').glob(f'joint_{step}_*.npz'))
            evidence = 'Diffusion arrays not saved at this rejection.'
            raw = None
            if packs:
                raw = np.load(packs[0])
                chosen = int(raw['selected_index'])
                for row, agent_id in enumerate(raw['agent_ids']):
                    if agent_id == target:
                        for trajectory in raw['positions'][1:, row]:
                            ax.plot(trajectory[:,0], trajectory[:,1], color='#7ac9e9', alpha=.45, linewidth=1)
                    selected = raw['positions'][chosen,row]
                    ax.plot(selected[:,0], selected[:,1], color='#d34d54', linewidth=1.5,
                            label='Saved selected diffusion' if row == 0 else None)
                evidence = f'Actual saved diffusion samples; selected index {chosen}. Not executed.'
                p = np.asarray(raw['positions'][chosen], float)
                s = np.asarray(raw['states'][raw['agent_ids']], float)
                dt = float(raw['dt'])
                v = np.diff(np.concatenate((s[:,None,:2], p), axis=1), axis=1)/dt
                a = np.diff(np.concatenate((s[:,None,2:4], v), axis=1), axis=1)/dt
                j = np.diff(a, axis=1)/dt
                diagnostics['selected_sample'] = chosen
                diagnostics['dynamics'] = {}
                for slot, (key, values, limit) in enumerate((('Speed (m/s)',v,20),('Acceleration (m/s2)',a,6),('Jerk (m/s3)',j,12))):
                    panel = fig.add_subplot(grid[slot,1])
                    norms = np.linalg.norm(values, axis=-1)
                    for row, agent_id in enumerate(raw['agent_ids']):
                        panel.plot((np.arange(norms.shape[1])+1)*dt, norms[row], label=f'BG {agent_id}')
                    panel.axhline(limit, color='red', linestyle='--', label=f'limit {limit}')
                    worst = np.unravel_index(np.argmax(norms), norms.shape)
                    diagnostics['dynamics'][key] = dict(value=float(norms[worst]), limit=limit,
                        agent_id=int(raw['agent_ids'][worst[0]]), future_index=int(worst[1]))
                    panel.set_title(key, fontsize=10)
                    panel.legend(fontsize=7, loc='upper right')
                    panel.grid(alpha=.2)
            else:
                predictions = [r for r in records if r.get('kind') == 'prediction' and r['step'] == step]
                if plan.get('attack') and predictions:
                    for i, positions in enumerate(predictions[0]['joint']['positions']):
                        p = np.asarray(positions)
                        ax.plot(p[:,0],p[:,1],color='#d34d54',label='Saved diffusion prediction' if i==0 else None)
                    evidence = 'Recorded joint prediction at the same step; inspect execution audit separately.'
                if not plan.get('attack'):
                    evidence = 'No accepted LLM attack; no attack-guided diffusion future available.'
                panel = fig.add_subplot(grid[:,1])
                panel.axis('off')
                details = f'Rejection: {reason}\n\nCandidates: {diagnostics["candidate_count"]}\n\nRecorded candidate rejection counts:\n'+json.dumps(diagnostics['candidate_rejections'],indent=2)+'\n\n'+evidence+'\n\nNo future path is invented.'
                panel.text(0, .98, details, va='top', fontsize=10, wrap=True)
            positions = np.vstack((ego[:2], agents[active,:2], anchors.reshape(-1,2) if anchors.size else ego[None,:2]))
            low, high = positions.min(axis=0)-8, positions.max(axis=0)+12
            ax.set_xlim(low[0],high[0]); ax.set_ylim(low[1],high[1])
            ax.set_aspect('equal'); ax.set_xlabel('World X (m)'); ax.set_ylabel('World Y (m)')
            ax.set_title(f'{arm}, step {step}\n{reason}', fontsize=11)
            handles, labels = ax.get_legend_handles_labels()
            if handles: ax.legend(fontsize=8)
            fig.suptitle(experiment.name, fontsize=10)
            fig.savefig(OUT/'figures'/(name+'.png'), dpi=140)
            plt.close(fig)
            index.append(dict(experiment=experiment.name, mode=arm, step=step, reason=reason,
                              evidence=evidence, diagnostics=diagnostics, figure=name+'.png', video=video_name))

(OUT/'rejections.json').write_text(json.dumps(index,ensure_ascii=False,indent=2,allow_nan=False))
lines = ['# 被拒绝攻击逐次图册', '', '范围：本次多 Agent 两批闭环、动力学诊断及投影独立冒烟。每个被拒绝或未形成攻击的规划点均列出。原始未来轨迹缺失时明确标记；没有把未执行轨迹称为攻击结果。', '', '|批次 / 组别|帧|理由|图与指标|', '|---|---:|---|---|']
for row in index:
    lines.append(f'|{row["experiment"]} / {row["mode"]}|{row["step"]}|{row["reason"]}|[查看](<{LOCAL}/figures/{row["figure"]}>)|')
lines += ['', '## 所有尝试的完整视频', '']
for video in videos:
    lines.append(f'- [{video}](<{LOCAL}/videos/{video}>)')
lines += ['', f'共 {len(index)} 个拒绝或无攻击规划点，{len(videos)} 个完整尝试视频。具体已记录指标见 [rejections.json](<{LOCAL}/rejections.json>)。']
(OUT/'index.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(dict(directory=str(OUT),rejections=len(index),videos=len(videos))))
