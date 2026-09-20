"""只读现有产物，生成周报的危险度、四模式与画像统计图。"""
import glob
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path('/home2/zhaoyx/scenario-dreamer')
out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)

# 目标危险度：缺测单列显示，绝不作为数值零点。
paths=sorted(glob.glob(str(root/'experiments/riskweaver_target_formal_20260913/movies/scenario_*/attempt_*/attempt_result.json')))
rows=[json.load(open(path)) for path in paths]
values=[r.get('scenario_danger_score') for r in rows]
valid=[(i,float(v)) for i,v in enumerate(values) if v is not None]
missing=[i for i,v in enumerate(values) if v is None]
fig,ax=plt.subplots(figsize=(11,4.5),constrained_layout=True)
ax.axhspan(.4,.6,color='#73b68c',alpha=.22,label='Target interval [0.4, 0.6]')
ax.scatter([i for i,_ in valid],[v for _,v in valid],c='#3e78a8',s=29,label=f'Valid D ({len(valid)})')
ax.scatter(missing,[-.045]*len(missing),marker='x',c='#d85a5a',s=48,label=f'Missing D ({len(missing)}), plotted separately')
ax.set(xlabel='Attempt (fixed manifest order)',ylabel='Frame-weighted scenario danger D',ylim=(-.1,.72),
       title=f'Target difficulty formal: {sum(.4<=v<=.6 for _,v in valid)}/{len(values)} target hits')
ax.legend(loc='upper left',fontsize=8); ax.grid(alpha=.15)
fig.savefig(out/'difficulty_target.png',dpi=160); plt.close(fig)

# 四组冒烟：第四组的缺测不能当作 D=0。
b=root/'experiments/riskweaver_four_arm_multibg_1789702359802287367/comparison.json'
first=json.loads(b.read_text())[:3]
mem=root/'experiments/riskweaver_memory_multibg_r2_1789721816271880209/smoke/parallel_memory/movies/scenario_000'
r=json.loads((mem/'attempt_result.json').read_text())
t=next(x['planner_trace'] for x in map(json.loads,(mem/'execution_trace.jsonl').open()) if x.get('kind')=='llm_output')
calls=[y for e in t['events'] for y in e['api_calls']]
last=dict(mode='parallel_memory',D=r['scenario_danger_score'],attack_frames=r['attack_executed_frames'],api_calls=len(calls),
          planner_seconds=t['elapsed_seconds'])
allrows=first+[last]
labels=['Single','Critic','Parallel','Memory']
fig,axes=plt.subplots(2,2,figsize=(10.5,7),constrained_layout=True)
for ax,key,title in zip(axes.ravel(),('D','attack_frames','api_calls','planner_seconds'),
                        ('Measured D','Executed attack frames','Actual API calls','Planner wall seconds')):
    vals=[row.get(key) for row in allrows]
    bars=ax.bar(labels,[0 if v is None else v for v in vals],color=['#517fa8','#d59a4c','#7757a8','#62a386'])
    high=max([v for v in vals if v is not None]+[.01]); ax.set(title=title,ylim=(0,high*1.35))
    for bar,value in zip(bars,vals):
        label='missing' if key=='D' and value is None else 'n/a' if value is None else f'{value:.4f}' if key=='D' else f'{value:.2f}' if key=='planner_seconds' else str(value)
        ax.text(bar.get_x()+bar.get_width()/2,(value or 0)+high*.04,label,ha='center',fontsize=8)
    ax.tick_params(axis='x',rotation=15)
fig.suptitle('One frozen multi-background scene; not a population-level efficacy estimate')
fig.savefig(out/'four_mode_smoke.png',dpi=160); plt.close(fig)

# 画像统计：分位数与 EWMA 已计算；CI 缺失如实标注。
profile=json.loads((root/'experiments/riskweaver_scene3_step40_1789701529208278813/smoke/single/movies/scenario_000/ego_profile.json').read_text())
metrics=profile['metrics']
selected=[('speed_mps','Speed (m/s)'),('acceleration_mps2','Acceleration (m/s²)'),
          ('following_gap_m','Following gap (m)'),('min_ttc_s','Min TTC (s)')]
fig,axes=plt.subplots(2,2,figsize=(10,6.5),constrained_layout=True)
for ax,(key,title) in zip(axes.ravel(),selected):
    m=metrics[key]; q=m['quantiles']
    ax.hlines(0,q['q10'],q['q90'],color='#4c80a7',lw=5,label='q10–q90')
    ax.scatter([q['q50']],[0],color='#173e63',s=85,zorder=3,label='median')
    ax.scatter([m['ewma']],[0],marker='D',color='#c56a42',s=75,zorder=3,label='EWMA')
    ax.set(title=f'{title} | n={m["n"]}',yticks=[],xlabel=title)
    ax.grid(axis='x',alpha=.17)
    ax.legend(fontsize=7,loc='upper right')
fig.suptitle(f'Ego profile after {profile["frames"]} observed frames; block CI unavailable in this run')
fig.savefig(out/'ego_profile_quantiles.png',dpi=160); plt.close(fig)
print(json.dumps(dict(figures=[str(p) for p in out.glob('*.png')],difficulty_attempts=len(values),
                      difficulty_valid=len(valid),difficulty_missing=len(missing),difficulty_hits=sum(.4<=v<=.6 for _,v in valid),
                      profile_frames=profile['frames']),ensure_ascii=False))
