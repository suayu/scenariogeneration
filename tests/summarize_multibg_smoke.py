"""汇总冻结多背景场景的真实请求、执行、安全与失败证据。"""
import csv
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

batch=Path(sys.argv[1]); rows=[]
for mode in ('single','conditional_critic','parallel_arbiter','parallel_memory'):
    folder=batch/'smoke'/mode/'movies/scenario_000'
    result_path=folder/'attempt_result.json'
    if not result_path.exists():
        rows.append(dict(mode=mode,status='configuration_failure',reason='parallel_memory requires scene-conditioned profile context'))
        continue
    result=json.loads(result_path.read_text())
    trace=[json.loads(line) for line in (folder/'execution_trace.jsonl').read_text().splitlines()]
    outputs=[x for x in trace if x.get('kind')=='llm_output']
    planner=[x.get('planner_trace') or {} for x in outputs]
    calls=[call for t in planner for event in t.get('events',[]) for call in event.get('api_calls',[])]
    token_counts=[(call.get('usage') or {}).get('total_tokens') for call in calls]
    api_calls=len(calls) if mode!='single' else sum(bool(x.get('validated_plan',{}).get('attack')) for x in outputs)
    role_plans=[dict(role=e.get('role'),target=(e.get('plan') or {}).get('attack_target_id'),strategy=(e.get('plan') or {}).get('strategy'))
                for t in planner for e in t.get('events',[])]
    rows.append(dict(mode=mode,status='passed',D=result['scenario_danger_score'],danger_valid=result['danger_valid'],
                     attack_plans=result['attack_plan_count'],attack_frames=result['attack_executed_frames'],
                     api_calls=api_calls,tokens=sum(token_counts) if token_counts and all(isinstance(t,int) for t in token_counts) else None,
                     planner_seconds=sum(float(t.get('elapsed_seconds') or 0) for t in planner) if mode!='single' else None,
                     wall_seconds=result['feasibility']['wall_seconds'],background_collision_frames=result['feasibility']['background_collision_frames'],
                     static_collision_frames=result['feasibility']['background_static_collision_frames'],
                     generation_failure=result.get('generation_failure'),role_plans=role_plans,
                     video_exists=(folder/'scenario_000.mp4').is_file(),trace_exists=(folder/'execution_trace.jsonl').is_file()))
(batch/'comparison.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False))
with (batch/'comparison.csv').open('w',newline='') as handle:
    fields=['mode','status','D','attack_plans','attack_frames','api_calls','tokens','planner_seconds','wall_seconds','background_collision_frames','generation_failure']
    writer=csv.DictWriter(handle,fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
valid=[r for r in rows if r['status']=='passed']; labels=['Single','Critic','Parallel']
for ax,key,title in zip(axes,('D','api_calls','planner_seconds'),('Measured D','API calls','Planner seconds')):
    vals=[r.get(key) for r in valid]
    bars=ax.bar(labels,[v or 0 for v in vals],color=['#517fa8','#d59a4c','#7757a8'])
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,(v or 0)+max([x or 0 for x in vals]+[.01])*.03,'unavailable' if v is None else (f'{v:.5f}' if key=='D' else f'{v:.2f}' if key=='planner_seconds' else str(v)),ha='center',fontsize=8)
    ax.set(title=title,ylim=(0,max([v or 0 for v in vals]+[.01])*1.3))
fig.suptitle('Frozen multi-background smoke; memory arm failed before planning')
fig.savefig(batch/'comparison.png',dpi=150)
print(json.dumps(rows,ensure_ascii=False,allow_nan=False))
