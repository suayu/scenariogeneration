"""校准四组的真实请求、耗时、执行、安全与视频清单，不混淆逻辑规划次数。"""
import csv
import json
from pathlib import Path
import sys

batch=Path(sys.argv[1])
names=('single','conditional_critic','parallel_arbiter','parallel_memory')
memory=json.loads(Path('/home2/zhaoyx/scenario-dreamer/experiments/riskweaver_multiagent_api_1789627364562698612/memory.json').read_text())
rows=[]
for name in names:
    scenario=next((batch/'smoke'/name/'movies').glob('scenario_*'))
    result=json.loads((scenario/'attempt_result.json').read_text())
    records=[json.loads(s) for s in (scenario/'execution_trace.jsonl').read_text().splitlines()]
    planning=[x for x in records if x.get('kind')=='llm_output']
    traces=[x.get('planner_trace') for x in planning]
    if name=='single':
        # 原单 Agent 不记录 usage；成功返回的单模型请求可计数，但 token 保持未知。
        api_calls=sum(bool(x.get('validated_plan',{}).get('attack')) for x in planning)
        tokens=None
    else:
        calls=[call for trace in traces for event in trace.get('events',[])
               for call in event.get('api_calls',[])]
        api_calls=len(calls)
        totals=[x.get('usage',{}).get('total_tokens') for x in calls]
        tokens=sum(totals) if all(isinstance(x,int) for x in totals) else None
    video=Path(result['video_path'])
    row=dict(mode=name,scenario='derived_single_attacker_6_5',planning_opportunities=result['llm_call_count'],
             actual_api_calls=api_calls,total_tokens=tokens,
             planner_seconds=sum(float(t.get('elapsed_seconds') or 0) for t in traces),
             simulation_wall_seconds=result['feasibility']['wall_seconds'],
             attack_plans=result['attack_plan_count'],attack_executed_frames=result['attack_executed_frames'],
             D=result['scenario_danger_score'],D_valid=result['danger_valid'],
             background_collision_frames=result['feasibility']['background_collision_frames'],
             background_static_collision_frames=result['feasibility']['background_static_collision_frames'],
             generation_failure=result.get('generation_failure'),
             planner_service_failures=result['planner_service_failure_count'],
             memory_hits=sum(int(t.get('memory_hits',0)) for t in traces),
             critic_triggers=[trigger for t in traces for trigger in t.get('critic_triggers',[])],
             video_exists=video.is_file() and video.stat().st_size>0,
             trace_exists=(scenario/'execution_trace.jsonl').is_file(),
             result_exists=(scenario/'attempt_result.json').is_file())
    rows.append(row)
output=batch/'calibration_comparison.json'
output.write_text(json.dumps(dict(rows=rows,offline_reflection_seconds=memory['reflection_elapsed_seconds'],
    offline_reflection_source_scenes=memory['source_scene_ids'],
    interpretation='single derived scene, one planning point; not real-traffic success-rate evidence'),indent=2,ensure_ascii=False,allow_nan=False))
with (batch/'calibration_comparison.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows,ensure_ascii=False,allow_nan=False))
