"""从全部尝试与角色追踪汇总实际计量，不将规划成功当作执行成功。"""
import json
from pathlib import Path
import sys


root = Path(sys.argv[1])
summary = []
for result_path in sorted(root.glob('smoke/*/movies/scenario_*/attempt_result.json')):
    result = json.loads(result_path.read_text())
    trace = [json.loads(line) for line in result_path.with_name('execution_trace.jsonl').read_text().splitlines()]
    outputs = [r.get('planner_trace') or {} for r in trace if r.get('kind') == 'llm_output']
    calls, seconds = 0, 0.0
    for output in outputs:
        seconds += output.get('elapsed_seconds', 0)
        if 'events' in output:
            calls += sum(len(e.get('api_calls', [])) for e in output['events'])
        elif output.get('plan_validation') != 'no_feasible_candidates':
            # 现有单 Agent 追踪没有令牌明细；此数只表示一次已发出的逻辑请求。
            calls += 1
    summary.append(dict(mode=result_path.parents[2].name,
                        accepted_plans=result['attack_plan_count'],
                        executed_attack_frames=result['attack_executed_frames'],
                        D=result['scenario_danger_score'], failure=result.get('generation_failure'),
                        llm_seconds=seconds, request_count=calls,
                        simulation_seconds=result['feasibility']['wall_seconds'],
                        background_collision_frames=result['feasibility']['background_collision_frames'],
                        background_static_collision_frames=result['feasibility']['background_static_collision_frames'],
                        video=result['video_path']))
print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
