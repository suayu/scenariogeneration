"""只读输出实验进度与关键审计摘要，不展示凭据或完整配置。"""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
exit_file = root / 'exit_status.json'
print('exit:', exit_file.read_text() if exit_file.exists() else 'running_or_queued')
for result_file in sorted((root / 'movies').glob('scenario_*/attempt_*/attempt_result.json')):
    result = json.loads(result_file.read_text())
    print(json.dumps({key: result.get(key) for key in (
        'scenario_index', 'replay_count', 'executed_steps', 'scenario_danger_score',
        'generation_failure', 'collision', 'completed', 'progress', 'attack_plan_count',
        'obstacle_plan_count', 'feasibility')}, ensure_ascii=False))
traces = sorted((root / 'movies').glob('scenario_*/attempt_*/execution_trace.jsonl'))
if traces:
    trace = traces[-1]
    last = None
    queries = []
    predictions = []
    for line in trace.open():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        last = row
        if row['kind'] == 'llm_input':
            queries.append({'step': row['step'], 'agent_count': len(row['state']['agents']),
                            'ids': [a['id'] for a in row['state']['agents']]})
        if row['kind'] == 'prediction':
            predictions.append({'step': row['step'], 'intensity': row['intensity'],
                                'controls': row['controls']})
    print('latest_trace:', str(trace), 'last_step:', last['step'] if last else None)
    print('recent_query_inputs:', json.dumps(queries[-6:]))
    print('latest_applied_controls:', json.dumps(predictions[-1:] if predictions else []))
