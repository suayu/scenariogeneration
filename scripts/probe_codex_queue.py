"""真实 SSH/Codex 请求冒烟：远端验证语义后保存无凭据结果。"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policies.llm_adversarial_planner import LLMAdversarialPlanner

planner = LLMAdversarialPlanner(provider='codex', model_names=['codex-local'], model_name='codex-local')
state = dict(ego_state=[0, 0, 0, 10, 1.57079632679], route=[[0, 0], [0, 100]],
             agents=[dict(id=1, type=0, state=[0, 12, 0, 8, 1.57079632679])], static_obstacles=[])
plan = planner.generate_attack_plan(state, ['Use a feasible slow_down interaction with the lead vehicle.'],
                                   difficulty_context=dict(mode='target', target=.5, tolerance=.1))
result = dict(plan=plan, trace=planner.last_trace, service_failed=planner.last_request_failed)
destination = Path(sys.argv[1])
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open('x', encoding='utf-8') as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
print(json.dumps(result, ensure_ascii=False))
raise SystemExit(0 if plan and plan['attack'] and not planner.last_request_failed else 1)
