"""四模式真实 API 协议冒烟；固定同一输入，不将此结果冒充闭环效果。"""
import copy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from policies.llm_adversarial_planner import LLMAdversarialPlanner
from policies.multiagent_memory import load_records, reflect, save_memory
from policies.multiagent_planner import AuditedClient, MODES, MultiAgentPlanner, fingerprint

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
OUT = ROOT/'experiments'/('riskweaver_multiagent_api_'+str(time.time_ns()))
OUT.mkdir()


def write(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def planner():
    return LLMAdversarialPlanner(provider='dashscope', model_name='qwen3.5-plus',
                                 model_names=['qwen3.5-plus'], attack_mode='trajectory_only')


def main():
    write('source_hashes.json', {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                               for name in ['policies/multiagent_planner.py', 'policies/multiagent_memory.py',
                                            'policies/profile_planner.py', 'policies/llm_adversarial_planner.py',
                                            'tests/probe_multiagent_api.py']})
    trace = ROOT/'experiments/riskweaver_bailian_profile_20260916_r4/smoke/trajectory_profile/movies/scenario_000/execution_trace.jsonl'
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    contexts = [r for r in rows if r.get('kind') == 'profile_ranking_input' and r['context']['candidates']]
    entry = contexts[0]
    state = next(r['state'] for r in rows if r.get('kind') == 'llm_input' and r['step'] == entry['step'])
    context = copy.deepcopy(entry['context'])
    returns_path = ROOT/'experiments/riskweaver_bailian_profile_20260916_r4/smoke/trajectory_profile/movies/scenario_000/strategy_returns.jsonl'
    current = [json.loads(line) for line in returns_path.read_text().splitlines()][-1]
    context.update(policy_id=current['policy_id'], scene_conditions=current['scene_conditions'])
    source = ROOT/'experiments/riskweaver_bailian_profile_20260916_r3/smoke/trajectory_profile/movies/scenario_000/strategy_returns.jsonl'
    records = load_records([source], [current['scene_id']])
    write('frozen_input.json', dict(state=state, context=context, input_hash=fingerprint([state, context])))
    api = planner()
    reflection_calls = []
    api.client = AuditedClient(api.client.with_options(timeout=60, max_retries=0), reflection_calls)
    try:
        bank = reflect(records, lambda prompt: api._request_attack_plan(prompt)[1])
        save_memory(bank, OUT/'memory.json')
    except Exception as error:
        write('failure.json', dict(stage='offline_reflection', error_type=type(error).__name__, calls=reflection_calls))
        raise
    write('reflection_calls.json', reflection_calls)
    results = []
    for mode in MODES:
        api = planner()
        calls = []
        if mode == 'single':
            api.client = AuditedClient(api.client.with_options(timeout=60, max_retries=0), calls)
        engine = MultiAgentPlanner(SimpleNamespace(mode=mode, timeout_seconds=60, memory_path=str(OUT/'memory.json')))
        started = time.monotonic()
        plan = engine.run(api, state, ['Generate an avoidable adversarial interaction.'], context=copy.deepcopy(context))
        trace = api.last_trace
        if mode != 'single':
            calls = [c for event in trace['events'] for c in event['api_calls']]
        result = dict(mode=mode, elapsed_seconds=time.monotonic()-started, api_calls=calls,
                      accepted=bool(plan and plan.get('attack')), plan=plan, trace=trace)
        results.append(result)
        write('results.json', results)
        print(json.dumps(dict(mode=mode, seconds=result['elapsed_seconds'], calls=len(calls), accepted=result['accepted'])), flush=True)
        if api.last_request_failed:
            raise RuntimeError('planner_service_failure: remaining arms stopped')
    print(str(OUT), flush=True)


if __name__ == '__main__':
    main()
