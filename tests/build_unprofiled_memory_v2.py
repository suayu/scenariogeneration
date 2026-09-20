"""用互不重叠的真实闭环训练尝试，构造四组实验的离线复盘库。"""
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from policies.ego_profile import policy_id_for_sim, scene_conditions
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from policies.multiagent_memory import load_records, reflect, save_memory
from policies.multiagent_planner import AuditedClient


ROOT = Path('/home2/zhaoyx/scenario-dreamer')
TRAIN = [
    'riskweaver_four_arm_multibg_1789702359802287367',
    'riskweaver_four_arm_second_scene_1789722510720317160',
    'riskweaver_memory_train_1789703744058152089',
    'riskweaver_four_arm_scene14_1789723979086190975',
]
OUT = ROOT / 'experiments/riskweaver_unprofiled_memory_v2_20260918'


def source_record(batch_name):
    # 使用闭环实测值，不从 LLM 的计划文字猜测攻击效果。
    batch = ROOT / 'experiments' / batch_name
    folder = batch / 'smoke/single/movies/scenario_000'
    result = json.loads((folder / 'attempt_result.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (folder / 'execution_trace.jsonl').read_text(encoding='utf-8').splitlines()]
    state = next(row for row in rows if row.get('kind') == 'state' and row['step'] == 0)
    llm = next(row for row in rows if row.get('kind') == 'llm_output')
    with Path(result['scenario_source']).open('rb') as handle:
        source = pickle.load(handle)
    env = SimpleNamespace(ego_state=np.asarray(state['ego'], float),
                          data_dict={'agent': [np.asarray(state['agents'], float)]},
                          agent_active=np.asarray(state['active'], bool),
                          scenario_dict={'route': source['route']})
    config = OmegaConf.load(batch / 'smoke/single/resolved_config.yaml')
    plan = llm.get('validated_plan') or {}
    if result['attack_executed_frames']:
        failure = None
    elif result.get('generation_failure'):
        failure = str(result['generation_failure'])
    elif llm.get('request_failed'):
        failure = 'planner_service_failure'
    else:
        failure = 'plan_validation_failed_or_no_attack'
    return dict(record_id=hashlib.sha256((batch_name + ':single:0').encode()).hexdigest()[:32],
                policy_id=policy_id_for_sim(config.sim),
                scene_id=Path(result['scenario_source']).name,
                scene_conditions=scene_conditions(env),
                strategy=plan.get('strategy') if plan.get('attack') else None,
                target=plan.get('attack_target_id') if plan.get('attack') else None,
                actual_attack_execution_frames=result['attack_executed_frames'],
                D=result['scenario_danger_score'],
                generation_failure_reason=failure,
                planning_regime='unprofiled', attack_mode='trajectory_only',
                evidence=dict(planner_service_failure=bool(result['planner_service_failure_count']),
                              validated_plan=bool(plan.get('attack')),
                              simulated_frames=result['executed_steps'],
                              source_batch=batch_name))


def main():
    OUT.mkdir(exist_ok=False)
    records = [source_record(batch) for batch in TRAIN]
    assert len({row['scene_id'] for row in records}) == len(records)
    assert len({row['policy_id'] for row in records}) == 1
    assert any(row['actual_attack_execution_frames'] == 0 for row in records)
    assert any(row['actual_attack_execution_frames'] > 0 for row in records)
    record_path = OUT / 'training_records.jsonl'
    record_path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'
                                   for row in records), encoding='utf-8')
    training = load_records([record_path])
    planner = LLMAdversarialPlanner(provider='dashscope', model_name='qwen3.5-plus',
                                    model_names=['qwen3.5-plus'], attack_mode='trajectory_only')
    calls = []
    planner.client = AuditedClient(planner.client.with_options(timeout=60, max_retries=0), calls)
    try:
        bank = reflect(training, lambda prompt: planner._request_attack_plan(prompt)[1])
        bank['planning_regime'] = 'unprofiled'
        save_memory(bank, OUT / 'memory_raw.json')
    finally:
        (OUT / 'reflection_calls.json').write_text(
            json.dumps(calls, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(out=str(OUT), training_scenes=bank['source_scene_ids'],
                          source_records=len(records), lessons=len(bank['records']),
                          reflection_seconds=bank['reflection_elapsed_seconds'],
                          api_calls=len(calls)), ensure_ascii=False))


if __name__ == '__main__':
    main()
