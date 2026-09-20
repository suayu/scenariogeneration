"""从独立训练场景的真实失败记录生成无画像冻结复盘库。"""
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

root=Path('/home2/zhaoyx/scenario-dreamer')
train=root/'experiments/riskweaver_memory_train_1789703744058152089'
folder=train/'smoke/single/movies/scenario_000'
result=json.loads((folder/'attempt_result.json').read_text())
assert result['planner_service_failure_count']==0 and result['attack_executed_frames']==0
assert result['scenario_danger_score'] is None and result['scenario_source'].endswith('/6_5.pkl')
trace=[json.loads(line) for line in (folder/'execution_trace.jsonl').read_text().splitlines()]
state=next(x for x in trace if x.get('kind')=='state' and x['step']==0)
output=next(x for x in trace if x.get('kind')=='llm_output')
assert output['validated_plan'] is None and output['request_failed'] is False
with open(result['scenario_source'],'rb') as handle: source=pickle.load(handle)
env=SimpleNamespace(ego_state=np.asarray(state['ego'],float),
                    data_dict={'agent':[np.asarray(state['agents'],float)]},
                    agent_active=np.asarray(state['active'],bool),
                    scenario_dict={'route':source['route']})
config=OmegaConf.load(train/'smoke/single/resolved_config.yaml')
policy_id=policy_id_for_sim(config.sim)
record=dict(record_id=hashlib.sha256((str(train)+':0').encode()).hexdigest()[:32],
            policy_id=policy_id,scene_id=Path(result['scenario_source']).name,
            scene_conditions=scene_conditions(env),strategy=None,target=None,
            actual_attack_execution_frames=0,D=None,
            generation_failure_reason='plan_validation_failed: nearest anchor to predicted ego at 3 s = 18.7 m; no effective interaction',
            planning_regime='unprofiled',attack_mode='trajectory_only',
            evidence=dict(planner_service_failure=False,validated_plan=False,simulated_frames=result['executed_steps']))
record_path=train/'unprofiled_training_records.jsonl'
with record_path.open('x',encoding='utf-8') as handle:
    handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
records=load_records([record_path],excluded_scene_ids=['23_5.pkl'])
client=LLMAdversarialPlanner(provider='dashscope',model_name='qwen3.5-plus',model_names=['qwen3.5-plus'],attack_mode='trajectory_only')
calls=[]
client.client=AuditedClient(client.client.with_options(timeout=60,max_retries=0),calls)
try:
    bank=reflect(records,lambda prompt: client._request_attack_plan(prompt)[1])
    bank['planning_regime']='unprofiled'
    assert bank['source_scene_ids']==['6_5.pkl'] and all(row['D'] is None for row in bank['records'])
    save_memory(bank,train/'unprofiled_memory.json')
finally:
    (train/'reflection_calls.json').write_text(json.dumps(calls,ensure_ascii=False,indent=2,allow_nan=False))
print(json.dumps(dict(bank_path=str(train/'unprofiled_memory.json'),source_scenes=bank['source_scene_ids'],
                      records=len(bank['records']),reflection_seconds=bank['reflection_elapsed_seconds'],
                      api_calls=len(calls)),ensure_ascii=False))
