"""核验单条离线复盘的证据范围，保留原始输出并另存可检索版本。"""
import json
from pathlib import Path
from policies.multiagent_memory import save_memory
from policies.multiagent_planner import fingerprint

base=Path('/home2/zhaoyx/scenario-dreamer/experiments/riskweaver_memory_train_1789703744058152089')
raw=json.loads((base/'unprofiled_memory.json').read_text())
assert raw['planning_regime']=='unprofiled' and raw['source_scene_ids']==['6_5.pkl']
assert len(raw['source_records'])==len(raw['records'])==1
source=raw['source_records'][0]; record=raw['records'][0]
assert source['D'] is None and source['actual_attack_execution_frames']==0
assert '18.7 m' in source['generation_failure_reason'] and record['evidence_id']==source['record_id']
verified=dict(raw)
verified['records']=[dict(record,lesson='One observed trajectory-only plan was rejected because its closest anchor was 18.7 m from the ego three-second prediction. No attack executed; D was unmeasured. Check interaction distance before proposing a similar plan.')]
verified['evidence_review']=dict(status='manually_checked_against_single_source',
                                 raw_bank_hash=fingerprint(raw),
                                 correction='Removed unsupported frequency claim from one observation.')
save_memory(verified,base/'unprofiled_memory_verified.json')
print(json.dumps(dict(path=str(base/'unprofiled_memory_verified.json'),records=1,D=None,source_scene='6_5.pkl')))
