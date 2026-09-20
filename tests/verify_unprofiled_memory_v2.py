"""审查离线 LLM 经验，保留原件并冻结只陈述实测事实的检索版本。"""
import json
from pathlib import Path

from policies.multiagent_memory import save_memory
from policies.multiagent_planner import fingerprint


BASE = Path('/home2/zhaoyx/scenario-dreamer/experiments/riskweaver_unprofiled_memory_v2_20260918')
raw = json.loads((BASE / 'memory_raw.json').read_text(encoding='utf-8'))
source = {row['scene_id']: row for row in raw['source_records']}
assert set(source) == {'23_5.pkl', '17_7.pkl', '6_5.pkl', '26_2.pkl'}
assert len({row['policy_id'] for row in source.values()}) == 1


def lesson(row):
    # 单场景结果只支持事实描述，不能据此声称策略普遍有效或导致 D 上升。
    frames, danger = row['actual_attack_execution_frames'], row['D']
    if frames:
        assert danger is not None and row['strategy'] is not None
        return (f"One observed {row['scene_id']} run used {row['strategy']} on target "
                f"{row['target']}; {frames} attack frames executed and measured D={danger:.5f}. "
                "The causal gain over another planner was not measured in this record.")
    assert danger is None and row['strategy'] is None
    return (f"One observed {row['scene_id']} run produced no validated attack and zero "
            "executed attack frames. D was unmeasured (null), not zero; the planner service did not fail.")


verified = dict(raw)
verified['records'] = [dict(policy_id=row['policy_id'], scene_conditions=row['scene_conditions'],
                            scene_id=row['scene_id'], evidence_id=row['record_id'],
                            lesson=lesson(row), strategy=row['strategy'], D=row['D'],
                            actual_attack_execution_frames=row['actual_attack_execution_frames'],
                            generation_failure_reason=row['generation_failure_reason'])
                       for row in raw['source_records']]
verified['evidence_review'] = dict(status='checked_against_four_attempt_results',
                                   raw_bank_hash=fingerprint(raw),
                                   correction='Removed cross-scene efficacy and causal claims from single-run observations.')
save_memory(verified, BASE / 'memory_verified.json')
print(json.dumps(dict(path=str(BASE / 'memory_verified.json'),
                      source_scenes=verified['source_scene_ids'],
                      records=len(verified['records'])), ensure_ascii=False))
