"""离线记忆必须引用真实证据，并保留失败和缺失危险度。"""
import json
from types import SimpleNamespace

import pytest

from policies.multiagent_memory import load_records, reflect, save_memory
from policies.multiagent_planner import MultiAgentPlanner


def record():
    return dict(record_id='r1', policy_id='ego1', scene_id='train', scene_conditions={'road': 'straight'},
                D=None, actual_attack_execution_frames=0, generation_failure_reason='profile_joint_dynamics')


def test_reflection_preserves_invalid_attempt_and_retrieves_matching_policy(tmp_path):
    def request(prompt):
        payload = json.loads(prompt)
        return dict(request_id=payload['request_id'], lessons=[dict(evidence_ids=['r1'], lesson='No executed attack; benefit is unknown.')])
    bank = reflect([record()], request)
    assert bank['source_records'][0]['D'] is None
    path = tmp_path/'memory.json'
    save_memory(bank, path)
    engine = MultiAgentPlanner(SimpleNamespace(mode='parallel_memory', memory_path=str(path)))
    assert len(engine.retrieve({'policy_id': 'ego1', 'scene_conditions': {'road': 'straight'}})) == 1
    assert engine.retrieve({'policy_id': 'ego2'}) == []
    with pytest.raises(ValueError, match='overlaps'):
        engine.retrieve({'policy_id': 'ego1', 'scene_id': 'train'})
    with pytest.raises(ValueError, match='memory context'):
        engine.run(SimpleNamespace(), {}, [])
    with pytest.raises(FileExistsError):
        save_memory(bank, path)


def test_memory_rejects_eval_leakage_and_fabricated_evidence(tmp_path):
    path = tmp_path/'returns.jsonl'
    path.write_text(json.dumps(record())+'\n')
    with pytest.raises(ValueError, match='overlaps'):
        load_records([path], ['train'])
    def request(prompt):
        return dict(request_id=json.loads(prompt)['request_id'], lessons=[dict(evidence_ids=['invented'], lesson='wrong')])
    with pytest.raises(ValueError, match='unknown evidence'):
        reflect([record()], request)
