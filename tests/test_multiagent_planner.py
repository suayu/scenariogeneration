"""验证关闭兼容、独立并行、失败传播及有界质疑调用。"""
from types import SimpleNamespace
import json
import threading

import pytest

from policies.multiagent_planner import MultiAgentPlanner


def settings(mode):
    return SimpleNamespace(mode=mode, timeout_seconds=5)


def test_single_preserves_original_call_and_identity():
    expected = object()
    planner = SimpleNamespace(generate_attack_plan=lambda state, instruction, **kw: expected)
    assert MultiAgentPlanner(settings('single')).run(planner, {}, [], {'scene_image': None}) is expected
    assert not hasattr(planner, 'last_trace')


def test_parallel_workers_are_isolated_and_concurrent(monkeypatch):
    barrier = threading.Barrier(2, timeout=2)
    workers = []
    def rank(worker, state, instruction, context, image):
        workers.append(worker)
        role = context['collaboration']['role']
        barrier.wait()
        worker.last_trace = {'candidate_ranking': ['a', 'b'] if role == 'proposer_risk' else ['b', 'a']}
        return {'attack': True, 'candidate_id': 'a' if role == 'proposer_risk' else 'b'}
    monkeypatch.setattr('policies.profile_planner.rank_profile_candidates', rank)
    planner = SimpleNamespace()
    engine = MultiAgentPlanner(settings('parallel_arbiter'))
    result = engine.run(planner, {}, [], context={'candidates': [{'candidate_id': 'a'}, {'candidate_id': 'b'}]})
    assert workers[0] is not workers[1] and all(w is not planner for w in workers)
    assert result['candidate_id'] == 'b'
    assert len(engine.last_trace['events']) == 2


def test_service_failure_is_not_hidden_by_other_proposal(monkeypatch):
    def rank(worker, state, instruction, context, image):
        worker.last_request_failed = context['collaboration']['role'] == 'proposer_risk'
        return None if worker.last_request_failed else {'attack': True}
    monkeypatch.setattr('policies.profile_planner.rank_profile_candidates', rank)
    planner = SimpleNamespace()
    engine = MultiAgentPlanner(settings('parallel_arbiter'))
    assert engine.run(planner, {}, [], context={'candidates': [{'candidate_id': 'a'}]}) is None
    assert planner.last_request_failed


def test_critic_is_conditional_and_does_not_retry_service_failure(monkeypatch):
    calls = []
    def rank(worker, state, instruction, context, image):
        calls.append(context.get('collaboration', {}).get('role', 'baseline'))
        return {'attack': True}
    monkeypatch.setattr('policies.profile_planner.rank_profile_candidates', rank)
    engine = MultiAgentPlanner(settings('conditional_critic'))
    engine.run(SimpleNamespace(), {}, [], context={'candidates': [{'candidate_id': 'a'}], 'history': [{}]})
    assert calls == ['baseline']
    calls.clear()
    engine.run(SimpleNamespace(), {}, [], context={'candidates': [{'candidate_id': 'a'}]})
    assert calls == ['baseline', 'critic']


def test_invalid_mode_and_missing_memory_fail_before_calls():
    with pytest.raises(ValueError):
        MultiAgentPlanner(settings('unknown'))
    with pytest.raises(ValueError):
        MultiAgentPlanner(settings('parallel_memory'))


def test_empty_catalog_makes_no_agent_request(monkeypatch):
    calls = []
    expected = {'attack': False, 'reason': 'no_feasible_candidates'}
    def rank(worker, state, instruction, context, image):
        calls.append(context)
        assert 'collaboration' not in context
        return expected
    monkeypatch.setattr('policies.profile_planner.rank_profile_candidates', rank)
    engine = MultiAgentPlanner(settings('parallel_arbiter'))
    assert engine.run(SimpleNamespace(), {}, [], context={'candidates': []}) is expected
    assert len(calls) == 1
    assert engine.last_trace['events'] == []


def test_critic_stops_after_service_failure(monkeypatch):
    calls = []
    def rank(worker, state, instruction, context, image):
        calls.append(context.get('collaboration', {}).get('role', 'baseline'))
        worker.last_request_failed = True
        return None
    monkeypatch.setattr('policies.profile_planner.rank_profile_candidates', rank)
    planner = SimpleNamespace()
    engine = MultiAgentPlanner(settings('conditional_critic'))
    assert engine.run(planner, {}, [], context={'candidates': [{'candidate_id': 'a'}]}) is None
    assert calls == ['baseline']
    assert planner.last_request_failed


def test_single_profile_keeps_original_context_and_image(monkeypatch):
    context, state, instruction, picture = {'candidates': []}, {}, [], object()
    planner, expected = SimpleNamespace(), object()
    def rank(worker, actual_state, actual_instruction, actual_context, image):
        assert worker is planner and actual_state is state and actual_instruction is instruction
        assert actual_context is context and image is picture
        return expected
    monkeypatch.setattr('policies.profile_planner.rank_profile_candidates', rank)
    assert MultiAgentPlanner(settings('single')).run(planner, state, instruction, context=context, scene_image=picture) is expected


def test_conditional_first_proposal_has_identical_baseline_payload(monkeypatch):
    original = {'candidates': [{'candidate_id': 'a'}], 'history': {'attempts': 1}}
    def rank(worker, state, instruction, context, image):
        assert context == original
        return {'attack': True}
    monkeypatch.setattr('policies.profile_planner.rank_profile_candidates', rank)
    extended = dict(original, policy_id='ego', scene_id='test', scene_conditions={'road': 'straight'})
    MultiAgentPlanner(settings('conditional_critic')).run(SimpleNamespace(), {}, [], context=extended)


def test_unprofiled_memory_preserves_original_planner_path(tmp_path):
    """无画像检索只扩展协作提案，仍由原规划器验证离散计划。"""
    bank = dict(schema_version=1, planning_regime='unprofiled',
                source_scene_ids=['train.pkl'], records=[dict(policy_id='ego',
                scene_conditions={'road':'straight'}, scene_id='train.pkl',
                lesson='Recorded failure; danger unmeasured.', D=None)])
    path=tmp_path/'memory.json'
    path.write_text(json.dumps(bank),encoding='utf-8')
    seen=[]
    def generate(state,instructions,**kwargs):
        seen.append(instructions)
        return {'attack':True,'attack_target_id':2,'strategy':'cut_in'}
    planner=SimpleNamespace(generate_attack_plan=generate)
    engine=MultiAgentPlanner(SimpleNamespace(mode='parallel_memory',memory_path=str(path)))
    result=engine.run(planner,{'agents':[{'id':2}]},['original'],
                      memory_context={'policy_id':'ego','scene_id':'eval.pkl',
                                      'scene_conditions':{'road':'straight'}})
    assert result['attack'] and len(seen)==2
    assert all(instructions[0]=='original' for instructions in seen)
    assert engine.last_trace['memory_hits']==1
    assert all(len(instructions)==2 for instructions in seen)
    with pytest.raises(ValueError,match='overlaps'):
        engine.run(planner,{'agents':[]},['original'],memory_context={'policy_id':'ego','scene_id':'train.pkl'})


def test_unprofiled_memory_rejects_profile_training_bank(tmp_path):
    """画像筛选失败经验不能直接充当无画像规划证据。"""
    path=tmp_path/'memory.json'
    path.write_text(json.dumps(dict(schema_version=1,source_scene_ids=[],records=[])),encoding='utf-8')
    engine=MultiAgentPlanner(SimpleNamespace(mode='parallel_memory',memory_path=str(path)))
    with pytest.raises(ValueError,match='regime mismatch'):
        engine.run(SimpleNamespace(),{'agents':[]},[],memory_context={'policy_id':'ego'})


def test_unprofiled_feasibility_veto_is_explicit():
    """两个有效角色意见不同时，不依赖角色名偶然排序。"""
    def generate(state,instructions,**kwargs):
        collaboration=json.loads(instructions[-1])
        if collaboration['role']=='proposer_feasibility':
            return {'attack':False,'reason':'avoidability_unverified'}
        return {'attack':True,'attack_target_id':2,'strategy':'cut_in'}
    planner=SimpleNamespace(generate_attack_plan=generate)
    engine=MultiAgentPlanner(settings('parallel_arbiter'))
    result=engine.run(planner,{'agents':[{'id':2}]},['original'])
    assert result=={'attack':False,'reason':'avoidability_unverified'}
    assert engine.last_trace['arbitration_reason']=='feasibility_veto'
    assert len(engine.last_trace['events'])==2
    assert {row['role'] for row in engine.last_trace['agent_opinions']}=={'proposer_risk','proposer_feasibility'}
    assert engine.last_trace['agent_opinions'][1]['reason']=='avoidability_unverified'
    assert 'veto' in engine.last_trace['arbitration_detail'].lower()
