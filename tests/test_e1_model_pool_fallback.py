"""模型池故障回退、永久标记、多进程安全和实验配置验证。"""
from concurrent.futures import ThreadPoolExecutor
import pytest
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from policies.codex_planner_bridge import PlannerServiceFailure
from experiments.comparison.e1_protocol import planning_model_pool, group_overrides, protocol_snapshot


def test_404_permanently_disables_then_calls_next(monkeypatch):
    planner=object.__new__(LLMAdversarialPlanner)
    planner.provider='dashscope';planner.model_names=('removed','healthy')
    planner._preferred_model_index=0;planner.model_name='removed'
    calls=[];marks=[]
    class Error(Exception): status_code=404
    def request(model, prompt, image):
        calls.append(model)
        if model=='removed': raise Error('model not exist')
        return '',{'attack':False}
    monkeypatch.setattr(planner,'_request_attack_plan_once',request)
    monkeypatch.setattr(planner,'_comment_out_exhausted_model',lambda name,reason: marks.append((name,reason)) or True)
    assert planner._request_attack_plan('json')[1]=={'attack':False}
    assert calls==['removed','healthy'] and planner.model_names==('healthy',)
    assert marks==[('removed','model_not_found')]


def test_duplicate_entries_and_404_persist_across_new_processes(tmp_path):
    p=tmp_path/'planner.py'
    p.write_text('FREE_MODEL_POOL = (\n "gone",\n "keep",\n "gone",\n)\nOTHER="gone"\n')
    assert LLMAdversarialPlanner._comment_out_exhausted_model('gone',p,reason='model_not_found')
    s=p.read_text()
    assert s.count('# "gone",')==2 and 'OTHER="gone"' in s and ' "keep",' in s
    assert LLMAdversarialPlanner._exhausted_models_from_source(p)=={'gone'}


def test_concurrent_disabling_preserves_all_updates(tmp_path):
    p=tmp_path/'planner.py';names=[f'm{i}' for i in range(12)]
    p.write_text('FREE_MODEL_POOL = (\n'+''.join(f' "{n}",\n' for n in names)+' "keep",\n)\n')
    with ThreadPoolExecutor(max_workers=12) as executor:
        results=list(executor.map(lambda name:LLMAdversarialPlanner._comment_out_exhausted_model(name,p),names))
    assert all(results)
    assert LLMAdversarialPlanner._exhausted_models_from_source(p)==set(names)
    assert ' "keep",\n' in p.read_text()


def test_existing_worker_skips_disabled_model_before_call(monkeypatch):
    p=object.__new__(LLMAdversarialPlanner)
    p.provider='dashscope';p.model_names=('gone','healthy');p._preferred_model_index=0
    monkeypatch.setattr(p,'_exhausted_models_from_source',lambda:{'gone'})
    seen=[]
    monkeypatch.setattr(p,'_request_attack_plan_once',lambda name,*a:seen.append(name) or ('',{}))
    p._request_attack_plan('json')
    assert seen==['healthy']


def test_all_disabled_is_service_failure_not_no_attack(monkeypatch):
    p=object.__new__(LLMAdversarialPlanner)
    p.provider='dashscope';p.model_names=('gone',);p._preferred_model_index=0
    monkeypatch.setattr(p,'_exhausted_models_from_source',lambda:{'gone'})
    with pytest.raises(PlannerServiceFailure) as error:p._request_attack_plan('json')
    assert p._is_service_failure(error.value)


def test_profile_groups_share_same_frozen_pool():
    pool,skipped=planning_model_pool(['qwen3.5-ocr','qwen3.8-flash','kimi-k3','qwen3.8-flash','qwen-mt-uni'])
    assert pool==['qwen3.8-flash','kimi-k3'] and len(skipped)==2
    a=group_overrides('riskweaver_profile_off',models=pool)
    b=group_overrides('riskweaver_profile_on',models=pool)
    assert set(a)^set(b)=={'sim.traffic_model.iterative_adversarial.profile_enabled=false','sim.traffic_model.iterative_adversarial.profile_enabled=true'}
    assert 'sim.llm.model_names=[qwen3.8-flash,kimi-k3]' in a
    assert protocol_snapshot(models=pool)['llm_fixed_single_model'] is False
