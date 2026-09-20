"""验证实验专用规划时刻配置及原默认调度。"""
from types import SimpleNamespace

from scenario_generator import AdversarialScenarioGenerator


def generator(min_step, interval):
    value = object.__new__(AdversarialScenarioGenerator)
    value.min_planning_step = min_step
    value.attack_frequency = interval
    value.attack_duration = 10
    value.last_attack_frame = 0
    value.last_query_frame = 0
    value._attacks_enabled = True
    value._llm_call_times = 0
    value._max_call_times = 10
    value._ignore_call_times = 0
    calls = []
    def plan(env):
        calls.append(env.current_step)
        value.last_query_frame = env.current_step
    value.plan_and_inject = plan
    return value, calls


def test_default_queries_at_first_frame_and_original_interval():
    value, calls = generator(0, 3)
    env = SimpleNamespace(attack_intent=None, current_step=0)
    for step in range(15):
        env.current_step = step
        value.step(env, step)
    assert calls == [0, 10, 14]


def test_deferred_first_query_preserves_budget_until_preheat():
    value, calls = generator(20, 10)
    env = SimpleNamespace(attack_intent=None, current_step=0)
    for step in range(22):
        env.current_step = step
        value.step(env, step)
    assert calls == [20]
    assert value._llm_call_times == 1
