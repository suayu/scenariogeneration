"""验证 LLM 服务连续失败时只停用高级攻击规划，不中断仿真。"""

from types import SimpleNamespace

from scenario_generator import AdversarialScenarioGenerator


class _UnavailablePlanner:
    """模拟返回服务端不可用错误的规划器。"""

    available = True
    use_multimodal = False

    def __init__(self):
        self.last_request_failed = False
        self.call_count = 0

    def generate_attack_plan(self, env_state, user_instruction, scene_image=None):
        self.call_count += 1
        self.last_request_failed = True
        return None


class _MinimalEnvironment:
    """提供失败路径所需的最小仿真器接口。"""

    def __init__(self):
        self.current_step = 0

    def get_state_for_planning(self):
        return {}


def test_llm_service_failure_disables_attacks_after_configured_retries():
    """第三次连续服务失败后，后续帧不得再调用 LLM。"""
    cfg = SimpleNamespace(sim=SimpleNamespace(llm=SimpleNamespace(max_consecutive_failures=3)))
    planner = _UnavailablePlanner()
    generator = AdversarialScenarioGenerator(cfg, [], llm_planner=planner)
    generator._ignore_call_times = 0
    environment = _MinimalEnvironment()

    for step in (12, 16, 20, 24):
        environment.current_step = step
        assert generator.step(environment, step) is True

    assert planner.call_count == 3
    assert generator._consecutive_llm_failures == 3
    assert generator._attacks_enabled is False
