"""Safe-Sim 扩散控制器的兼容入口。

旧实现构造了不完整的合成批次并加载行为克隆策略。调用方现在应向真实的 Safe-Sim
控制器传入 ``ScenarioFrame``。
"""

from policies.diffusion_model_wrapper import SafeSimDiffusionController


class TrajectoryPredictor(SafeSimDiffusionController):
    pass
