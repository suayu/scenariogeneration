"""统一危险场景生成方法注册表。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    overrides: tuple[str, ...]
    uses_llm: bool
    fidelity: str


METHODS = {
    "ours": MethodSpec("ours", (
        "sim.traffic_model.backend=safe_sim_diffusion",
        "sim.traffic_model.guidance.mode=llm_joint",
        "sim.llm.attack_mode=joint",
        "sim.traffic_model.iterative_adversarial.profile_enabled=true",
        "sim.traffic_model.iterative_adversarial.escalation_enabled=true",
        "sim.traffic_model.iterative_adversarial.full_method_enabled=true",
    ), True, "full"),
    "safe_sim": MethodSpec("safe_sim", (
        "sim.traffic_model.backend=safe_sim_diffusion",
        "sim.traffic_model.guidance.mode=default",
        "sim.traffic_model.iterative_adversarial.profile_enabled=false",
        "sim.traffic_model.iterative_adversarial.escalation_enabled=false",
        "sim.traffic_model.iterative_adversarial.full_method_enabled=false",
    ), False, "original-guidance-compatible"),
    "scenario_dreamer": MethodSpec("scenario_dreamer", (
        "sim.mode=scenario_dreamer",
        "sim.traffic_model.backend=ctrl_sim",
        "sim.behaviour_model.tilt=-10",
        "sim.behaviour_model.run_name=ctrl_sim_waymo_1M_steps",
    ), False, "official-ctrl-sim-negative-tilt"),
}


def register_method(name, spec):
    """供外部比较方法注册适配器，不修改统一运行器。"""
    if name in METHODS:
        raise ValueError(f"method already registered: {name}")
    if not isinstance(spec, MethodSpec):
        raise TypeError("spec must be MethodSpec")
    METHODS[name] = spec
