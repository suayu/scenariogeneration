from experiments.comparison.methods import METHODS, MethodSpec, register_method


def test_builtin_comparison_methods_have_distinct_paths():
    assert set(METHODS) >= {"ours", "safe_sim", "scenario_dreamer"}
    assert METHODS["ours"].uses_llm
    assert "sim.traffic_model.guidance.mode=default" in METHODS["safe_sim"].overrides
    assert "sim.behaviour_model.tilt=-10" in METHODS["scenario_dreamer"].overrides
    assert "sim.traffic_model.backend=ctrl_sim" in METHODS["scenario_dreamer"].overrides
    assert METHODS["scenario_dreamer"].fidelity == "official-ctrl-sim-negative-tilt"


def test_external_method_registration_interface():
    name = "unit_external_method"
    METHODS.pop(name, None)
    register_method(name, MethodSpec(name, ("sim.traffic_model.backend=log_replay",), False, "unit"))
    assert METHODS[name].fidelity == "unit"
    METHODS.pop(name)
