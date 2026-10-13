from types import SimpleNamespace

from experiments.comparison.formal_metrics import FormalRiskObserver


class FakeRisk:
    def __init__(self, ea=2.0, event=None):
        self.ea = ea
        self.event = event
        self.reset_calls = 0
        self.begin_calls = 0

    def reset_episode(self):
        self.reset_calls += 1

    def evaluate_ea(self, env):
        return self.ea

    def begin_attack(self, env):
        self.begin_calls += 1

    def evaluate_dangerous_reachability(self, env):
        return self.event


def config():
    weights = SimpleNamespace(
        collision=0.30, near_miss=0.15, min_ttc=0.15,
        max_ea=0.12, mean_ea=0.04,
        max_reachability_difficulty=0.12,
        mean_reachability_difficulty=0.04,
        unsolvable=0.05, off_route=0.01,
        incomplete=0.01, low_progress=0.01,
    )
    return SimpleNamespace(
        composite=SimpleNamespace(
            weights=weights, ttc_scale_seconds=3.0, ea_scale_mps2=3.0
        )
    )


def test_missing_joint_prediction_keeps_danger_null():
    risk = FakeRisk(event={"difficulty": 0.4, "dangerous_solvable": True})
    observer = FormalRiskObserver(config(), risk)
    env = SimpleNamespace(pending_joint_trajectory=None)
    observer.begin_frame(env)
    assert observer.finish_prediction(env) is None
    record = observer.finish_transition({}, 2.0)
    assert not record["valid"]
    assert record["measurement_status"] == "missing_joint_prediction"
    assert observer.result()["danger"] is None


def test_valid_joint_prediction_produces_auditable_danger():
    event = {"difficulty": 0.4, "dangerous_solvable": True}
    risk = FakeRisk(ea=2.0, event=event)
    observer = FormalRiskObserver(config(), risk)
    env = SimpleNamespace(pending_joint_trajectory=object())
    observer.begin_frame(env)
    assert observer.finish_prediction(env) == event
    record = observer.finish_transition({"collision": False}, 2.0)
    result = observer.result()
    assert record["valid"] and 0.0 <= record["difficulty"] <= 1.0
    assert result["valid"] and result["valid_frame_count"] == 1
    assert result["danger"] == record["difficulty"]


def test_invalid_ea_never_becomes_zero_danger():
    event = {"difficulty": 0.4, "dangerous_solvable": True}
    observer = FormalRiskObserver(config(), FakeRisk(ea=float("nan"), event=event))
    env = SimpleNamespace(pending_joint_trajectory=object())
    observer.begin_frame(env)
    observer.finish_prediction(env)
    record = observer.finish_transition({}, 2.0)
    assert record["measurement_status"] == "invalid_ea"
    assert observer.result()["danger"] is None
