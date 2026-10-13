"""正式基线的共同低反应余量指标观察器。"""

import copy

import numpy as np

from policies.risk_metrics import (
    AdversarialRiskMetrics,
    compute_scenario_danger_score,
)


class FormalRiskObserver:
    """按帧对齐干预前基线、未来联合预测和执行后自车响应。"""

    def __init__(self, evaluation_cfg, risk_metrics=None):
        self.evaluation_cfg = evaluation_cfg
        self.risk = risk_metrics or AdversarialRiskMetrics(evaluation_cfg)
        self.records = []
        self._pending_ea = None
        self._pending_reachability = None
        self._prediction_available = False
        self._expect_joint_prediction = True

    def reset(self):
        """开始新场景；不得把上一场的测量带入当前结果。"""
        self.risk.reset_episode()
        self.records = []
        self._pending_ea = None
        self._pending_reachability = None
        self._prediction_available = False
        self._expect_joint_prediction = True

    def begin_frame(self, env, expect_joint_prediction=True):
        """在高级计划、障碍物或联合轨迹改变环境前冻结自然基线。"""
        self._pending_ea = self.risk.evaluate_ea(env)
        # CtRL-Sim 没有与当前帧对齐的联合未来预测，跳过昂贵且不可比较的基线传播。
        self._expect_joint_prediction = bool(expect_joint_prediction)
        if self._expect_joint_prediction:
            self.risk.begin_attack(env)
        self._pending_reachability = None
        self._prediction_available = False

    def finish_prediction(self, env, forecast=None, source="native", error=None):
        """联合预测存在时才计算干预后的可达性；缺失预测不以 CV 冒充。"""
        joint = getattr(env, "pending_joint_trajectory", None) if source == "native" else forecast
        # 同一计划跨帧记录响应空间，不混入其它攻击窗口。
        intent = getattr(env, "attack_intent", None)
        self._attack = ({k: intent.get(k) for k in ("source_step", "target_id", "strategy")}
                        if intent else None)
        self._source_step = getattr(env, "current_step", None)
        self._frame_dt = getattr(env, "dt", None)
        self._prediction_source = source
        self._prediction_error = error
        if source != "native" and joint is not None:
            from experiments.comparison.ctrl_sim_forecast import validate_forecast
            try:
                validate_forecast(joint, env, self.risk.reachability.horizon)
            except ValueError as exc:
                self._prediction_error, joint = str(exc), None
        self._prediction_available = self._expect_joint_prediction and joint is not None
        if self._prediction_available:
            # 评估对象是浅副本；真实执行仍消费原 SAFE-SIM 联合轨迹。
            evaluation_env = env
            if source != "native":
                evaluation_env = copy.copy(env)
                evaluation_env.pending_joint_trajectory = joint
            self._pending_reachability = self.risk.evaluate_dangerous_reachability(evaluation_env)
        return self._pending_reachability

    def finish_transition(self, info, min_ttc):
        """在状态转移后合并响应指标，并生成一帧有效或无效的审计记录。"""
        event = self._pending_reachability
        ea = self._pending_ea
        valid = bool(
            self._prediction_available
            and event is not None
            and np.isfinite(event.get("difficulty", np.nan))
            and ea is not None
            and np.isfinite(ea)
        )
        reason = None
        if not self._prediction_available:
            reason = getattr(self, "_prediction_error", None) or "missing_joint_prediction"
        elif event is None or not np.isfinite(event.get("difficulty", np.nan)):
            reason = "invalid_reachability"
        elif ea is None or not np.isfinite(ea):
            reason = "invalid_ea"

        danger = None
        metrics = None
        if valid:
            collision = bool((info or {}).get("collision", False))
            near_miss = bool(np.isfinite(min_ttc) and min_ttc < 1.5)
            metrics = {
                "collision_rate": float(collision),
                "near_miss_rate": float(near_miss),
                "avg_min_ttc": float(min_ttc),
                "max_evasive_acceleration_mps2": float(ea),
                "mean_evasive_acceleration_mps2": float(ea),
                "reachability_event_count": 1,
                "reachability_difficulty": float(event["difficulty"]),
                "mean_reachability_difficulty": float(event["difficulty"]),
                "dangerous_scene_solvable": float(event["dangerous_solvable"]),
            }
            danger = float(
                compute_scenario_danger_score(
                    metrics, self.evaluation_cfg.composite
                )
            )
            valid = bool(np.isfinite(danger) and 0.0 <= danger <= 1.0)
            if not valid:
                danger, reason = None, "invalid_danger"

        record = {
            "difficulty": danger,
            "weight": 1,
            "valid": valid,
            "measurement_status": "valid" if valid else reason,
            "ea_mps2": float(ea) if ea is not None and np.isfinite(ea) else None,
            "ea_diagnostics": getattr(self.risk, "last_ea_diagnostics", None),
            "prediction_source": getattr(self, "_prediction_source", "native"),
            "prediction_error": getattr(self, "_prediction_error", None),
            "reachability": event,
            "attack": getattr(self, "_attack", None),
            "source_step": getattr(self, "_source_step", None),
            "frame_dt": getattr(self, "_frame_dt", None),
            "metrics": metrics,
        }
        self.records.append(record)
        self._pending_ea = None
        self._pending_reachability = None
        self._prediction_available = False
        return record

    def result(self):
        """按有效帧数加权；没有有效帧时严格返回 null。"""
        valid = [
            record for record in self.records
            if record["valid"] and record["difficulty"] is not None
        ]
        weight = sum(int(record["weight"]) for record in valid)
        return {
            "danger": (
                sum(record["difficulty"] * int(record["weight"]) for record in valid)
                / weight
                if weight else None
            ),
            "valid": bool(weight),
            "valid_frame_count": int(weight),
            "total_frame_count": len(self.records),
            "coverage": float(weight / len(self.records)) if self.records else 0.0,
            "records": self.records,
        }
