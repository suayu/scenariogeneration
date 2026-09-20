"""独立可测试的有界连续控制、统一计分与周期采样规则。"""
import math

DEFAULT_BOUNDS = {
    "llm_anchor": (0.5, 6.0), "scenario_ttc": (0.0, 3.0),
    "inner_lr": (0.03, 0.25), "inner_beta": (0.10, 0.60),
    "n_guide_steps": (2, 4),
}


class DifficultyController:
    """通过连续强度调节各参数，区分自适应与指定目标模式，优先调节攻击权重。"""
    def __init__(self, config=None):
        config = {} if config is None else dict(config)
        # 两种反馈语义由参数选择；off 模式由调用方关闭控制。
        self.mode = config.get("mode", "adaptive")
        if self.mode not in {"off", "adaptive", "target"}:
            raise ValueError("不支持的难度控制模式")
        # 优化器参数保持固定，避免将求解行为变化混入攻击强度。
        self.fixed_optimizer = dict(config.get("fixed_optimizer", {
            "inner_lr": 0.14, "inner_beta": 0.35, "n_guide_steps": 3}))
        self.initial = float(config.get("initial_intensity", 0.5))
        self.gain = float(config.get("gain", 0.5))
        self.max_step = float(config.get("max_step", 0.15))
        self.collision_step = float(config.get("collision_step", 0.15))
        self.bounds = {k: tuple(v) for k, v in dict(config.get("bounds", DEFAULT_BOUNDS)).items()}
        if set(self.bounds) != set(DEFAULT_BOUNDS):
            raise ValueError("必须配置全部五个引导参数的上下界")
        if not 0 <= self.initial <= 1 or not all(math.isfinite(x) and x > 0 for x in (self.gain, self.max_step, self.collision_step)):
            raise ValueError("控制初值、增益与步长配置无效")
        for key, pair in self.bounds.items():
            if len(pair) != 2 or not all(math.isfinite(x) for x in pair) or not 0 <= pair[0] <= pair[1]:
                raise ValueError("参数边界无效: " + key)
            if key != "scenario_ttc" and pair[0] <= 0:
                raise ValueError("参数下界必须为正: " + key)
            if key == "n_guide_steps" and any(int(x) != x for x in pair):
                raise ValueError("引导步数边界必须为整数")

        if set(self.fixed_optimizer) != {"inner_lr", "inner_beta", "n_guide_steps"}:
            raise ValueError("必须提供三个固定优化器参数")
        for key, value in self.fixed_optimizer.items():
            lo, hi = self.bounds[key]
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError("固定优化器参数越界: " + key)
        if int(self.fixed_optimizer["n_guide_steps"]) != self.fixed_optimizer["n_guide_steps"]:
            raise ValueError("固定引导步数必须为整数")

    def parameters(self, intensity):
        if not math.isfinite(intensity):
            raise ValueError("控制强度必须有限")
        u = min(1.0, max(0.0, intensity))
        params = {k: lo + u * (hi - lo) for k, (lo, hi) in self.bounds.items()}
        params.update(self.fixed_optimizer)
        params["n_guide_steps"] = int(math.floor(params["n_guide_steps"] + 0.5))
        return params

    def update(self, before, observed, collision, target, tolerance):
        if not 0 <= target <= 1 or not 0 < tolerance <= 1:
            raise ValueError("目标或容差无效")
        if not math.isfinite(float(before)):
            raise ValueError("控制强度必须有限")
        before = min(1.0, max(0.0, float(before)))
        valid = observed is not None and math.isfinite(observed) and 0 <= observed <= 1
        error = target - observed if valid else None
        if self.mode == "off":
            delta, reason = 0.0, "control_disabled"
        elif collision and self.mode == "adaptive":
            delta, reason = -self.collision_step, "collision_decrease"
        elif not valid:
            delta, reason = 0.0, "missing_observation_hold"
        elif abs(error) <= tolerance:
            delta, reason = 0.0, "within_tolerance_hold"
        else:
            delta = max(-self.max_step, min(self.max_step, self.gain * error))
            reason = "below_target_increase" if delta > 0 else "above_target_decrease"
        after = min(1.0, max(0.0, before + delta))
        return {"mode": self.mode, "before": before, "after": after, "reason": reason,
                "observed": observed if valid else None, "collision": bool(collision),
                "target": target, "signed_error": -error if valid else None,
                "saturated": after == before and delta != 0,
                "parameters_before": self.parameters(before), "parameters_after": self.parameters(after)}


def weighted_difficulty(windows):
    """无有效测量返回 None，防止用零危险伪装缺失数据。"""
    valid = [w for w in windows if w.get("valid", True) and w.get("difficulty") is not None
             and math.isfinite(w["difficulty"]) and 0 <= w["difficulty"] <= 1 and w.get("weight", 0) > 0]
    total = sum(w["weight"] for w in valid)
    return sum(w["difficulty"] * w["weight"] for w in valid) / total if total else None


def ability_score(danger, completed, collision, off_route, progress, alpha=0.5):
    """能力分直接复用目标控制的场景危险度，不再重复计算另一危险分。"""
    if danger is None or not math.isfinite(danger):
        return None
    success = bool(completed and not collision and not off_route)
    partial = 0.0 if success else min(1.0, max(0.0, alpha)) * min(1.0, max(0.0, progress))
    return 100.0 * danger * (float(success) + partial)


def reachability_query_due(query_number, interval=3):
    """按实际请求计数：第三、六、九次均采样，不依赖计划是否攻击。"""
    if interval < 1:
        raise ValueError("可达集查询间隔必须为正整数")
    return query_number > 0 and query_number % interval == 0
