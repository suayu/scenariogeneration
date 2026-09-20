"""按场景重采样的初始/最终难度统计，保留缺测与重放失败。"""
import math
import random
import statistics


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def quantile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    x = (len(ordered) - 1) * q
    lo = int(x)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered)-1)]-ordered[lo])*(x-lo)


def summarize_attempts(episodes, samples=2000, seed=42):
    rng = random.Random(seed)
    def describe(records):
        def statistics_for(rows):
            errors, hits = [], []
            for r in rows:
                d = r.get("attack_difficulty")
                valid = d is not None and math.isfinite(d)
                e = d-r["difficulty_target"] if valid else None
                hits.append(float(valid and not r.get("generation_failure") and abs(e) <= r["difficulty_tolerance"]))
                if valid:
                    errors.append(e)
            ae = list(map(abs, errors))
            return {"hit_rate": statistics.mean(hits) if hits else None,
                    "mean_absolute_error": statistics.mean(ae) if ae else None,
                    "p50_absolute_error": quantile(ae,.5), "p95_absolute_error": quantile(ae,.95),
                    "signed_bias": statistics.mean(errors) if errors else None,
                    "valid_count": len(errors), "missing_count": len(rows)-len(errors)}
        result = statistics_for(records)
        # 缺测场景也参与重采样，避免只保留成功场景抬高命中率。
        draws = [statistics_for(rng.choices(records,k=len(records))) for _ in range(samples)] if records else []
        result["bootstrap95"] = {k:[quantile(v,.025),quantile(v,.975)] for k in
            ("hit_rate","mean_absolute_error","p50_absolute_error","p95_absolute_error","signed_bias")
            for v in [[d[k] for d in draws if d[k] is not None]]}
        return result
    first = [e.get("attempts", [e])[0] for e in episodes]
    final = [e.get("attempts", [e])[-1] for e in episodes]
    # 重放代价与不可控率同样以场景为重采样单位。
    def outcome(rows):
        return {"mean_replays": statistics.mean([e.get("replay_count",0) for e in rows]),
                "uncontrollable_rate": statistics.mean([e.get("difficulty_control_status") == "uncontrollable" for e in rows])}
    outcome_draws = [outcome(rng.choices(episodes,k=len(episodes))) for _ in range(samples)] if episodes else []
    return {"initial": describe(first), "final": describe(final),
            "replays": [e.get("replay_count",0) for e in episodes],
            "uncontrollable_rate": statistics.mean([e.get("difficulty_control_status") == "uncontrollable" for e in episodes]) if episodes else None,
            "mean_replays": statistics.mean([e.get("replay_count",0) for e in episodes]) if episodes else None,
            "outcome_bootstrap95": {k:[quantile([d[k] for d in outcome_draws],.025),quantile([d[k] for d in outcome_draws],.975)]
                                    for k in ("mean_replays","uncontrollable_rate")},
            "bootstrap_unit": "scene", "bootstrap_seed": seed, "bootstrap_samples": samples}
