# scenario_generator.py
import os
import random

import numpy as np
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from policies.risk_metrics import AdversarialRiskMetrics, compute_scenario_danger_score

_STRATEGIES = ("cut_in", "hard_brake", "slow_down", "lane_change", "occlusion", "sudden_acceleration", "others")
_POLICY_ATTACK_MEMORY = {}

_ITERATIVE_STAGES = (
    {"llm_anchor": 4.00, "scenario_ttc": 2.00, "inner_lr": .20, "inner_beta": .50, "n_guide_steps": 2},
    {"llm_anchor": 4.25, "scenario_ttc": 2.15, "inner_lr": .21, "inner_beta": .52, "n_guide_steps": 2},
    {"llm_anchor": 4.50, "scenario_ttc": 2.30, "inner_lr": .22, "inner_beta": .54, "n_guide_steps": 2},
    {"llm_anchor": 4.75, "scenario_ttc": 2.45, "inner_lr": .23, "inner_beta": .56, "n_guide_steps": 2},
)

class AdversarialScenarioGenerator:
    """对抗性危险场景生成器：统筹大模型决策、扩散模型细化与评估反馈"""
    def __init__(self, cfg, user_instruction: list, llm_planner=None):
        self.cfg = cfg

        # 大模型规划器由 Simulator 创建，使其生命周期与仿真环境一致。
        self.llm_planner = llm_planner or LLMAdversarialPlanner(
            model_name=self._read_llm_model_name(),
            model_names=self._read_llm_model_names(),
            provider=self._read_llm_provider(),
        )
        self.attack_duration = 10       # 一次攻击持续时间

        # 大模型调用次数
        self._llm_call_times = 0
        self._max_call_times = 30  # 每个场景最多调用大模型次数
        # 首帧即允许规划；短场景不能在预热阶段结束前错过自然语言攻击机会。
        self._ignore_call_times = 0
        self._attacks_enabled = bool(getattr(self.llm_planner, "available", True))
        # LLM 服务连续失败达到阈值后，仅停用当前场景的高级攻击规划。
        self._max_consecutive_llm_failures = self._read_max_consecutive_llm_failures()
        self._consecutive_llm_failures = 0

        # 控制参数
        self.attack_frequency = 3       # 攻击频率（每多少步调用一次大模型）
        self.last_attack_frame = 0      # 上一次攻击的时间
        self.last_query_frame = 0       # 上一次查询攻击的时间
        self.history_frames = 20        # 历史帧数

        # 评估指标统计
        self.collision_list = []
        self.near_miss_list = []
        self.ttc_list = []
        self._current_episode_min_ttc = float('inf')
        # 单场景攻击统计用于批量评测及代表性视频筛选。
        self._episode_attack_plan_count = 0
        self._episode_attack_active_frames = 0
        self._episode_obstacle_plan_count = 0
        # 保留每次 LLM 主动请求攻击的原因，供实验审计和可视化使用。
        self._episode_attack_reasons = []
        self._episode_attack_difficulties = []

        # 用户指令
        self.user_instruction = user_instruction
        self.selected_user_instruction = None
        # 指令按场景循环：同一场景内的所有 LLM 查询共享同一条自然语言指令。
        self._scene_instruction_index = -1
        self._scene_instruction = None

        # 其他状态变量
        self.llm_anchors = None
        self.diffusion_trajectory = None
        evaluation_cfg = None
        try:
            evaluation_cfg = self.cfg.sim.evaluation
        except AttributeError:
            pass
        self.risk_metrics = AdversarialRiskMetrics(evaluation_cfg)
        self.difficulty_mode, self.target_difficulty, self.difficulty_tolerance = self._read_difficulty_control()
        self.profile_enabled, self.escalation_enabled, self.full_method_enabled = self._read_iterative_switches()
        self._iterative_stage = 0
        self._iterative_window = []
        self._iterative_context = None
        self._profile_previous = None
        self._recent_attack_context = None
        self._recent_attack_expired_step = None

    def step(self, env, current_t):
        """在仿真主循环中调用，控制低频决策与轨迹注入"""
        if not self._attacks_enabled:
            return True

        # 攻击帧编号为 0–9；第 10 帧先清除旧动态意图，即使新请求失败也不得继续攻击。
        attack_window_expired = current_t >= (
            self.last_attack_frame + self.attack_duration
        )
        if attack_window_expired and getattr(env, "attack_intent", None) is not None:
            if self.profile_enabled or self.escalation_enabled or self.difficulty_mode != "off":
                self._finish_iterative_window()
            # 保留刚结束攻击的意图与锚点，供短暂重规划窗口保持策略连续性。
            self._recent_attack_context = dict(env.attack_intent)
            self._recent_attack_expired_step = current_t
            env.clear_attack_intent()
            self.llm_anchors = None
            self.diffusion_trajectory = None

        # 根据上一次攻击和查询时间判断当前帧是否需要调用大模型进行攻击规划。
        if (attack_window_expired and current_t > self.last_query_frame + self.attack_frequency) or current_t == 0:
            print(f"[Adversarial Generator] Planning attack at step {current_t}")
            if self._llm_call_times >= self._max_call_times:
                self._attacks_enabled = False
                print("[对抗场景生成器] 已达到大模型调用上限；停止攻击规划，仿真继续。")
                return True
            if self._llm_call_times < self._ignore_call_times:
                self._llm_call_times += 1
                return True  # 继续仿真
            self.plan_and_inject(env)
            self._llm_call_times += 1
            if self._llm_call_times >= self._max_call_times:
                self._attacks_enabled = False
                print("[对抗场景生成器] 已达到大模型调用上限；停止攻击规划，仿真继续。")
        return True  # 继续仿真

    def plan_and_inject(self, env):
        """执行完整规划流程：搜索路段/对象 -> 制定策略 -> 生成锚点 -> 细化轨迹 -> 注入"""
        # 1. 获取包含自车、路网、其他交通参与者及历史轨迹的环境状态。
        env_state = env.get_state_for_planning()

        # 2. 多模态模式额外生成当前帧图像；纯文本模式不执行图像渲染。
        scene_image = None
        if getattr(self.llm_planner, "use_multimodal", False):
            planning_agent_ids = [agent["id"] for agent in env_state["agents"]]
            scene_image = env.render_llm_scene_image(agent_ids=planning_agent_ids)

        # 3. 每次攻击只抽取一条自然语言需求，避免冲突指令共同约束同一计划。
        selected_instruction = self._sample_user_instruction()
        if selected_instruction:
            print(f"[对抗场景生成器] 本次攻击使用的自然语言指令：{selected_instruction[0]}")

        # 4. 调用大模型搜索攻击对象、制定策略并生成轨迹锚点。
        planner_kwargs = {"scene_image": scene_image}
        # 关闭时不扩展调用契约，保持旧规划器与测试替身完全兼容。
        if self.profile_enabled:
            planner_kwargs["adversarial_context"] = self._iterative_context
        if self._has_recent_attack_context(env.current_step):
            planner_kwargs["previous_attack_context"] = self._recent_attack_context
        if self.full_method_enabled and (self.profile_enabled or self.escalation_enabled) and hasattr(self.llm_planner, "_build_prompt"):
            planner_kwargs["strategy_prior"] = self._strategy_prior()
        attack_plan = self.llm_planner.generate_attack_plan(env_state, selected_instruction, **planner_kwargs)
        # 无论计划是否有效，请求已完成，均记录查询帧以避免逐帧重试。
        self.last_query_frame = env.current_step

        # print("attack_plan:", attack_plan)
        if not attack_plan:
            if getattr(self.llm_planner, "last_request_failed", False):
                self._consecutive_llm_failures += 1
                if self._consecutive_llm_failures >= self._max_consecutive_llm_failures:
                    self._attacks_enabled = False
                    print(
                        "[对抗场景生成器] 大模型服务连续失败 "
                        f"{self._consecutive_llm_failures} 次；本场景停止高级攻击规划，仿真继续。"
                    )
            else:
                # 能收到响应但计划不合规则不视为服务故障，也不触发熔断。
                self._consecutive_llm_failures = 0
            return
        self._consecutive_llm_failures = 0
        if not isinstance(attack_plan, dict):
            print("[对抗场景生成器] 大模型计划不是字典，已跳过本次攻击。")
            return
        target_id = attack_plan.get("attack_target_id")
        anchors = attack_plan.get("anchors", [])
        strategy = attack_plan.get("strategy", "unknown")
        attack_requested = attack_plan.get("attack") is True
        obstacle_plan = attack_plan.get("obstacle_plan", [])
        if attack_requested:
            # 仅记录 LLM 明确发起的动态攻击；执行器后续拒绝时也保留原因。
            self._episode_attack_reasons.append({
                "step": int(env.current_step),
                "reason": str(attack_plan.get("reason", "")).strip(),
                "target_id": target_id,
                "strategy": str(strategy),
                "anchor_count": len(anchors) if isinstance(anchors, list) else 0,
                "requested_obstacle_count": len(obstacle_plan) if isinstance(obstacle_plan, list) else 0,
                "accepted": False,
            })
        # 必须在新障碍物或对抗轨迹进入环境前冻结原始可达集。
        if obstacle_plan or attack_requested:
            self.risk_metrics.begin_attack(env)
        if obstacle_plan:
            # 障碍物可作为独立危险源，因此即使 attack 为 false 也允许执行已校验的摆放计划。
            try:
                created_obstacles = env.apply_obstacle_plan(obstacle_plan, max_groups=2)
            except (TypeError, ValueError, RuntimeError) as error:
                print(f"[对抗场景生成器] 障碍物计划创建失败，已跳过：{error}")
            else:
                self.last_attack_frame = env.current_step
                self._episode_obstacle_plan_count += 1
                print(f"[对抗场景生成器] 已创建 {len(created_obstacles)} 个静态障碍物。")
        if not attack_requested:
            self.llm_anchors = None
            self.diffusion_trajectory = None
            env.clear_attack_intent()
            print("[对抗场景生成器] 大模型决定本次不攻击：", attack_plan.get("reason", ""))
            return

        # 边界检查：确保攻击目标属于当前可见且活跃的交通参与者。
        if target_id == -1 or target_id is None:
            print("[对抗场景生成器] 攻击计划缺少有效目标编号，已跳过本次攻击。")
            return
        self.last_attack_frame = env.current_step
        print("\n攻击计划:", attack_plan)
        print(f"[Adversarial Generator] Attack Plan: Target ID: {target_id}, Anchors: {anchors}, Strategy: {strategy}")

        # 检查 target_id 是否存在于 env_state['agents'] 中的 "id" 字段
        if not any(agent.get("id") == target_id for agent in env_state['agents']):
            print(f"[对抗场景生成器] 攻击目标 {target_id} 当前不可用，已跳过本次攻击。")
            return

        if not anchors or not all(len(anchor) == 2 for anchor in anchors):
            print("[对抗场景生成器] 锚点格式无效，已跳过本次攻击。")
            return
        print(f"[Adversarial Generator] Target ID: {target_id}, Strategy: {strategy}")

        # 保存大模型高级攻击意图，Safe-Sim 将在下一次扩散推理中读取锚点 guidance。
        self.llm_anchors = anchors
        self.diffusion_trajectory = None
        env.set_attack_intent(target_id=target_id, anchors=anchors, strategy=strategy)
        self._episode_attack_plan_count += 1
        # 当前有效计划对应本轮刚记录的最后一个攻击原因。
        if self._episode_attack_reasons:
            self._episode_attack_reasons[-1]["accepted"] = True
        if self.escalation_enabled or self.difficulty_mode != "off":
            self._apply_iterative_stage(env)

    def _sample_user_instruction(self):
        """返回当前场景固定指令；场景切换时再按输入顺序循环下一条。"""
        if not self._scene_instruction:
            self._advance_scene_instruction()
        self.selected_user_instruction = self._scene_instruction
        return [self.selected_user_instruction] if self.selected_user_instruction else []

    def _advance_scene_instruction(self):
        candidates = [str(item).strip() for item in (self.user_instruction or []) if str(item).strip()]
        if not candidates:
            self._scene_instruction = None
            return
        self._scene_instruction_index = (self._scene_instruction_index + 1) % len(candidates)
        self._scene_instruction = candidates[self._scene_instruction_index]

    def evaluate_reaction(self, env, info):
        """评估自车反应：保留碰撞/TTC统计，并额外采集二维 EA。"""
        if getattr(env, "attack_intent", None) is not None:
            # 兼容无 __init__ 的轻量测试替身，正式运行仍由回合重置初始化。
            self._episode_attack_active_frames = getattr(self, "_episode_attack_active_frames", 0) + 1
        self.risk_metrics.evaluate_ea(env)
        if (getattr(self, "profile_enabled", False) or getattr(self, "escalation_enabled", False)) and getattr(env, "attack_intent", None) is not None:
            self._record_full_profile(env, info)
        # 记录碰撞
        if info.get('collision', False):
            self.collision_list.append(1.0)
            # 碰撞是近失事件的最严重结果，因此同时计入近失率。
            self.near_miss_list.append(1.0)
        else:
            self.collision_list.append(0.0)
            # 若未碰撞，计算并记录TTC (Time-to-Collision)
            ttc = self._compute_min_ttc(env)
            self._current_episode_min_ttc = min(self._current_episode_min_ttc, ttc)
            # 记录险情
            if ttc < 1.5:
                self.near_miss_list.append(1.0)
            else:
                self.near_miss_list.append(0.0)

    def get_anchors_and_trajectory(self):
        """返回当前场景的LLM锚点和扩散轨迹"""
        return self.llm_anchors, self.diffusion_trajectory

    def set_diffusion_trajectory(self, trajectory):
        """将当前目标的 Safe-Sim 预测轨迹提供给可视化模块。"""
        self.diffusion_trajectory = trajectory

    def evaluate_reachability(self, env):
        """在 Safe-Sim 已生成本帧联合轨迹后，计算攻击后可达集。"""
        return self.risk_metrics.evaluate_dangerous_reachability(env)

    def reset_episode_stats(self):
        """每个场景开始前重置单回合统计"""
        self._current_episode_min_ttc = float('inf')
        self._episode_attack_plan_count = 0
        self._episode_attack_active_frames = 0
        self._episode_obstacle_plan_count = 0
        self._episode_attack_reasons = []
        self._episode_attack_difficulties = []
        self._advance_scene_instruction()
        self._llm_call_times = 0
        self._attacks_enabled = bool(getattr(self.llm_planner, "available", True))
        self._consecutive_llm_failures = 0
        self.last_attack_frame = 0
        self.last_query_frame = 0
        self.llm_anchors = None
        self.diffusion_trajectory = None
        self.risk_metrics.reset_episode()
        if self.profile_enabled or self.escalation_enabled:
            self._iterative_stage, self._iterative_window, self._iterative_context = 0, [], None
            self._profile_previous = None
        self._recent_attack_context = None
        self._recent_attack_expired_step = None

    def _has_recent_attack_context(self, current_step):
        """攻击结束后 1.5 个查询周期内，向下一次 LLM 查询提供连续性上下文。"""
        if self._recent_attack_context is None or self._recent_attack_expired_step is None:
            return False
        return 0 <= current_step - self._recent_attack_expired_step < 1.5 * self.attack_frequency

    def _read_iterative_switches(self):
        """画像与强度增强分别开关；缺失配置时均保持关闭。"""
        try:
            settings = self.cfg.sim.iterative_adversarial
            profile, escalation = bool(settings.profile_enabled), bool(settings.escalation_enabled)
            return profile, escalation, bool(settings.full_method_enabled) and (profile or escalation)
        except AttributeError:
            return False, False, False

    def _record_full_profile(self, env, info):
        """从状态转移提取与策略实现无关的完整黑盒驾驶响应。"""
        state = np.asarray(env.ego_state, dtype=float)
        speed = float(np.linalg.norm(state[2:4]))
        heading = float(state[4])
        previous = self._profile_previous
        dt = float(getattr(env, "step_time", 0.1))
        accel = 0.0 if previous is None else (speed - previous["speed"]) / max(dt, 1e-6)
        jerk = 0.0 if previous is None else (accel - previous["accel"]) / max(dt, 1e-6)
        yaw_rate = 0.0 if previous is None else np.arctan2(np.sin(heading - previous["heading"]), np.cos(heading - previous["heading"])) / max(dt, 1e-6)
        route_error, route_progress = float("nan"), float("nan")
        try:
            route = np.asarray(env.get_state_for_planning().get("route", []), dtype=float)
            if len(route):
                distances = np.linalg.norm(route[:, :2] - state[:2], axis=1)
                route_error, route_progress = float(np.min(distances)), float(np.argmin(distances))
        except (AttributeError, TypeError, ValueError):
            pass
        progress_loss = 0.0 if previous is None or not np.isfinite(route_progress) else max(0.0, previous.get("route_progress", route_progress) - route_progress)
        item = {"speed": speed, "accel": accel, "jerk": jerk, "yaw_rate": float(yaw_rate),
                "route_error": route_error, "route_progress": route_progress, "progress_loss": progress_loss, "ttc": self._compute_min_ttc(env),
                "collision": bool(info.get("collision", False)),
                "near_miss": self._compute_min_ttc(env) < 1.5}
        self._profile_previous = {"speed": speed, "accel": accel, "heading": heading, "route_progress": route_progress}
        self._iterative_window.append(item)

    def _finish_iterative_window(self):
        """聚合完整自车画像，并仅在增强开关开启时调整下一阶段。"""
        if not self._iterative_window:
            return
        data = self._iterative_window
        min_ttc = min(item["ttc"] for item in data)
        collision = any(item["collision"] for item in data)
        peak_brake = min(item["accel"] for item in data)
        peak_turn = max(abs(item["yaw_rate"]) for item in data)
        peak_jerk = max(abs(item["jerk"]) for item in data)
        route_errors = [item["route_error"] for item in data if np.isfinite(item["route_error"])]
        if peak_brake < -2.0: exploit = "forward_pressure"
        elif peak_turn > .25 or (route_errors and max(route_errors) > 1.5): exploit = "cut_in_or_lateral_conflict"
        else: exploit = "timing_or_gap_pressure"
        self._iterative_context = {"stage": self._iterative_stage, "min_ttc": float(min_ttc),
            "collision": collision, "near_miss_rate": float(np.mean([x["near_miss"] for x in data])),
            "peak_brake_mps2": float(peak_brake), "peak_jerk_mps3": float(peak_jerk),
            "peak_yaw_rate_rps": float(peak_turn), "max_route_error_m": float(max(route_errors)) if route_errors else None,
            "preferred_exploit": exploit}
        window_metrics = {
            "collision_rate": float(collision),
            "near_miss_rate": float(np.mean([x["near_miss"] for x in data])),
            "avg_min_ttc": float(min_ttc),
            **self.risk_metrics.compute_metrics(),
        }
        observed_difficulty = compute_scenario_danger_score(window_metrics, self.cfg.sim.evaluation.composite)
        self._episode_attack_difficulties.append({"difficulty": float(observed_difficulty), "weight": int(len(data)), "stage": int(self._iterative_stage)})
        self._iterative_context["observed_difficulty"] = float(observed_difficulty)
        self._iterative_context["difficulty_mode"] = self.difficulty_mode
        self._iterative_context["target_difficulty"] = float(self.target_difficulty)
        # 难度控制优先保证可行性：碰撞代表过强，不把它作为提高目标分的奖励。
        if self.difficulty_mode == "target":
            if collision or observed_difficulty > self.target_difficulty + self.difficulty_tolerance:
                self._iterative_stage = max(self._iterative_stage - 1, 0)
            elif observed_difficulty < self.target_difficulty - self.difficulty_tolerance:
                self._iterative_stage = min(self._iterative_stage + 1, len(_ITERATIVE_STAGES) - 1)
        elif self.difficulty_mode == "adaptive":
            if collision or min_ttc < .5:
                self._iterative_stage = max(self._iterative_stage - 1, 0)
            elif min_ttc > 2.5:
                self._iterative_stage = min(self._iterative_stage + 1, len(_ITERATIVE_STAGES) - 1)
        elif self.full_method_enabled:
            state = "overstrong_or_unsolvable" if collision or min_ttc < .5 else ("risk_insufficient" if min_ttc > 2.5 else "high_risk_solvable")
            self._last_dual_objective_state = state
            if self._active_attack_strategy:
                item = self._policy_memory()[self._active_attack_strategy]
                item["trials"] += 1; item["reward"] += {"risk_insufficient": .25, "high_risk_solvable": 1.0, "overstrong_or_unsolvable": -0.20}[state]; item["solvable"] += int(state != "overstrong_or_unsolvable")
            if self.escalation_enabled and state == "risk_insufficient": self._iterative_stage = min(self._iterative_stage + 1, len(_ITERATIVE_STAGES) - 1)
            elif self.escalation_enabled and state == "overstrong_or_unsolvable":
                # 单次过强仅回退一级；保留攻击意图与历史策略，下一窗口继续施压。
                self._iterative_stage = max(self._iterative_stage - 1, 0)
        elif self.escalation_enabled and not collision:
            self._iterative_stage = min(self._iterative_stage + 1, len(_ITERATIVE_STAGES) - 1)
        self._iterative_window = []

    def _policy_memory(self):
        key = str(getattr(getattr(self.cfg, "sim", None), "policy", "unknown"))
        return _POLICY_ATTACK_MEMORY.setdefault(key, {x: {"trials": 0, "reward": 0.0, "solvable": 0} for x in _STRATEGIES})

    def _strategy_prior(self):
        table = self._policy_memory()
        return sorted(({"strategy": k, "trials": v["trials"], "mean_reward": round(v["reward"] / v["trials"], 3) if v["trials"] else 0.0, "solvable_rate": round(v["solvable"] / v["trials"], 3) if v["trials"] else 0.0} for k, v in table.items()), key=lambda x: x["mean_reward"], reverse=True)

    def _apply_iterative_stage(self, env):
        """分层控制：语义层来自策略先验，几何/扩散层按画像更新。"""
        controller = getattr(env, "diffusion_controller", None)
        if controller is None or not hasattr(controller, "update_iterative_guidance"):
            return
        stage = _ITERATIVE_STAGES[self._iterative_stage]
        exploit = (self._iterative_context or {}).get("preferred_exploit")
        ttc, anchor = stage["scenario_ttc"], stage["llm_anchor"]
        if exploit == "forward_pressure": ttc += .10
        elif exploit: anchor += .10
        controller.update_iterative_guidance({k: stage[k] for k in ("inner_lr", "inner_beta", "n_guide_steps")}, [1.5, 1.0, ttc, anchor])

    def _read_max_consecutive_llm_failures(self):
        """读取可配置的 LLM 连续服务失败阈值，并保留无配置时的安全默认值。"""
        try:
            return max(1, int(self.cfg.sim.llm.max_consecutive_failures))
        except (AttributeError, TypeError, ValueError):
            return 3

    def _read_difficulty_control(self):
        """读取唯一的难度迭代模式，并对目标与容差做安全范围约束。"""
        try:
            config = self.cfg.sim.evaluation.difficulty_control
            mode = str(config.mode).strip().lower()
            if mode not in {"off", "adaptive", "target"}:
                raise ValueError("difficulty_control.mode must be off, adaptive, or target")
            target = float(config.target_difficulty)
            tolerance = float(config.tolerance)
            if not 0.0 <= target <= 1.0 or not 0.0 < tolerance <= 1.0:
                raise ValueError("difficulty target/tolerance must be within [0, 1]")
            return mode, target, tolerance
        except AttributeError:
            return "off", 0.75, 0.10

    def episode_difficulty(self):
        """场景难度为各攻击窗口难度按有效帧数的加权平均。"""
        if not self._episode_attack_difficulties:
            return 0.0
        weights = np.asarray([item["weight"] for item in self._episode_attack_difficulties], dtype=float)
        values = np.asarray([item["difficulty"] for item in self._episode_attack_difficulties], dtype=float)
        return float(np.average(values, weights=weights))

    def _read_llm_model_name(self):
        """环境变量优先读取模型名，便于在免费额度切换时无需改 YAML。"""
        environment_model = os.getenv("LLM_MODEL_NAME")
        if environment_model:
            return environment_model
        try:
            return str(self.cfg.sim.llm.model_name)
        except AttributeError:
            return "qwen3-32b"

    def _read_llm_model_names(self):
        """读取固定候选池，避免在不同提供方之间隐式切换模型。"""
        environment_models = os.getenv("LLM_MODEL_NAMES")
        if environment_models:
            return [name.strip() for name in environment_models.split(",") if name.strip()]
        if os.getenv("LLM_MODEL_NAME"):
            return [os.environ["LLM_MODEL_NAME"]]
        try:
            return list(self.cfg.sim.llm.model_names)
        except (AttributeError, TypeError):
            return None

    def _read_llm_provider(self):
        """环境变量优先，便于同一配置文件下安全切换提供方。"""
        configured_provider = os.getenv("LLM_PROVIDER")
        if configured_provider:
            return configured_provider
        try:
            return str(self.cfg.sim.llm.provider)
        except AttributeError:
            return "dashscope"

    def finalize_episode_stats(self):
        """场景结束后记录本回合最小TTC"""
        if self._current_episode_min_ttc != float('inf'):
            self.ttc_list.append(self._current_episode_min_ttc)

    def compute_final_metrics(self):
        """计算并返回所有场景的最终评估指标"""
        original_metrics = {
            'collision_rate': np.mean(self.collision_list) if self.collision_list else 0.0,
            'near_miss_rate': np.mean(self.near_miss_list) if self.near_miss_list else 0.0,
            'avg_min_ttc': np.mean(self.ttc_list) if self.ttc_list else float('inf')
        }
        return {**original_metrics, **self.risk_metrics.compute_metrics()}

    def _compute_min_ttc(self, env):
        """辅助函数:计算自车与最近活跃他车的最小TTC (简化版)"""
        ego_state = env.ego_state
        agents = env.data_dict['agent'][-1]
        active_mask = env.agent_active
        min_ttc = float('inf')
        ego_x, ego_y, ego_vx, ego_vy = ego_state[0], ego_state[1], ego_state[2], ego_state[3]
        ego_speed = np.sqrt(ego_vx**2 + ego_vy**2)
        if ego_speed < 0.1: # 自车静止，跳过TTC计算
            return min_ttc
        for idx, agent in enumerate(agents):
            if not active_mask[idx]:
                continue
            ag_x, ag_y, ag_vx, ag_vy = agent[0], agent[1], agent[2], agent[3]
            dx, dy = ag_x - ego_x, ag_y - ego_y
            rel_vx, rel_vy = ag_vx - ego_vx, ag_vy - ego_vy
            # 计算相对距离与相对速度
            dist = np.sqrt(dx**2 + dy**2)
            rel_speed = np.sqrt(rel_vx**2 + rel_vy**2)
            # 距离导数为 r·v_rel/|r|；其为负时距离缩短，闭合速度取相反数。
            if rel_speed > 0.1 and dist > 1e-6:
                closing_speed = -(dx * rel_vx + dy * rel_vy) / dist
                if closing_speed > 0:
                    ttc = dist / closing_speed
                    if ttc < min_ttc:
                        min_ttc = ttc
        return min_ttc

    def _prepare_diffusion_input(self, env_state):
        """辅助函数:将环境状态转换为扩散模型输入格式"""
        # TODO: 处理已有环境数据，并加载历史数据
        return env_state  # 这里假设扩散模型直接接受 env_state，实际可能需要进一步处理
