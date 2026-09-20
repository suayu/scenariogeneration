# scenario_generator.py
import os
import random
from types import SimpleNamespace

import numpy as np
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from policies.risk_metrics import AdversarialRiskMetrics, compute_scenario_danger_score
from policies.difficulty_control import DifficultyController, weighted_difficulty, reachability_query_due
from policies.collision_ttc import minimum_obb_ttc

_STRATEGIES = ("cut_in", "hard_brake", "slow_down", "lane_change", "occlusion", "sudden_acceleration", "others")
_POLICY_ATTACK_MEMORY = {}

class AdversarialScenarioGenerator:
    """对抗性危险场景生成器：统筹大模型决策、扩散模型细化与评估反馈"""
    def __init__(self, cfg, user_instruction: list, llm_planner=None):
        # 同时接受根配置与 sim 子配置，内部始终使用根层级。
        self.cfg = cfg if hasattr(cfg, "sim") else SimpleNamespace(sim=cfg)

        # 大模型规划器由 Simulator 创建，使其生命周期与仿真环境一致。
        self.llm_planner = llm_planner or LLMAdversarialPlanner(
            model_name=self._read_llm_model_name(),
            model_names=self._read_llm_model_names(),
            provider=self._read_llm_provider(),
        )
        self.attack_duration = 10       # 一次攻击持续时间

        # 大模型调用次数
        self._llm_call_times = 0
        self._max_call_times = int(getattr(getattr(getattr(self.cfg, 'sim', None), 'llm', None), 'max_planning_calls', 30))
        if not 1 <= self._max_call_times <= 30:
            raise ValueError('每场景规划调用上限必须为 1–30')
        # 首帧即允许规划；短场景不能在预热阶段结束前错过自然语言攻击机会。
        self._ignore_call_times = 0
        self._attacks_enabled = bool(getattr(self.llm_planner, "available", True))
        # LLM 服务连续失败达到阈值后，仅停用当前场景的高级攻击规划。
        self._max_consecutive_llm_failures = self._read_max_consecutive_llm_failures()
        self._consecutive_llm_failures = 0

        # 控制参数
        llm_cfg = getattr(self.cfg.sim, 'llm', None)
        # 默认保持首帧规划与原有查询间隔；实验可统一延后首次规划，以观察画像预热后的交互。
        self.min_planning_step = int(getattr(llm_cfg, 'min_planning_step', 0))
        self.attack_frequency = int(getattr(llm_cfg, 'planning_interval_frames', 3))
        if self.min_planning_step < 0 or self.attack_frequency < 1:
            raise ValueError('最早规划帧必须非负，查询间隔必须至少一帧')
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
        self._episode_planner_service_failures = 0
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
        # 兼容无配置的最小测试及关闭攻击的入口。
        difficulty_cfg = getattr(evaluation_cfg, "difficulty_control", None)
        controller_config = dict(getattr(difficulty_cfg, "controller", None) or {})
        controller_config["mode"] = self.difficulty_mode
        self.difficulty_controller = DifficultyController(controller_config)
        self._iterative_stage = self.difficulty_controller.initial
        self._iterative_window = []
        self._iterative_context = None
        self._profile_previous = None
        self._recent_attack_context = None
        self._recent_attack_expired_step = None
        self._iterative_window_metric_offsets = None
        self._scene_replay_stage = self.difficulty_controller.initial
        self._reachability_query_pending = False
        self._observation_started = False
        self._active_attack_strategy = None

        # 单一画像开关启用完整链路，不依赖旧的 escalation/full_method 开关。
        # 默认单 Agent 不创建协作控制器，完整保留既有调用分支。
        self.multiagent_planner = None
        multi_cfg = getattr(getattr(self.cfg.sim, "llm", None), "multiagent", None)
        if str(getattr(multi_cfg, "mode", "single")) != "single":
            from policies.multiagent_planner import MultiAgentPlanner
            self.multiagent_planner = MultiAgentPlanner(multi_cfg)
        self.profile_pipeline = None
        if self.profile_enabled:
            from policies.ego_profile import ProfileAttackPipeline
            self.profile_pipeline = ProfileAttackPipeline(self.cfg.sim)

    def step(self, env, current_t):
        """在仿真主循环中调用，控制低频决策与轨迹注入"""
        # 攻击帧编号为 0–9；第 10 帧先清除旧动态意图，即使新请求失败也不得继续攻击。
        attack_window_expired = current_t >= (
            self.last_attack_frame + self.attack_duration
        )
        if attack_window_expired and getattr(env, "attack_intent", None) is not None:
            # 观测窗口与控制开关独立：off 模式仍需结算真实危险度。
            self._finish_iterative_window()
            # 保留刚结束攻击的意图与锚点，供短暂重规划窗口保持策略连续性。
            self._recent_attack_context = dict(env.attack_intent)
            self._recent_attack_expired_step = current_t
            env.clear_attack_intent()
            self.llm_anchors = None
            self.diffusion_trajectory = None

        # 服务熔断或调用耗尽仅停止新查询，旧攻击仍须按期限结束并结算。
        if not self._attacks_enabled:
            return True

        if current_t < self.min_planning_step:
            return True

        # 根据上一次攻击和查询时间判断当前帧是否需要调用大模型进行攻击规划。
        if (attack_window_expired and current_t > self.last_query_frame + self.attack_frequency) or current_t == self.min_planning_step:
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
        # 新查询前结算上一观察窗口；不把未攻击时的有效测量丢弃。
        self._finish_iterative_window()
        self._active_attack_strategy = None
        self._iterative_window_metric_offsets = {
            "ea": len(self.risk_metrics.ea_values), "reachability": len(self.risk_metrics.reachability_events),
        }
        self._observation_started = True
        difficulty_cfg = getattr(getattr(self.cfg.sim, "evaluation", None), "difficulty_control", None)
        interval = int(getattr(difficulty_cfg, "reachability_every_queries", 3))
        self._reachability_query_pending = reachability_query_due(self._llm_call_times + 1, interval)
        if self._reachability_query_pending:
            # 查询结果未知时先冻结同一时刻基线，后续即使不攻击也执行测量。
            self.risk_metrics.begin_attack(env)
        env_state = env.get_state_for_planning()
        # 查询输入按尝试独立留存，区分真实空场景与模型误读，不触及提供方配置。
        query_trace = getattr(self, 'query_trace', None)
        if query_trace is not None:
            query_trace.append({'kind': 'llm_input', 'step': int(env.current_step), 'state': env_state})

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
        if getattr(self.llm_planner, "provider", None) == "codex":
            planner_kwargs["difficulty_context"] = {"mode": self.difficulty_mode, "target": self.target_difficulty, "tolerance": self.difficulty_tolerance}
        # 关闭时不扩展调用契约，保持旧规划器与测试替身完全兼容。
        if self.profile_enabled:
            planner_kwargs["adversarial_context"] = self._iterative_context
        if self._has_recent_attack_context(env.current_step):
            planner_kwargs["previous_attack_context"] = self._recent_attack_context
        if self.full_method_enabled and (self.profile_enabled or self.escalation_enabled) and hasattr(self.llm_planner, "_build_prompt"):
            planner_kwargs["strategy_prior"] = self._strategy_prior()
        # 在现有 LLM 调用边界计时；单 Agent 与协作模式使用相同墙钟口径。
        import time
        planner_started = time.monotonic()
        if self.profile_enabled:
            from policies.profile_planner import rank_profile_candidates
            context = self.profile_pipeline.context(env, self._iterative_stage)
            # 只给规划器当前可见的目标，拒绝全局活跃但不在请求中的车辆。
            visible_ids = {a['id'] for a in env_state['agents']}
            context['candidates'] = [c for c in context['candidates'] if c['target_id'] in visible_ids]
            if getattr(self.llm_planner,'use_obstacles',False):
                from policies.profile_obstacles import build_obstacle_candidates
                context['obstacle_candidates']=build_obstacle_candidates(env,self.llm_planner,context['candidates'],self.profile_pipeline.builder)
            if query_trace is not None:
                query_trace.append({'kind':'profile_ranking_input','step':int(env.current_step),'context':context})
            if getattr(self, "multiagent_planner", None) is None:
                attack_plan = rank_profile_candidates(self.llm_planner,env_state,selected_instruction,context,scene_image)
            else:
                # 画像身份和场景条件仅扩展协作输入，不改变原单 Agent 提示。
                context["policy_id"] = self.profile_pipeline.policy_id
                context["scene_id"] = self.profile_pipeline.pending["scene_id"]
                context["scene_conditions"] = self.profile_pipeline.pending["scene_conditions"]
                attack_plan = self.multiagent_planner.run(self.llm_planner, env_state, selected_instruction,
                                                         context=context, scene_image=scene_image)
            self.profile_pipeline.planned(attack_plan, bool(getattr(self.llm_planner,'last_request_failed',False)))
        else:
            if getattr(self, "multiagent_planner", None) is None:
                attack_plan = self.llm_planner.generate_attack_plan(env_state, selected_instruction, **planner_kwargs)
            else:
                memory_context = None
                if self.multiagent_planner.mode == 'parallel_memory':
                    from policies.ego_profile import policy_id_for_sim, scene_conditions
                    # 无画像时只计算检索身份；不启动画像候选排序或改变原规划器输入。
                    memory_context = dict(policy_id=policy_id_for_sim(self.cfg.sim),
                                          scene_id=str(getattr(env,'current_scene_id','unknown')),
                                          scene_conditions=scene_conditions(env))
                attack_plan = self.multiagent_planner.run(self.llm_planner, env_state, selected_instruction,
                                                         planner_kwargs=planner_kwargs,
                                                         memory_context=memory_context)
        planner_wall_seconds = time.monotonic() - planner_started
        if query_trace is not None:
            query_trace.append({'kind': 'llm_output', 'step': int(env.current_step),
                                'instruction': selected_instruction, 'validated_plan': attack_plan,
                                'request_failed': bool(getattr(self.llm_planner, 'last_request_failed', False)),
                                'planner_trace': getattr(self.llm_planner, 'last_trace', None),
                                'planner_wall_seconds': planner_wall_seconds})
        # 无论计划是否有效，请求已完成，均记录查询帧以避免逐帧重试。
        self.last_query_frame = env.current_step

        # print("attack_plan:", attack_plan)
        if not attack_plan:
            if getattr(self.llm_planner, "last_request_failed", False):
                self._episode_planner_service_failures += 1
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
        if (obstacle_plan or attack_requested) and not self._reachability_query_pending:
            self.risk_metrics.begin_attack(env)
        if obstacle_plan:
            # 障碍物可作为独立危险源，因此即使 attack 为 false 也允许执行已校验的摆放计划。
            try:
                previous_obstacle_ids={str(o['id']) for o in env.get_static_obstacles()}
                created_obstacles = env.apply_obstacle_plan(obstacle_plan, max_groups=2)
            except (TypeError, ValueError, RuntimeError) as error:
                print(f"[对抗场景生成器] 障碍物计划创建失败，已跳过：{error}")
            else:
                if self.profile_enabled and self.profile_pipeline.pending:
                    self.profile_pipeline.pending['obstacle_created_ids']=[str(o['id']) for o in env.get_static_obstacles() if str(o['id']) not in previous_obstacle_ids]
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
        # 请求编号贯穿意图、扩散预测和真实执行，不将锚点当作连续控制。
        self.attack_duration = int(attack_plan.get("duration", 10))
        env.attack_intent["request_id"] = attack_plan.get("request_id")
        env.attack_intent["duration"] = self.attack_duration
        if self.profile_enabled and self.profile_pipeline.pending:
            self.profile_pipeline.pending['accepted'] = True
            env.attack_intent['profile_guided'] = True
            env.attack_intent['ttc_range_s'] = list(self.profile_pipeline.builder.ttc_range)
        if query_trace is not None:
            query_trace.append({"kind": "plan_accepted", "step": int(env.current_step), "request_id": attack_plan.get("request_id"), "target_id": target_id})
        self._active_attack_strategy = strategy if strategy in _STRATEGIES else "others"
        self._episode_attack_plan_count += 1
        # 当前有效计划对应本轮刚记录的最后一个攻击原因。
        if self._episode_attack_reasons:
            self._episode_attack_reasons[-1]["accepted"] = True
        if self.difficulty_mode != "off":
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
        # 所有模式都记录窗口观测；仅难度控制器按模式决定是否更新强度。
        if getattr(self, "_observation_started", False):
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
        result = self.risk_metrics.evaluate_dangerous_reachability(env)
        self._reachability_query_pending = False
        return result

    def reset_episode_stats(self, advance_instruction=True):
        """重置单回合统计；同一场景重放时保持自然语言指令不变。"""
        self._current_episode_min_ttc = float('inf')
        self._episode_attack_plan_count = 0
        self._episode_planner_service_failures = 0
        self._episode_attack_active_frames = 0
        self._episode_obstacle_plan_count = 0
        self._episode_attack_reasons = []
        self._episode_attack_difficulties = []
        if advance_instruction:
            self._advance_scene_instruction()
        self._llm_call_times = 0
        self._attacks_enabled = bool(getattr(self.llm_planner, "available", True))
        self._current_episode_min_ttc = float('inf')
        self._active_attack_strategy = None
        self._consecutive_llm_failures = 0
        self.last_attack_frame = 0
        self.last_query_frame = 0
        self.llm_anchors = None
        self.diffusion_trajectory = None
        self.risk_metrics.reset_episode()
        if self.profile_enabled or self.escalation_enabled or self.difficulty_mode != "off":
            self._iterative_stage = float(self._scene_replay_stage)
            self._iterative_window, self._iterative_context = [], None
            self._profile_previous = None
        self._iterative_window_metric_offsets = None
        self._reachability_query_pending = False
        self._observation_started = False
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
            settings = self.cfg.sim.traffic_model.iterative_adversarial
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
        dt = float(getattr(env, "dt", 0.1))
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
            if getattr(self, 'profile_pipeline', None) is not None:
                self.profile_pipeline.finish_window()
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
        offsets = self._iterative_window_metric_offsets or {
            "ea": len(self.risk_metrics.ea_values),
            "reachability": len(self.risk_metrics.reachability_events),
        }
        ea_values = self.risk_metrics.ea_values[offsets["ea"]:]
        reachability_events = self.risk_metrics.reachability_events[
            offsets["reachability"]:
        ]
        finite_events = [
            event
            for event in reachability_events
            if np.isfinite(event["difficulty"])
        ]
        window_metrics = {
            "collision_rate": float(collision),
            "near_miss_rate": float(np.mean([x["near_miss"] for x in data])),
            "avg_min_ttc": float(min_ttc),
            "max_evasive_acceleration_mps2": float(max(ea_values)) if ea_values else 0.0,
            "mean_evasive_acceleration_mps2": float(np.mean(ea_values)) if ea_values else 0.0,
            "reachability_event_count": len(reachability_events),
        }
        if finite_events:
            hardest = max(finite_events, key=lambda event: event["difficulty"])
            window_metrics.update({
                "reachability_difficulty": float(hardest["difficulty"]),
                "mean_reachability_difficulty": float(np.mean([
                    event["difficulty"] for event in finite_events
                ])),
                "dangerous_scene_solvable": float(hardest["dangerous_solvable"]),
            })
        # 完整窗口必须有 EA 和可达集测量；不能以缺失项的默认零值冒充低风险。
        measurement_valid = bool(ea_values and finite_events) and all(np.isfinite(x) for x in ea_values)
        observed_difficulty = (float(compute_scenario_danger_score(window_metrics, self.cfg.sim.evaluation.composite))
                               if measurement_valid else None)
        measurement_valid = measurement_valid and np.isfinite(observed_difficulty) and 0 <= observed_difficulty <= 1
        if not measurement_valid:
            observed_difficulty = None
        # 每条窗口记录保留测量来源，允许审计未发起攻击时的真实观察。
        window_record = {"difficulty": observed_difficulty, "weight": int(len(data)),
                         "intensity": float(self._iterative_stage), "valid": bool(measurement_valid),
                         "measurement_status": "valid" if measurement_valid else "missing_or_invalid_ea_reachability",
                         "metrics": window_metrics, "ea_count": len(ea_values),
                         "reachability_count": len(reachability_events),
                         "query_number": int(self._llm_call_times)}
        self._episode_attack_difficulties.append(window_record)
        if getattr(self, 'profile_pipeline', None) is not None:
            self.profile_pipeline.finish_window(window_record, reachability_events)
        self._iterative_context["observed_difficulty"] = observed_difficulty
        self._iterative_context["difficulty_mode"] = self.difficulty_mode
        self._iterative_context["target_difficulty"] = float(self.target_difficulty)
        # adaptive 碰撞降强度；target 仅按有效危险度误差更新。
        if self.difficulty_mode != "off":
            feedback = self.difficulty_controller.update(
                self._iterative_stage, observed_difficulty, collision,
                self.target_difficulty, self.difficulty_tolerance,
            )
            self._iterative_stage = feedback["after"]
            window_record["control"] = feedback
        # 记忆收益更新独立于目标控制分支，避免开启 target 后被 elif 跳过。
        if self.full_method_enabled and not self.profile_enabled:
            state = "overstrong_or_unsolvable" if collision or min_ttc < .5 else ("risk_insufficient" if min_ttc > 2.5 else "high_risk_solvable")
            self._last_dual_objective_state = state
            if self._active_attack_strategy:
                item = self._policy_memory()[self._active_attack_strategy]
                item["trials"] += 1; item["reward"] += {"risk_insufficient": .25, "high_risk_solvable": 1.0, "overstrong_or_unsolvable": -0.20}[state]; item["solvable"] += int(state != "overstrong_or_unsolvable")
        self._iterative_window = []
        self._iterative_window_metric_offsets = None

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
        stage = self.difficulty_controller.parameters(self._iterative_stage)
        ttc, anchor = stage["scenario_ttc"], stage["llm_anchor"]
        controller.update_iterative_guidance({k: stage[k] for k in ("inner_lr", "inner_beta", "n_guide_steps")}, [1.5, 1.0, ttc, anchor])

    def apply_attack_route_weight(self, env, attack_active):
        """仅显式启用时切换道路软引导；不修改求解器参数与道路硬门槛。"""
        traffic = getattr(self.cfg.sim, "traffic_model", None)
        configured = getattr(traffic, "attack_route_weight", None)
        margin_configured = getattr(traffic, "attack_route_lane_margin", None)
        if configured is None and margin_configured is None:
            return None
        attack_weight = float(configured) if configured is not None else None
        if attack_weight is not None and (not np.isfinite(attack_weight) or attack_weight < 0):
            raise ValueError("攻击窗口道路引导权重必须是非负有限数")
        controller = getattr(env, "diffusion_controller", None)
        functions = list(getattr(controller, "active_guidance_functions", ()))
        if not hasattr(controller, "update_iterative_guidance") or functions.count("route") != 1:
            raise ValueError("攻击窗口道路引导需要 llm_joint 中唯一的 route 函数")
        route_index = functions.index("route")
        net = controller.policy_replicas[0][1].nets["policy"]
        weights = net.Loss_Calculater.weights.detach().cpu().tolist()
        if len(weights) != len(functions):
            raise ValueError("道路引导函数与权重数量不一致")
        if not hasattr(self, "_attack_route_base_weight"):
            self._attack_route_base_weight = float(weights[route_index])
        route_losses = []
        if margin_configured is not None:
            for _, policy in controller.policy_replicas:
                root = policy.nets["policy"].Loss_Calculater
                route_loss = controller._find_loss_calculator(root, "route")
                if route_loss is None or not hasattr(route_loss, "lane_margin"):
                    raise ValueError("未找到 route loss，不能调整中心线软余量")
                route_losses.append(route_loss)
            if not hasattr(self, "_attack_route_base_margin"):
                self._attack_route_base_margin = float(route_losses[0].lane_margin)
            attack_margin = float(margin_configured)
            if not np.isfinite(attack_margin) or not 0 < attack_margin <= self._attack_route_base_margin:
                raise ValueError("攻击窗口道路软余量必须为正且不大于原值")
        applied = (attack_weight if attack_weight is not None else self._attack_route_base_weight) if attack_active else self._attack_route_base_weight
        weights[route_index] = applied
        if not controller.update_iterative_guidance({}, weights):
            raise ValueError("攻击窗口道路引导更新失败")
        margin = (attack_margin if attack_active else self._attack_route_base_margin) if route_losses else None
        for route_loss in route_losses:
            route_loss.lane_margin = margin
            route_loss.non_linear_margin = margin + 0.5
        state = (applied, margin)
        changed = state != getattr(self, "_last_attack_route_guidance", None)
        self._last_attack_route_guidance = state
        return dict(changed=changed, active=bool(attack_active), route_weight=applied,
                    route_lane_margin=margin)

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
            return "off", 0.50, 0.10

    def episode_difficulty(self):
        """场景难度为各攻击窗口难度按有效帧数的加权平均。"""
        return weighted_difficulty(self._episode_attack_difficulties)

    def finalize_episode_difficulty(self):
        """场景提前终止时也结算最后一个攻击窗口，不额外创建校准攻击。"""
        if self._iterative_window:
            self._finish_iterative_window()
        return self.episode_difficulty()

    def apply_scene_replay_feedback(self, observed_difficulty, collision=False):
        """根据场景最终难度调整下一次完整重放的初始引导阶段。"""
        if self.difficulty_mode != "target":
            return self._scene_replay_stage
        self._last_replay_feedback = self.difficulty_controller.update(
            self._scene_replay_stage, observed_difficulty, collision,
            self.target_difficulty, self.difficulty_tolerance,
        )
        self._scene_replay_stage = self._last_replay_feedback["after"]
        return self._scene_replay_stage

    def reset_scene_replay_control(self):
        """进入新场景前清除上一场景的重放强度偏置。"""
        self._scene_replay_stage = self.difficulty_controller.initial

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
        """使用 OBB 时间区间相交计算首次接触，自车静止时也检查来车。"""
        return minimum_obb_ttc(env.ego_state,env.data_dict['agent'][-1],env.agent_active)

    def _prepare_diffusion_input(self, env_state):
        """辅助函数:将环境状态转换为扩散模型输入格式"""
        # TODO: 处理已有环境数据，并加载历史数据
        return env_state  # 这里假设扩散模型直接接受 env_state，实际可能需要进一步处理
