# scenario_generator.py
import numpy as np
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from policies.risk_metrics import AdversarialRiskMetrics

class AdversarialScenarioGenerator:
    """对抗性危险场景生成器：统筹大模型决策、扩散模型细化与评估反馈"""
    def __init__(self, cfg, user_instruction: list, llm_planner=None):
        self.cfg = cfg

        # 大模型规划器由 Simulator 创建，使其生命周期与仿真环境一致。
        self.llm_planner = llm_planner or LLMAdversarialPlanner(
            model_name=self._read_llm_model_name()
        )
        self.attack_duration = 10       # 一次攻击持续时间

        # 大模型调用次数
        self._llm_call_times = 0
        self._max_call_times = 30  # 每个场景最多调用大模型次数
        self._ignore_call_times = 2  # 每个场景忽略調用大模型次数
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

        # 用户指令
        self.user_instruction = user_instruction

        # 其他状态变量
        self.llm_anchors = None
        self.diffusion_trajectory = None
        evaluation_cfg = None
        try:
            evaluation_cfg = self.cfg.sim.evaluation
        except AttributeError:
            pass
        self.risk_metrics = AdversarialRiskMetrics(evaluation_cfg)

    def step(self, env, current_t):
        """在仿真主循环中调用，控制低频决策与轨迹注入"""
        if not self._attacks_enabled:
            return True

        # 攻击帧编号为 0–9；第 10 帧先清除旧动态意图，即使新请求失败也不得继续攻击。
        attack_window_expired = current_t >= (
            self.last_attack_frame + self.attack_duration
        )
        if attack_window_expired and getattr(env, "attack_intent", None) is not None:
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

        # 3. 调用大模型搜索攻击对象、制定策略并生成轨迹锚点。
        attack_plan = self.llm_planner.generate_attack_plan(
            env_state,
            self.user_instruction,
            scene_image=scene_image,
        )
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

    def evaluate_reaction(self, env, info):
        """评估自车反应：保留碰撞/TTC统计，并额外采集二维 EA。"""
        self.risk_metrics.evaluate_ea(env)
        # 记录碰撞
        if info.get('collision', False):
            self.collision_list.append(1.0)
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
        self._llm_call_times = 0
        self._attacks_enabled = bool(getattr(self.llm_planner, "available", True))
        self._consecutive_llm_failures = 0
        self.last_attack_frame = 0
        self.last_query_frame = 0
        self.llm_anchors = None
        self.diffusion_trajectory = None
        self.risk_metrics.reset_episode()

    def _read_max_consecutive_llm_failures(self):
        """读取可配置的 LLM 连续服务失败阈值，并保留无配置时的安全默认值。"""
        try:
            return max(1, int(self.cfg.sim.llm.max_consecutive_failures))
        except (AttributeError, TypeError, ValueError):
            return 3

    def _read_llm_model_name(self):
        """从统一配置读取模型名，便于在免费额度切换时只修改配置。"""
        try:
            return str(self.cfg.sim.llm.model_name)
        except AttributeError:
            return "qwen3-32b"

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
            # 简化判断：若相对速度方向与距离方向一致(即靠近)，则计算TTC
            if rel_speed > 0.1:
                approach_rate = (dx * rel_vx + dy * rel_vy) / dist
                if approach_rate > 0:
                    ttc = dist / approach_rate
                    if ttc < min_ttc:
                        min_ttc = ttc
        return min_ttc

    def _prepare_diffusion_input(self, env_state):
        """辅助函数:将环境状态转换为扩散模型输入格式"""
        # TODO: 处理已有环境数据，并加载历史数据
        return env_state  # 这里假设扩散模型直接接受 env_state，实际可能需要进一步处理
