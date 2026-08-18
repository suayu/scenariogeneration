# scenario_generator.py
import numpy as np
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from policies.diffusion_adversarial_planner import DiffusionTrajectoryRefiner

class AdversarialScenarioGenerator:
    """对抗性危险场景生成器：统筹大模型决策、扩散模型细化与评估反馈"""
    def __init__(self, cfg, user_instruction: list):
        self.cfg = cfg

        # 初始化大模型与扩散模型接口
        self.llm_planner = LLMAdversarialPlanner(model_name="qwen3.5-35b-a3b")
        self.attack_duration = 10       # 一次攻击持续时间
        self.diffusion_refiner = DiffusionTrajectoryRefiner(num_steps=self.attack_duration)

        # 大模型调用次数
        self._llm_call_times = 0
        self._max_call_times = 30  # 每个场景最多调用大模型次数
        self._ignore_call_times = 2  # 每个场景忽略調用大模型次数

        # 控制参数
        self.attack_frequency = 3       # 攻击频率（每多少步调用一次大模型）
        self.last_attack_frame = 0   # 上一次攻击的时间
        self.last_query_frame = 0    # 上一次查询攻击的时间
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

    def step(self, env, current_t):
        """在仿真主循环中调用，控制低频决策与轨迹注入"""
        # 根据上一次攻击和查询时间判断当前帧是否需要调用大模型进行攻击规划
        if ( current_t > self.last_attack_frame + self.attack_duration and current_t > self.last_query_frame + self.attack_frequency ) or current_t == 0:
            print(f"[Adversarial Generator] Planning attack at step {current_t}")
            if self._llm_call_times < self._ignore_call_times:
                self._llm_call_times += 1
                return True  # 继续仿真
            self.plan_and_inject(env)
            self._llm_call_times += 1
            if self._llm_call_times >= self._max_call_times:
                print("[Adversarial Generator] Reached max LLM calls for this episode.")
                return False  # 停止进一步攻击
        return True  # 继续仿真

    def plan_and_inject(self, env):
        """执行完整规划流程：搜索路段/对象 -> 制定策略 -> 生成锚点 -> 细化轨迹 -> 注入"""
        # 1. 获取当前环境状态 (包含自车、路网、他车状态)
        env_state = env.get_state_for_planning()

        # 2. 调用大模型：搜索攻击对象、制定策略、生成轨迹锚点
        attack_plan = self.llm_planner.generate_attack_plan(env_state, self.user_instruction)

        # print("attack_plan:", attack_plan)
        if not attack_plan:
            return
        target_id = attack_plan.get("attack_target_id")
        anchors = attack_plan.get("anchors", [])
        strategy = attack_plan.get("strategy", "unknown")
        # 边界检查：确保目标ID在当前活跃代理范围内
        if target_id != -1 and target_id is not None:
            # LLM 决定攻击
            self.last_attack_frame = env.current_step
            print("\n攻擊計劃:", attack_plan)
        elif target_id == -1:
            # LLM 决定不攻击
            self.last_query_frame = env.current_step
            print("reason:",attack_plan)
        print(f"[Adversarial Generator] Attack Plan: Target ID: {target_id}, Anchors: {anchors}, Strategy: {strategy}")
        if target_id is None:
            print("[Error] target_id is None.")  # DEBUG
            return

        # 检查 target_id 是否存在于 env_state['agents'] 中的 "id" 字段
        if not any(agent.get("id") == target_id for agent in env_state['agents']):
            print(f"[Error] Invalid attack_target_id: {target_id}. No matching agent found in env_state['agents'].")  # DEBUG
            return

        if not anchors or not all(len(anchor) == 2 for anchor in anchors):
            print("[Error] Invalid anchors.")
            return
        print(f"[Adversarial Generator] Target ID: {target_id}, Strategy: {strategy}")

        # 3. 提取目标初始状态
        for agent in env_state['agents']:
            if agent.get("id") == target_id:
                init_state = agent['state']
                break

        # 4. 当前使用插值轨迹补全；后续可替换为已配置的扩散模型。
        refined_traj = self.diffusion_refiner.refine_trajectory(init_state, anchors)

        # 保存 LLM 锚点和细化轨迹到可视化状态
        self.llm_anchors = anchors
        self.diffusion_trajectory = refined_traj

        # 5. 将细化后的对抗轨迹注入仿真环境
        env.inject_adversarial_trajectory(target_id, refined_traj)

    def evaluate_reaction(self, env, info):
        """评估自车反应:收集碰撞、偏航及TTC指标"""
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

    def reset_episode_stats(self):
        """每个场景开始前重置单回合统计"""
        self._current_episode_min_ttc = float('inf')

    def finalize_episode_stats(self):
        """场景结束后记录本回合最小TTC"""
        if self._current_episode_min_ttc != float('inf'):
            self.ttc_list.append(self._current_episode_min_ttc)

    def compute_final_metrics(self):
        """计算并返回所有场景的最终评估指标"""
        return {
            'collision_rate': np.mean(self.collision_list) if self.collision_list else 0.0,
            'near_miss_rate': np.mean(self.near_miss_list) if self.near_miss_list else 0.0,
            'avg_min_ttc': np.mean(self.ttc_list) if self.ttc_list else float('inf')
        }

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
