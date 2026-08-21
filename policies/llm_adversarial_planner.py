# adversarial_modules.py
import json
import os
import base64
import numpy as np
from openai import OpenAI
from policies.obstacles import ObstacleCatalog, ObstaclePlacement
class LLMAdversarialPlanner:
    """基于大模型的对抗性策略规划器"""
    ATTACK_MODES = {"trajectory_only", "obstacle_only", "joint"}

    def __init__(self, model_name="qwen3.5-27b", client=None, use_multimodal=False,
                 use_obstacles=False, attack_mode=None):
        """
            LLM的输入输出全部基于局部坐标系,局部坐标系的原点为自车位置,y轴正方向为自车朝向。
            LLM类方法的输入输出全部基于全局坐标系,全局坐标系的原点为地图原点,y轴正方向为地图北方。因此需要在调用LLM前将环境状态归一化到局部坐标系,并在获取LLM输出后将其转换回全局坐标系。
        """
        self.model_name = model_name
        self.last_error = None
        # 仅在连接、鉴权、额度或服务端错误时置位；格式校验失败不代表服务不可用。
        self.last_request_failed = False
        self.use_multimodal = bool(use_multimodal)
        # 未指定模式时保留旧接口语义；新配置应显式传入 attack_mode。
        self.attack_mode = attack_mode or ("joint" if use_obstacles else "trajectory_only")
        if self.attack_mode not in self.ATTACK_MODES:
            raise ValueError(f"不支持的 LLM 攻击模式：{self.attack_mode}")
        self.allow_trajectory = self.attack_mode in {"trajectory_only", "joint"}
        # 模式优先于旧开关，避免仅障碍物模式被旧配置意外关闭。
        self.use_obstacles = self.attack_mode in {"obstacle_only", "joint"}
        self.obstacle_catalog = ObstacleCatalog()
        api_key = os.getenv("API_KEY", "sk-ws-H.EDIRXYI.CIQJ.MEQCIEMNMXNJhnr1bIUzENhRdivO3EEcLzTP8YnG21XC2MctAiAOi080BKTRkI7Wjn_sSJACVvlIOdrUHbvExwrO3Pu-DQ")
        if client is not None:
            self.client = client
            self.available = True
        elif api_key:
            self.client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("LLM_BASE_URL", "https://ws-qvq9xoxtkn7fpem7.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"),
            )
            self.available = True
        else:
            self.client = None
            self.available = False
            self.last_error = "服务器未设置 LLM_API_KEY"
            print("[大模型规划器] 未设置 LLM_API_KEY,本次运行将停用高级攻击规划,但仿真会继续。")
        self._use_local_coordinate = True  # 是否将环境状态归一化到以自车为中心的局部坐标系

    def _normalize_env_state(self, env_state_json):
        """
        归一化环境状态，将道路拓扑和交通参与者状态转换到以自车为中心、自车朝向为 y 轴正方向的局部坐标系。

        Args:
            env_state_json (dict): 包含自车状态、道路拓扑和交通参与者状态的 JSON 对象。

        Returns:
            dict: 归一化后的环境状态，包括局部坐标系下的道路拓扑和交通参与者状态。
        """
        if isinstance(env_state_json, str):
            env_state_json = json.loads(env_state_json)

        # 提取自车状态
        ego_state = env_state_json["ego_state"]
        ego_x, ego_y, ego_vx, ego_vy, ego_heading = ego_state

        # 计算旋转矩阵（将自车朝向调整为 y 轴正方向）
        cos_theta = np.cos(-ego_heading + np.pi / 2)  # 旋转角度为 -heading + 90°
        sin_theta = np.sin(-ego_heading + np.pi / 2)
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                    [sin_theta, cos_theta]])
        local_ego_velocity = rotation_matrix @ np.array([ego_vx, ego_vy])

        # 归一化道路拓扑
        normalized_route = []
        for point in env_state_json["route"]:
            global_point = np.array(point) - np.array([ego_x, ego_y])  # 平移到以自车为原点
            local_point = rotation_matrix @ global_point  # 旋转到局部坐标系
            normalized_route.append(local_point.tolist())

        # 归一化交通参与者状态
        normalized_agents = []
        for agent in env_state_json["agents"]:
            agent_id = agent["id"]
            agent_type = agent["type"]
            if isinstance(agent_type, str):
                agent_type = {
                    "vehicle": 0,
                    "pedestrian": 1,
                    "cyclist": 2,
                }.get(agent_type, 3)
            agent_x, agent_y, agent_vx, agent_vy, agent_heading = agent["state"]

            # 平移到以自车为原点
            global_position = np.array([agent_x, agent_y]) - np.array([ego_x, ego_y])
            local_position = rotation_matrix @ global_position  # 旋转到局部坐标系

            # 计算局部朝向
            local_heading = agent_heading - ego_heading + np.pi / 2

            # 归一化速度
            global_velocity = np.array([agent_vx, agent_vy])
            local_velocity = rotation_matrix @ global_velocity

            # 使用相同坐标变换保留每一帧历史状态，供大模型判断运动趋势。
            normalized_history = []
            for history_item in agent.get("history", []):
                history_state = history_item.get("state", [])
                if len(history_state) != 5:
                    continue
                hist_x, hist_y, hist_vx, hist_vy, hist_heading = history_state
                hist_position = rotation_matrix @ (
                    np.array([hist_x, hist_y]) - np.array([ego_x, ego_y])
                )
                hist_velocity = rotation_matrix @ np.array([hist_vx, hist_vy])
                normalized_history.append({
                    "state": [
                        float(hist_position[0]),
                        float(hist_position[1]),
                        float(hist_velocity[0]),
                        float(hist_velocity[1]),
                        float(hist_heading - ego_heading + np.pi / 2),
                    ],
                    "valid": bool(history_item.get("valid", False)),
                })

            # 保存归一化后的当前状态及历史状态。
            normalized_agents.append({
                "id": agent_id,
                "type": agent_type,
                "state": [
                    local_position[0],  # x
                    local_position[1],  # y
                    local_velocity[0],  # vx
                    local_velocity[1],  # vy
                    local_heading        # heading
                ],
                "history": normalized_history,
            })

        # 已存在障碍物也必须转到同一局部坐标系，供 LLM 避免重复或重叠摆放。
        normalized_obstacles = []
        for obstacle in env_state_json.get("static_obstacles", []):
            center = obstacle.get("center", [])
            if len(center) != 2:
                continue
            local_center = rotation_matrix @ (np.asarray(center, dtype=float) - np.array([ego_x, ego_y]))
            normalized_obstacle = dict(obstacle)
            normalized_obstacle["center"] = [float(local_center[0]), float(local_center[1])]
            normalized_obstacle["yaw"] = float(obstacle.get("yaw", 0.0) - ego_heading + np.pi / 2)
            normalized_obstacles.append(normalized_obstacle)

        # 返回归一化后的环境状态
        return {
            "ego_state": [
                0,
                0,
                float(local_ego_velocity[0]),
                float(local_ego_velocity[1]),
                np.pi / 2,
            ],  # 自车在局部坐标系中的状态
            "route": normalized_route,
            "agents": normalized_agents,
            "static_obstacles": normalized_obstacles,
            "history_order": env_state_json.get("history_order", "oldest_to_newest"),
            "current_step": env_state_json.get("current_step"),
        }

    def _convert_to_global(self, items, ego_x, ego_y, ego_heading, item_type="anchors"):
        """
        将局部坐标系下的点或状态转换回全局坐标系。

        Args:
            items (list): 局部坐标系下的点或状态列表。
                - 如果 item_type 为 "anchors"，每个元素是 [x, y]。
                - 如果 item_type 为 "agents"，每个元素是包含 "state" 的字典。
            ego_x (float): 自车的全局 x 坐标。
            ego_y (float): 自车的全局 y 坐标。
            ego_heading (float): 自车的全局朝向（弧度制）。
            item_type (str): 要转换的对象类型，"anchors" 或 "agents"。

        Returns:
            list: 转换到全局坐标系的点或状态列表。
        """
        cos_theta = np.cos(ego_heading - np.pi / 2)
        sin_theta = np.sin(ego_heading - np.pi / 2)
        inverse_rotation_matrix = np.array([[cos_theta, -sin_theta],
                                            [sin_theta, cos_theta]])

        global_items = []
        for item in items:
            if item_type == "anchors":
                # 转换锚点
                local_point = np.array(item)
                global_point = inverse_rotation_matrix @ local_point + np.array([ego_x, ego_y])
                global_items.append(global_point.tolist())
            elif item_type == "agents":
                # 转换交通参与者状态
                local_position = np.array(item["state"][:2])  # 提取局部位置 (x, y)
                global_position = inverse_rotation_matrix @ local_position + np.array([ego_x, ego_y])
                global_item = item.copy()
                global_item["state"][:2] = global_position.tolist()  # 更新全局位置
                global_items.append(global_item)

        return global_items

    def _convert_obstacle_plan_to_global(self, placements, ego_x, ego_y, ego_heading):
        """将大模型输出的局部障碍物摆放请求转换为全局坐标。"""
        converted = []
        rotation = ego_heading - np.pi / 2
        for placement in placements:
            item = dict(placement)
            local_center = np.asarray(item["center"], dtype=float)
            global_center = self._convert_to_global([local_center], ego_x, ego_y, ego_heading)[0]
            item["center"] = global_center
            # 坐标与朝向必须同步逆旋转，包装器及后端只接收全局约定。
            item["yaw"] = float(item.get("yaw", 0.0) + rotation)
            converted.append(item)
        return converted

    def _build_prompt(
        self,
        env_state_json,
        user_instruction=None,
        include_image=None,
    ):
        """构造动态轨迹攻击（可选联合静态障碍物）的完整提示词。

        输入的 ``env_state_json`` 是 Simulator 导出的结构化场景状态，可以是字典或
        JSON 字符串；开启局部坐标后会统一转换为“自车在原点、向前为正 y”的坐标契约。
        ``user_instruction`` 保留多条用户需求，``include_image`` 决定是否声明 BEV 图像输入。
        本函数只构造提示词，不执行请求、不修改仿真状态，也不代替响应后的程序化校验。
        """

        # 第一步：把 Simulator 全局坐标转为 LLM 统一使用的自车局部坐标。
        if self._use_local_coordinate:
            env_state_json = self._normalize_env_state(env_state_json)

        # 第二步：合并实例默认值和单次调用覆盖值，确定输入模态与攻击能力边界。
        if include_image is None:
            include_image = self.use_multimodal
        include_obstacles = self.use_obstacles

        # 第三步：将用户指令规范为非空列表，保证提示词始终有明确的任务目标。
        if user_instruction is None:
            user_instruction = ["Generate a safety-critical scenario for the ego vehicle."]
        elif isinstance(user_instruction, str):
            user_instruction = [user_instruction]
        instructions = [str(item).strip() for item in user_instruction if str(item).strip()]
        if not instructions:
            instructions = ["Generate a safety-critical scenario for the ego vehicle."]
        instruction_str = "\n".join(f"- {item}" for item in instructions)

        # obstacle_only 有独立的简化提示词，避免动态轨迹的严格门控抑制障碍物调度。
        if self.attack_mode == "obstacle_only":
            return self._build_obstacle_only_prompt(env_state_json, instruction_str)

        if include_obstacles:
            # joint 模式只暴露抽象模板名称和语义，不向 LLM 泄露任何后端句柄或仿真器 API。
            # 只在显式开启时把领域模板暴露给 LLM；不暴露包装器、后端或 CARLA API。
            obstacle_catalog_text = "\n".join(
                f"        - `{item['type']}` — {item['description']}."
                for item in self.obstacle_catalog.describe()
            )
            obstacle_state_description = """
        4. **`static_obstacles`** — existing abstract static obstacles. They are already
        present in the scene and must not be duplicated or overlapped.
            """
            obstacle_planning_section = f"""
        ### Optional Static-Obstacle Scheduling
        Static-obstacle scheduling is enabled for this request. Decide independently whether
        to include an `obstacle_plan`; it may be empty even when a dynamic trajectory attack
        is emitted, and it may be non-empty without a dynamic attack. A route centerline is
        sufficient road evidence in text-only mode: place the obstacle near a forward route
        segment instead of declining solely because explicit drivable-area boundaries are absent.
        Use a temporary road obstruction when it can create a meaningful, avoidable planning
        challenge without immediate overlap with the ego or another participant.

        You may use only these abstract obstacle templates. They are design primitives, not
        simulator APIs:
        {obstacle_catalog_text}

        Each `obstacle_plan` entry must contain exactly `type`, `center`, `yaw`, `count`, and
        `spacing`. `center` and `yaw` use the same ego-centric local frame as the JSON state.
        `count` is the number of repeated elements and must be 1–12; `spacing` is 0.3–8 m.
        At most two template groups may be requested. Place them only on a plausible nearby
        road segment, normally 6–30 m ahead of the ego, without overlapping existing traffic or
        existing static obstacles. Do not create an unavoidable immediate collision. A
        disabled vehicle always uses `count: 1`.
            """
            obstacle_output_field = """
        - `"obstacle_plan"` (list) — an empty list or up to two abstract obstacle-template
          requests. This field is required when static-obstacle scheduling is enabled.
            """
        else:
            # trajectory_only 模式明确禁止障碍物字段，使模型输出与后续 JSON 契约一致。
            obstacle_state_description = ""
            obstacle_planning_section = """
        Static-obstacle scheduling is disabled. Do not output obstacle fields and do not
        propose static obstacle placement; generate only adversarial traffic trajectories.
            """
            obstacle_output_field = ""

        if include_image:
            # 多模态下，图像只用于车道拓扑和空间关系；精确坐标和编号仍以 JSON 为准。
            input_count = "three"
            image_input = """
        1. **A current-frame BEV image** showing the ego vehicle, other traffic
        participants with stable IDs and headings, lane centerlines, and drivable-area
        boundaries. Use it only for qualitative spatial and road-geometry understanding.

        2. **A JSON object** containing precise numerical states and history. Use the
        JSON, rather than estimating coordinates from the image, for quantitative analysis.

        3. **User instructions** describing the preferred dangerous scenario.
            """
            image_analysis_rule = (
                "- Use the BEV image to verify lane geometry, drivable boundaries, "
                "participant headings, and relative topology."
            )
            final_input_description = "the JSON state, the attached BEV image, and the user instructions"
        else:
            # 纯文本模式不暗示存在图像，几何推理完全依赖路线和交通参与者状态。
            input_count = "two"
            image_input = """
        1. **A JSON object** containing precise numerical states and history.

        2. **User instructions** describing the preferred dangerous scenario.
            """
            image_analysis_rule = (
                "- No image is provided in text-only mode; infer geometry only from the JSON route and states."
            )
            final_input_description = "the JSON state and the user instructions"

        # 第四步：组装主提示词。其内容按“输入契约→未来推演→可行性门控→输出契约”排列。
        prompt = f"""
        You are an expert in autonomous driving safety testing and adversarial scenario generation. Your task is to decide whether to create a safety-critical test case for the ego vehicle using the attack capabilities enabled in this request.

        ### Input Data Specification
        You are provided with **{input_count} types of input**:

        {image_input}

        Follow the user instructions when they are dynamically feasible and safe enough to
        constitute a meaningful test. If they conflict with the observed scene, select the
        closest feasible alternative or decline the attack and explain why.

        **User instruction**: {instruction_str}

        The JSON object uses an ego-centric local coordinate frame: the ego is at `[0, 0]`,
        positive y points forward along the ego heading, and positive x points to the ego's
        right. It has the following primary fields:

        1. **`ego_state`** — a list `[x, y, vx, vy, heading]` representing the ego vehicle's current state:
        - `x`, `y`: position coordinates (meters) in the ego-centric local frame.
        - `vx`, `vy`: velocity components (m/s) along the x and y axes.
        - `heading`: orientation (radians) of the ego vehicle (0 = positive x-axis, π/2 = positive y-axis).

        2. **`route`** — a list of `[x, y]` waypoints defining the centerline of the ego's planned path. The route is ordered from the ego's current position forward, and each point represents a position along the lane centerline (meters).

        3. **`agents`** — a list of other traffic participants in the vicinity. Each agent is a dictionary with three keys:
        - `"id"`: unique integer identifier for the agent (use this ID when selecting an attack target).
        - `"state"`: a list `[x, y, vx, vy, heading]` representing the agent's current position, velocity components, and heading (same units and frame as ego_state).
        - `"type"`: an integer indicating the agent category:
            - `0` = vehicle
            - `1` = pedestrian
            - `2` = cyclist
        - `"history"`: oldest-to-newest state records with a `valid` mask. Ignore invalid records.

        {obstacle_state_description}

        The additional `current_step` and `history_order` fields define the current simulation
        time index and confirm the temporal order of the history.

        **Note**: The agents' state does not include length/width in this input, but assume standard vehicle dimensions for reasoning about lane occupancy and collision risk.

        ### Decision Process
        Before selecting any attack, you must first evaluate the current scene by analyzing:
        - The ego's position, speed, heading, and planned route.
        - The relative position, speed, heading, and type of each agent with respect to the ego.
        - The spatial relationship between agents and the planned route (e.g., which agents are near the ego's path).
        {image_analysis_rule}
        - **The user instruction to understand what specific type of danger is requested (e.g., cut-in, braking, occlusion).**

        **Critical requirement**: You **must** reason about the **future motion** of both the ego vehicle and all relevant agents over the next 3 seconds. Specifically:
        - **Predict the ego's future path**: Given its current velocity (`vx`, `vy`) and heading, and assuming it roughly follows the provided `route`, estimate where the ego will be at t=1s, 2s, and 3s.
        - **Predict each agent's future state**: Use the valid history together with current position, velocity, and heading to estimate plausible states at the same timestamps.
        - Then, based on these predicted states, decide whether an attack is feasible and meaningful. **An attack should exploit the predicted future interaction** – e.g., an agent that will be near the ego's future path is a suitable target.

        ### Mandatory Feasibility Gates
        A dynamic-trajectory attack is valid only if every gate below passes. User preference never overrides these gates.
        1. The target can approach within 10 m of the ego's estimated t=1s, 2s, or 3s position without teleporting, reversing unexpectedly, or leaving the drivable area.
        2. `anchors[0]` must match the target's current position within 1 m. With one-second anchor intervals, derive every segment velocity and acceleration and reject plans that conflict with the target's current velocity or exceed the stated physical limits.
        3. `hard_brake` and `slow_down` require a target ahead in the ego lane, traveling in approximately the same direction. Never use these strategies for a lateral, crossing, stationary, or oncoming target.
        4. `cut_in` requires a moving target in an immediately adjacent lane. Its lateral shift should be approximately one lane width and should intersect the ego corridor within 3 seconds; do not move across multiple lanes.
        5. `lane_change` must remain connected to a visible adjacent drivable lane. `occlusion` requires visible geometry that actually blocks a relevant line of sight.
        6. If the image and JSON appear inconsistent, treat the JSON IDs, states, velocities, and history as authoritative and use the image only for qualitative geometry.

        Before returning `"attack": true`, silently verify the strategy semantics, target ID, initial anchor, derived velocities, derived accelerations, and estimated closest approach. If any check fails, return `"attack": false`.

        {obstacle_planning_section}

        Only proceed with a dynamic attack if it can produce a **meaningful, avoidable challenge** that tests the ego vehicle's perception, planning, or control capabilities. A short-TTC challenge is acceptable when the ego still has a plausible braking or lane-avoidance response; reject only an immediate overlap or a clearly unavoidable collision. If the scene is not suitable for a useful dynamic attack, output `"attack": false` and explain why. **If the user instruction cannot be realistically fulfilled due to scene constraints, mention this in the `reason` and choose the closest feasible alternative or decide not to attack.**

        **Important**: You **must** always provide a `"reason"` field in your final JSON output, regardless of whether you decide to attack or not. The reason should concisely justify your decision.

        **Examples of scenarios where you should NOT attack:**
        - All agents are too far away (e.g., >100 m) and cannot interact with the ego within the next 3–5 seconds.
        - No agents are near the ego's lane or adjacent lanes that could pose a threat.
        - All surrounding agents are moving at similar speeds and directions with no chance of cut‑in or sudden braking.
        - The ego vehicle is already in an extreme situation (e.g., about to collide or near a sharp turn) – adding an attack would be redundant or make the test meaningless.
        - An attack would leave the ego with insufficient reaction time or space (e.g., ego speed is very high and the target is too close), resulting in an unavoidable collision that does not provide useful evaluation data.

        **If you decide to attack, you must design the attack trajectory (`anchors`) by explicitly considering the ego's predicted future positions.** The anchors should lead the target agent to intersect or closely approach the ego's predicted path at a future time, creating a challenging but not impossible scenario. **Additionally, ensure the attack is consistent with the user instruction** – for example, if the user asks for a "sudden brake", the anchors should show the target decelerating sharply.

        If you decide **NOT** to attack, set:
        - `"attack": false`
        - `"attack_target_id": -1`
        - `"strategy": "none"`
        - `"anchors": []`
        - `"reason": "A brief explanation of why no attack is appropriate."`

        If you decide **TO** attack, you must choose **one** high-level strategy from the following list. **Prefer the predefined strategies when applicable; use `"others"` only if none of them fit the intended maneuver:**
        - `"cut_in"` – the target vehicle moves from an adjacent lane into the ego's lane, forcing the ego to react.
        - `"hard_brake"` – the target vehicle brakes sharply in front of the ego.
        - `"slow_down"` – the target gradually decelerates, requiring the ego to adjust speed or change lanes.
        - `"lane_change"` – the target changes lanes, potentially cutting across the ego's path (more general than cut_in).
        - `"occlusion"` – the target (typically a large vehicle) blocks the ego's view, e.g., at intersections or curves.
        - `"sudden_acceleration"` – the target accelerates unexpectedly, which may cause the ego to misjudge gap and risk a rear‑end collision.
        - **`"others"` – any other adversarial behavior not covered above. If you choose this, you must briefly describe the specific maneuver in the `"reason"` field.**

        When attacking, you must also generate a coarse trajectory (**anchors**) for the chosen agent over the next 3 seconds. Provide 4 waypoints (at t=0, 1, 2, 3 seconds) in the same coordinate frame and format as the input (list of `[x, y]` pairs). The trajectory should be:
        - Feasible and realistic for the selected strategy.
        - Consistent with the agent's current dynamics (position and heading).
        - Designed to create a meaningful challenge for the ego vehicle.
        - **The path must be spatially smooth** (no sharp discontinuities or unrealistic jumps between consecutive waypoints).
        - **The implied speeds and accelerations** (derived from the differences between waypoints at 1‑second intervals) must be within plausible limits for the agent type (e.g., for vehicles: lateral acceleration < 5 m/s², longitudinal acceleration between -5 and 5 m/s²; for pedestrians/cyclists, lower bounds).
        - **The heading changes should be gradual** – the trajectory should not require the agent to instantaneously reorient.
        - **The overall motion should resemble a natural driving/riding/walking behavior** that a real traffic participant could execute, avoiding physically impossible maneuvers (e.g., sudden 90° turns at high speed).
        - **Crucially, the anchors must be placed with respect to the ego's predicted positions at the corresponding times**, so that the attack reaches a critical point (e.g., intersection or close proximity) when the ego is nearby.

        ### Output Format
        Your final output must be a **single, valid JSON object** containing the following keys:
        - `"attack"` (boolean)
        - `"attack_target_id"` (integer, use `-1` when not attacking)
        - `"strategy"` (string, use `"none"` when not attacking; when attacking, use one of the listed strategies, including `"others"`)
        - `"anchors"` (list of `[x, y]` pairs, empty list when not attacking)
        - `"reason"` (string) – **Required for both attack and no‑attack cases.** A concise explanation of your decision (max 60 words). Include:
        - Why the chosen agent is the most suitable (if attacking), or why no agent is suitable (if not attacking).
        - How the selected strategy relates to the ego's route and current state.
        - What specific safety capability of the ego this test is designed to evaluate (if attacking).
        - **If you choose `"others"`, clearly state the specific behavior you intend.**
        - **Optionally, mention how the attack aligns with the user instruction, or why it cannot be fully satisfied.**

        {obstacle_output_field}

        Do **not** include any extra text, comments, or markdown outside the JSON.

        ### Example Input (for reference)
        The following is a simplified example of what the input may look like. Use this only to understand the data structure; your actual input will follow the same format but with different values.

        {{
        "ego_state": [9.40, 117.45, 1.52, 14.76, 1.47],
        "route": [[7.41, 98.26], [7.52, 99.25], [7.63, 100.25], [7.73, 101.24], [7.83, 102.24]],
        "agents": [
            {{"id": 18, "state": [13.61, 105.49, 0.0, 0.0, 2.95], "type": 2}},
            {{"id": 20, "state": [5.67, 112.60, 0.0, 0.0, -1.71], "type": 1}}
        ]
        }}

        Now analyze {final_input_description}. Generate the required JSON response:

        {env_state_json}
        """

        # 最后附加当前场景 JSON；响应必须是单个 JSON 对象，再由后续逻辑执行字段和物理校验。
        return prompt

    def _build_obstacle_only_prompt(self, normalized_env_state, instruction_str):
        """构造仅静态障碍物模式的独立提示词，避免动态轨迹门控抑制障碍物测试。"""
        catalog_text = "\n".join(
            f"- `{item['type']}`: {item['description']}"
            for item in self.obstacle_catalog.describe()
        )
        state_text = json.dumps(normalized_env_state, ensure_ascii=False, indent=2)
        return f"""
You are an autonomous-driving safety-test designer. This is **static-obstacle-only mode**.
Do not control traffic agents and do not create adversarial trajectories. On every query,
independently decide whether a temporary static obstacle can create a challenging but avoidable
test for the ego vehicle.

The input state uses an ego-centric local frame: ego is at [0, 0], positive y is forward,
and positive x is right. `route` is an ordered forward road centerline. In text-only mode,
the route centerline is sufficient evidence for a plausible road segment: do not decline merely
because an explicit drivable-area polygon is unavailable. Use agents and existing obstacles to
avoid overlap, but a dynamic-agent interaction is not required.

User instruction:
{instruction_str}

Available abstract templates (these are not simulator APIs):
{catalog_text}

Obstacle planning rules:
- Choose an empty `obstacle_plan` only when no forward route segment can support an avoidable test.
- Otherwise prefer one compact, temporary obstruction 6–30 m ahead near the route centerline.
- A short-TTC planning challenge is acceptable if the ego has a plausible braking or lateral-avoidance response.
- Do not place an obstacle at the ego position, overlap an existing agent/obstacle, or force an immediate overlap.
- Use at most two template groups. Each item contains exactly `type`, `center`, `yaw`, `count`, `spacing`.
- `count` is 1–12, `spacing` is 0.3–8 m, and `disabled_vehicle` always uses count 1.

Return exactly one JSON object with:
- `attack`: false
- `attack_target_id`: -1
- `strategy`: "none"
- `anchors`: []
- `reason`: concise explanation of the placement decision
- `obstacle_plan`: an empty list or one/two valid placement objects

Do not include markdown or other text.

Current scene state:
{state_text}
"""

    @staticmethod
    def _image_data_url(scene_image):
        """将当前帧 PNG 字节编码为多模态接口可读取的数据 URL。"""
        if isinstance(scene_image, str):
            if scene_image.startswith(("data:image/", "http://", "https://")):
                return scene_image
            with open(scene_image, "rb") as image_file:
                scene_image = image_file.read()
        if not isinstance(scene_image, (bytes, bytearray)):
            raise TypeError("多模态模式要求 scene_image 为 PNG 字节、文件路径或图像 URL")
        encoded = base64.b64encode(scene_image).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _parse_json_object(raw):
        """解析裸 JSON 或常见 Markdown 代码围栏中的单个 JSON 对象。"""
        if not isinstance(raw, str):
            raise TypeError("大模型输出必须是字符串")
        text = raw.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        return json.loads(text)

    def _request_attack_plan(self, prompt, scene_image=None):
        """根据显式模式开关执行纯文本或多模态请求。"""
        if self.use_multimodal:
            if scene_image is None:
                raise ValueError("多模态模式已开启，但未提供当前帧渲染")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._image_data_url(scene_image)},
                        },
                    ],
                }],
                extra_body={"enable_thinking": False},
            )
            message = response.choices[0].message
            reasoning = getattr(message, "reasoning_content", "") or ""
            raw = message.content
            try:
                return reasoning, self._parse_json_object(raw)
            except (TypeError, json.JSONDecodeError) as e:
                raise ValueError(f"无法将多模态大模型输出解析为 JSON：{raw}") from e

        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            extra_body={"enable_thinking": False},
        )
        return self._extract_reasoning_and_result(response)

    @staticmethod
    def _validate_attack_plan(attack_plan, normalized_env_state):
        """在坐标转换前验证大模型计划的语义、几何和短时动力学。"""
        if not isinstance(attack_plan.get("attack"), bool):
            raise ValueError("attack 字段必须是布尔值")
        if not attack_plan["attack"]:
            if attack_plan.get("attack_target_id") != -1:
                raise ValueError("不攻击时 attack_target_id 必须为 -1")
            if attack_plan.get("anchors"):
                raise ValueError("不攻击时 anchors 必须为空")
            return

        target_id = attack_plan.get("attack_target_id")
        target = next(
            (agent for agent in normalized_env_state["agents"] if agent["id"] == target_id),
            None,
        )
        if target is None:
            raise ValueError(f"攻击目标 {target_id} 不在当前规划参与者中")

        anchors = np.asarray(attack_plan.get("anchors"), dtype=np.float64)
        if anchors.shape != (4, 2) or not np.isfinite(anchors).all():
            raise ValueError("攻击轨迹必须包含四个有限二维锚点")
        target_state = np.asarray(target["state"], dtype=np.float64)
        if np.linalg.norm(anchors[0] - target_state[:2]) > 1.5:
            raise ValueError("第一个锚点与目标当前位置相差超过 1.5 米")

        segment_velocities = np.diff(anchors, axis=0)
        if np.linalg.norm(segment_velocities[0] - target_state[2:4]) > 9.0:
            raise ValueError("第一段锚点速度与目标当前速度不连续")
        if np.max(np.linalg.norm(segment_velocities, axis=1)) > 50.0:
            raise ValueError("锚点隐含速度超过车辆合理上限")
        if len(segment_velocities) > 1:
            accelerations = np.diff(segment_velocities, axis=0)
            if np.max(np.linalg.norm(accelerations, axis=1)) > 9.0:
                raise ValueError("锚点隐含加速度超过车辆合理上限")

        ego_state = np.asarray(normalized_env_state["ego_state"], dtype=np.float64)
        times = np.arange(1.0, 4.0)[:, None]
        ego_future = ego_state[:2] + times * ego_state[2:4]
        closest_approach = float(
            np.min(np.linalg.norm(anchors[1:] - ego_future, axis=1))
        )
        if closest_approach > 10.0:
            raise ValueError(
                f"锚点与自车三秒预测位置的最近距离为 {closest_approach:.1f} 米，不能形成有效交互"
            )

        strategy = attack_plan.get("strategy")
        target_speed = float(np.linalg.norm(target_state[2:4]))
        ego_speed = float(np.linalg.norm(ego_state[2:4]))
        direction_similarity = 0.0
        if target_speed > 1e-6 and ego_speed > 1e-6:
            direction_similarity = float(
                np.dot(target_state[2:4], ego_state[2:4]) / (target_speed * ego_speed)
            )
        if strategy in {"hard_brake", "slow_down"}:
            if abs(target_state[0]) > 4.5 or target_state[1] < -1.0:
                raise ValueError(f"{strategy} 只适用于自车同车道前方目标")
            if target_speed < 1.0 or direction_similarity < 0.5:
                raise ValueError(f"{strategy} 目标必须与自车同向行驶且不能近似静止")
        if strategy == "cut_in":
            if not 1.5 <= abs(target_state[0]) <= 9.0:
                raise ValueError("cut_in 目标必须位于紧邻车道")
            if abs(anchors[-1, 0]) > 5.5:
                raise ValueError("cut_in 终点没有进入自车行驶走廊")

    def _validate_obstacle_plan(self, obstacle_plan, normalized_env_state):
        """验证大模型障碍物请求只使用允许的抽象模板且不会形成即时碰撞。"""
        if not isinstance(obstacle_plan, list):
            raise ValueError("obstacle_plan 必须是列表")
        if len(obstacle_plan) > 2:
            raise ValueError("一次最多请求两组障碍物模板")
        # 当前已有障碍物同样参与间距检查，避免多次低频规划叠放障碍物。
        existing_centers = [
            np.asarray(item.get("center", []), dtype=float)
            for item in normalized_env_state.get("static_obstacles", [])
            if len(item.get("center", [])) == 2
        ]
        for payload in obstacle_plan:
            if not isinstance(payload, dict):
                raise ValueError("obstacle_plan 中的每项必须是对象")
            placement = ObstaclePlacement.from_dict(payload)
            if placement.type not in self.obstacle_catalog.type_names:
                raise ValueError(f"障碍物类型不在允许模板中：{placement.type}")
            if placement.type == "disabled_vehicle" and placement.count != 1:
                raise ValueError("disabled_vehicle 的 count 必须为 1")
            center = np.asarray(placement.center, dtype=float)
            if center[1] < 4.0 or np.linalg.norm(center) < 4.0:
                raise ValueError("障碍物必须放置在自车前方至少 4 米处")
            if any(np.linalg.norm(center - existing) < 2.0 for existing in existing_centers):
                raise ValueError("障碍物不能与已有静态障碍物重叠")

    def generate_attack_plan(self, env_state, user_instruction, scene_image=None):
        if not self.available or self.client is None:
            return None

        self.last_request_failed = False

        # 将结构化环境状态序列化后再交给大模型。
        env_state_json = json.dumps(env_state, indent=2)
        prompt = self._build_prompt(
            env_state_json,
            user_instruction,
            include_image=self.use_multimodal,
        )
        try:
            reasoning, attack_plan = self._request_attack_plan(prompt, scene_image)
            # print(f"推理过程: {reasoning[:200]}...")  # 仅打印开头
            # print(f"攻击计划: {attack_plan}")


            if attack_plan is None:
                return None
            if not isinstance(attack_plan, dict):
                raise ValueError("大模型输出不是 JSON 对象")
            required_keys = {"attack", "attack_target_id", "strategy", "anchors", "reason"}
            if self.use_obstacles:
                # 启用后 obstacle_plan 是契约字段，即使本次选择只生成轨迹也必须返回空列表。
                required_keys.add("obstacle_plan")
            missing_keys = required_keys.difference(attack_plan)
            if missing_keys:
                raise ValueError(f"大模型输出缺少字段：{sorted(missing_keys)}")

            normalized_env_state = (
                self._normalize_env_state(env_state)
                if self._use_local_coordinate
                else env_state
            )
            if not self.allow_trajectory and attack_plan.get("attack") is not False:
                raise ValueError("仅静态障碍物模式下 attack 必须为 false")
            self._validate_attack_plan(attack_plan, normalized_env_state)
            if self.use_obstacles:
                self._validate_obstacle_plan(attack_plan["obstacle_plan"], normalized_env_state)
            elif "obstacle_plan" in attack_plan:
                raise ValueError("静态障碍物调度未开启，不应输出 obstacle_plan")

            if attack_plan.get("attack") is True and attack_plan.get("anchors") and self._use_local_coordinate:
                # 提取自车状态
                ego_state = env_state["ego_state"]
                ego_x, ego_y, ego_heading = ego_state[0], ego_state[1], ego_state[4]

                # 转换锚点到全局坐标系
                print("\n(LOCAL) attack_plan['anchors']:", attack_plan["anchors"])
                global_anchors = self._convert_to_global(attack_plan["anchors"], ego_x, ego_y, ego_heading)
                attack_plan["anchors"] = global_anchors

            if self.use_obstacles and attack_plan.get("obstacle_plan") and self._use_local_coordinate:
                # 仅在校验通过后转换，避免不合法局部请求进入仿真后端。
                ego_state = env_state["ego_state"]
                attack_plan["obstacle_plan"] = self._convert_obstacle_plan_to_global(
                    attack_plan["obstacle_plan"],
                    ego_state[0],
                    ego_state[1],
                    ego_state[4],
                )

            return attack_plan

        except Exception as e:
            self.last_error = str(e)
            self.last_request_failed = self._is_service_failure(e)
            print(f"[大模型规划器] 本次高级攻击规划失败，已跳过且仿真继续：{e}")
            return None

    @staticmethod
    def _is_service_failure(error):
        """判断异常是否表示 LLM 服务暂不可用，而非模型计划本身不合规。"""
        status_code = getattr(error, "status_code", None)
        if status_code in {401, 403, 408, 429} or (isinstance(status_code, int) and status_code >= 500):
            return True
        return type(error).__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "RateLimitError",
            "AuthenticationError",
            "PermissionDeniedError",
        }

    def _extract_reasoning_and_result(self, response):
        reasoning = ""
        plan = None
        for item in response.output:
            if item.type == "reasoning" and item.summary:
                reasoning = item.summary[0].text
            elif item.type == "message" and item.content:
                raw = item.content[0].text
                try:
                    plan = self._parse_json_object(raw)
                except (TypeError, json.JSONDecodeError) as e:
                    raise ValueError(f"无法将大模型输出解析为 JSON：{raw}") from e
        return reasoning, plan




"""
    測試記錄:qwen3.7-flash-2026-07-15   LOCAL     無法生成威脅軌跡
    測試記錄:qwen3.7-flash-2026-07-15   GLOBAL    無法生成威脅軌跡
    測試記錄:qwen3.7-max                GLOBAL    放棄大部分對抗場合,生成了1個對抗場景,並造成了碰撞
    測試記錄:qwen3.7-max                LOCAL     無法生成威脅軌跡
    測試記錄:qwen3.7-plus               GLOBAL    無法生成威脅軌跡
    ————要求LLM推理未來軌跡————
    測試記錄:qwen3.7-plus               LOCAL     生成碰撞場景，但是似乎有些過於直接了
"""
