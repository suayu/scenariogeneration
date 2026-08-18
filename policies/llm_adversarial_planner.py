# adversarial_modules.py
import json
import os
import numpy as np
from openai import OpenAI
class LLMAdversarialPlanner:
    """基于大模型的对抗性策略规划器"""
    def __init__(self, model_name="qwen3.5-35b-a3b"):
        """
            LLM的输入输出全部基于局部坐标系,局部坐标系的原点为自车位置,y轴正方向为自车朝向。
            LLM类方法的输入输出全部基于全局坐标系,全局坐标系的原点为地图原点,y轴正方向为地图北方。因此需要在调用LLM前将环境状态归一化到局部坐标系,并在获取LLM输出后将其转换回全局坐标系。
        """
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.getenv("LLM_BASE_URL", "https://ws-qvq9xoxtkn7fpem7.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"),
        )
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
            agent_x, agent_y, agent_vx, agent_vy, agent_heading = agent["state"]

            # 平移到以自车为原点
            global_position = np.array([agent_x, agent_y]) - np.array([ego_x, ego_y])
            local_position = rotation_matrix @ global_position  # 旋转到局部坐标系

            # 计算局部朝向
            local_heading = agent_heading - ego_heading + np.pi / 2

            # 归一化速度
            global_velocity = np.array([agent_vx, agent_vy])
            local_velocity = rotation_matrix @ global_velocity

            # 保存归一化后的状态
            normalized_agents.append({
                "id": agent_id,
                "type": agent_type,
                "state": [
                    local_position[0],  # x
                    local_position[1],  # y
                    local_velocity[0],  # vx
                    local_velocity[1],  # vy
                    local_heading        # heading
                ]
            })

        # 返回归一化后的环境状态
        return {
            "ego_state": [0, 0, ego_vx, ego_vy, np.pi / 2],  # 自车在局部坐标系中的状态
            "route": normalized_route,
            "agents": normalized_agents
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

    def _build_prompt(self, env_state_json, user_instruction=["Generate a safety-critical scenario for the ego vehicle."]):
        """
            env_state_json:全局坐标系下的环境状态JSON字符串,包含ego_state、route、agents等信息
        """

        if self._use_local_coordinate:
            env_state_json = self._normalize_env_state(env_state_json)
            print("\n(LOCAL) env_state_json:",env_state_json['agents'])
        # assert False

        instruction_str = ", ".join(user_instruction)

        prompt = f"""
        You are an expert in autonomous driving safety testing and adversarial scenario generation. Your task is to decide whether to create a safety-critical test case for the ego vehicle by controlling one other traffic participant (agent).

        ### Input Data Specification
        You are provided with **three types of input**:

        1. **A BEV (Bird's Eye View) image** of the current simulation scene. This image shows the road layout, lane markings, ego vehicle (highlighted), and all surrounding agents with their shapes and orientations. **Use this image to understand spatial relationships, lane geometry, and relative positions intuitively.**

        2. **A JSON object** containing precise numerical states (detailed below). **Use this for quantitative analysis of velocities, headings, and future motion prediction.**

        3. **A user instruction** (natural language) describing the desired dangerous scenario. **You should make a best effort to generate an attack that aligns with this instruction**, while respecting the actual scene dynamics and feasibility.

        **User instruction**: {user_instruction}

        The JSON object has the following three top-level keys:

        1. **`ego_state`** — a list `[x, y, vx, vy, heading]` representing the ego vehicle's current state:
        - `x`, `y`: position coordinates (meters) in the global frame.
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

        **Note**: The agents' state does not include length/width in this input, but assume standard vehicle dimensions for reasoning about lane occupancy and collision risk.

        ### Decision Process
        Before selecting any attack, you must first evaluate the current scene by analyzing:
        - The ego's position, speed, heading, and planned route.
        - The relative position, speed, heading, and type of each agent with respect to the ego.
        - The spatial relationship between agents and the planned route (e.g., which agents are near the ego's path).
        - **The BEV image to confirm lane geometry, nearby obstacles, and the overall scene context.**
        - **The user instruction to understand what specific type of danger is requested (e.g., cut-in, braking, occlusion).**

        **Critical requirement**: You **must** reason about the **future motion** of both the ego vehicle and all relevant agents over the next 5 seconds. Specifically:
        - **Predict the ego's future path**: Given its current velocity (`vx`, `vy`) and heading, and assuming it roughly follows the provided `route`, estimate where the ego will be at t=1s, 2s, and 5s.
        - **Predict each agent's future state**: Using each agent's current position, velocity, and heading, extrapolate their likely positions at the same future timestamps (assuming constant velocity or plausible short-term motion).
        - Then, based on these predicted states, decide whether an attack is feasible and meaningful. **An attack should exploit the predicted future interaction** – e.g., an agent that will be near the ego's future path is a suitable target.

        Only proceed with an attack if it can produce a **meaningful challenge** that tests the ego vehicle's perception, planning, or control capabilities. If the scene is not suitable for a useful attack, output `"attack": false` and explain why. **If the user instruction cannot be realistically fulfilled due to scene constraints, mention this in the `reason` and choose the closest feasible alternative or decide not to attack.**

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
        Your final output must be a **single, valid JSON object** containing exactly the following keys:
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

        Now, analyze the following actual environment state (JSON below) together with the BEV image and the user instruction provided. Generate your response:

        {env_state_json}
        """

        return prompt

    def generate_attack_plan(self, env_state, user_instruction):
        # print("env_state:", env_state)
        env_state_json = json.dumps(env_state, indent=2)
        prompt = self._build_prompt(env_state_json, user_instruction)
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                extra_body={"enable_thinking":False},  # 是否深度思考。根据需要开启/关闭。开启后token消耗大幅提升。
            )
            # print(f"LLM response: ",response)  # 仅打印开头
            reasoning, attack_plan = self._extract_reasoning_and_result(response)
            # print(f"推理过程: {reasoning[:200]}...")  # 仅打印开头
            # print(f"攻击计划: {attack_plan}")


            if attack_plan and "anchors" in attack_plan and self._use_local_coordinate:
                # 提取自车状态
                ego_state = env_state["ego_state"]
                ego_x, ego_y, ego_heading = ego_state[0], ego_state[1], ego_state[4]

                # 转换锚点到全局坐标系
                print("\n(LOCAL) attack_plan['anchors']:", attack_plan["anchors"])
                global_anchors = self._convert_to_global(attack_plan["anchors"], ego_x, ego_y, ego_heading)
                attack_plan["anchors"] = global_anchors

            return attack_plan

        except Exception as e:
            assert False, f"LLM planning failed: {e}"
        return None

    def _extract_reasoning_and_result(self, response):
        reasoning = ""
        plan = None
        for item in response.output:
            if item.type == "reasoning" and item.summary:
                reasoning = item.summary[0].text
            elif item.type == "message" and item.content:
                raw = item.content[0].text
                try:
                    plan = json.loads(raw)
                except Exception as e:
                    plan = raw
                    print("Warning: Failed to parse LLM output as JSON. Raw output:", raw)
                    assert e
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