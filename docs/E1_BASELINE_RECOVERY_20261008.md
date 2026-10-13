# SAFE-SIM / STRIVE 基线修复与验证（2026-10-08）

## 结论与当前状态

SAFE-SIM 的原单场已实际执行171帧、完成路线、未发生自车碰撞，背景车间和背景车-静态障碍碰撞帧均为0。原冒烟拒绝的是公共风险评估预测支路，而非SAFE-SIM无法生成或执行轨迹。STRIVE原单场在初始化反向传播时失败，不能视为有效的零碰撞实验。

已完成原因定位、代码修复和组件验证；SAFE-SIM、STRIVE完整单场重测正在运行，尚未通过完整闭环门禁。未启动250场正式实验，未提交GitHub。

## SAFE-SIM原失败的具体原因

|原171帧分类|帧数|原因|处理|
|---|---:|---|---|
|EA与可达性同时有效|42|原有效覆盖率24.56%，D=0.054594435|保留原始结果|
|联合预测不完整|64|forecast_incomplete_agent_coverage；近场车辆未来离开80米观测视野，预测掩码变为无效|只在评估副本持续跟踪源帧已有车辆|
|EA无效|65|所有这些源帧近场动态/静态对象集合为空，原代码把空集合当作NaN|完整输入且空近场集合EA=0，真实缺失仍无效|

64个预测拒绝帧不等于64次服务调用故障，也不等于64次攻击被安全筛选拒绝。

从原trace恢复源帧32、50、100做组件对照：前两帧旧分支分别在未来第26、10步因离开FOV缺失，诊断offroad=false、collision=false；persistent分支均生成完整3秒联合未来并通过校验。源帧100两分支均通过。组件只做假想滚动，真实执行帧为0，不能据此推断碰撞率。

### 修复原则与边界

- 保持真实SAFE-SIM diffusion生成和执行、安全筛选不变。CtRL-Sim用于公共可达性评估的代理预测，在副本中连续滚动30步，每步0.1秒。
- 新参数 --forecast-tracking persistent 持续跟踪源帧车辆，明确区分物理存在与观测视野。不是恒速补齐、不是忽略无效预测；缺失历史、非有限值、短时域、ID不一致仍拒绝。
- 原观察掩码实现保留为默认 observation；原风险预测实现保留为 --reachability-source native，可显式回退。
- EA新增输入与求解诊断，任何纳入计算的对象求解失败均使该帧无效，不能只保留其他有限对象来制造有效测量。
- EA=0不意味着D=0。D仍只统计EA与可达性同时有效的帧，没有有效帧D=null。
- 空集合语义属于评估协议修订，标记为ea_empty_set_zero_v2，三种方法应统一使用。历史R1报告不改写，也不能与新版未经标注直接比较。
- CtRL-Sim是代理未来，历史动作由已发生状态逆运动学重建，历史RTG用最大值占位。需要后续预测误差审计，不能把代理D宣称为真实未来风险真值。

## STRIVE修复与保真性

1. 官方网络在eval模式使用cuDNN GRU，潜变量优化loss.backward()报错：cudnn RNN backward can only be called in training mode。
2. 仅为RNN/GRU/LSTM前向选择原生可导实现；保留eval、冻结参数与原优化目标。CNN继续使用cuDNN；未改train或放宽安全门槛。
3. 下一步发现官方闭环函数的torch.arange等未指定device，与CUDA图索引相减报错。在隔离worker设置默认设备，官方源码保持不变。
4. 真实官方权重、真实自车策略回调的小预算组件测试通过：初始化2次、对抗优化1次、策略回调2次，输出1辆背景车的60步/6秒联合轨迹，耗时约12.18秒。该结果只证明模型—优化—策略反馈链路可运行；official_adv_success=false、真实执行帧0，不能作为攻击效果证据。
5. 完整单场恢复原75次初始化、300次对抗优化、每10帧重规划预算。计划数、实际攻击帧、官方优化成功分开记录，完整门禁要求真实执行。
6. 空背景集合显式标为无可攻击车辆，不伪称模型调用或攻击。运行失败保留trace、失败原因、attempt_result、诊断视频；未执行的碰撞结果为null，不能计为零碰撞成功。

当前STRIVE仍是官方优化器的Waymo适配版本：车道走廊语义代理缺少原NuScenes部分图层、未来初始化为CV而非真实未来。这些差异需要跨域审计，formal_ready=false；不能称为原论文完整复现。

## 修改文件（均位于服务器工作区）

|文件|作用|
|---|---|
|policies/risk_metrics.py|区分空近场对象、坏输入、求解失败；新增EA诊断|
|experiments/comparison/formal_metrics.py|逐帧记录EA诊断，联合预测有效性仍严格校验|
|experiments/comparison/ctrl_sim_forecast.py|可切换的持续跟踪评估副本与离场诊断|
|experiments/comparison/strive_official_worker.py|原生RNN梯度与CUDA工厂设备兼容|
|experiments/comparison/run_passive_baseline.py|跟踪参数接线、协议记录、失败统计|
|experiments/comparison/run_strive_baseline.py|空场景续行、真实攻击计数、失败产物保存|
|experiments/comparison/run_e1_smoke.py|选择方法、闭环门禁、统一协议快照|
|experiments/comparison/probe_e1_recovery.py|可复用历史源帧与官方优化组件探测|
|tests/comparison/test_ea_validity.py等|空集合、无效值、历史存在、RNN梯度与入口参数回归|

## 验证与正在运行的实验

- 核心回归集56项通过，覆盖EA、CtRL副本隔离、完整掩码、RNN梯度、官方适配、失败审计与冒烟门禁。
- 修复启动参数误传后，相关入口/门禁回归14项通过（含新增真实main接线测试）。
- py_compile与git diff --check通过。保留用户已有改动。
- STRIVE：experiments/e1_baseline_recovery_single_scene_20261008_r2；launcher PID 341014，GPU1。
- SAFE-SIM：experiments/e1_safe_sim_recovery_single_scene_20261008_r3；launcher PID 381662，GPU2。
- R2中的SAFE-SIM因set_seed误传tracking在仿真前退出；已修复并保留R2日志，SAFE-SIM独立在R3重测，不重启正在运行的STRIVE。
- 两者使用同一冻结清单索引0，随机种子42，自然结束或最多400帧。每60分钟只读监控，无实质变化不报告，不自动启动正式实验。

## 后续达到有效对照的必要步骤

1. 完整单场门禁：真实执行、生成/预测异常明确、有效EA和可达性、trace与视频齐全。
2. 公共评估统一协议，特别是EA空集合语义和3秒预测来源；碰撞/TTC等真实执行指标与预测代理指标分开。
3. STRIVE地图和时域/坐标转换、原生数据校验及Waymo跨域差异审计；不能为提高碰撞率改变原目标或筛除失败场景。
4. 在冻结场景列表按公开预算继续测试，报告生成失败率和指标有效覆盖率，再评估是否适合250场正式比较。
5. 性能优化优先使用独立GPU按场景并行、减少无变化模型加载、缓存只读特征；涉及重规划预算或求解迭代数的变化应做独立实验并披露，不能混入当前基线成绩。

当前无需用户提供网络、密钥或权重协助。

## 可点击的本地历史视频与诊断图

- [e1_single_scene_smoke_20261008_r1 原R1单场视频](<C:/Users/赵宇轩/Documents/Codex/2026-09-15/riskweaver-api-github-riskweaver-scenario-dreamer/outputs/e1_single_scene_smoke_20261008_r1/safe_sim/scenario_000/scenario_000.mp4>)
- [e1_single_scene_smoke_20261008_r1 原R1单场视频](<C:/Users/赵宇轩/Documents/Codex/2026-09-15/riskweaver-api-github-riskweaver-scenario-dreamer/outputs/e1_single_scene_smoke_20261008_r1/scenario_dreamer/scenario_000/scenario_000.mp4>)
- [原评估缺失原因图](<C:/Users/赵宇轩/Documents/Codex/2026-09-15/riskweaver-api-github-riskweaver-scenario-dreamer/outputs/e1_recovery_20261008/SAFE-SIM原始评估缺失原因.png>)

## 启动验证补充（非最终成绩）

STRIVE首个完整预算计划已完成：official_adv_success=true、official_solution_success=false；trace已有11条状态（1条初始+10帧实际执行）与10条预测，进入下一次重规划。规避解未找到这一事实必须保留，不能将官方优化诊断替代实际自车碰撞统计。SAFE-SIM新重测已有30帧执行记录，任务存活。两者均尚无最终results.json。
