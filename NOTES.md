# 需要用户协助的事项（置顶，2026-10-12正式启动更新）

- 当前五组100场正式任务已启动，无需提供密钥或权重。旧的单模型确认待办不再作为本轮模型池启动门槛。
- 本轮按用户更新的FREE_MODEL_POOL运行，保持仅用免费额度约束；404/明确免费额度耗尽会永久停用并回退。若所有候选均失效，才需人工补充可用模型并在百炼控制台维护对应免费额度保护；客户端不替代平台开关、不自动启用付费。
- 当前任务：experiments/e1_five_arm_idm_100_modelpool_20261012_r4，launcher PID 53420；同名experiments/*.pid；work/*_launcher.log。
- 五组共享冻结100场、IDM、seed42、自然结束或最多400帧。R3为配置相对路径校验失败、执行0场的保留记录；R4已修复并实际进入formal阶段。
- 每60分钟只读监控，仅有实质成果、故障或需操作时通知。没有提交GitHub。以下旧状态仅为历史记录。

---

# 需要用户协助的事项（置顶，2026-10-12更新）

- 三基线冒烟正在运行，无需用户提供下载、权重或密钥。
- 100场中的两组 RiskWeaver 拟固定使用 qwen3.8-flash（已由服务端只读模型列表确认可用）。用户已确认的 qwen-plus-0112 免费保护对应模型在当前服务端返回404；用户允许任意更换模型并免额外模型切换冒烟。新模型的免费额度保护无法通过模型目录/API Key核实：正式调用前需有该模型已开启“免费额度用完即停”的明确记录。不要提供密钥，不自动放开付费，不用已耗尽模型。
- 最新实现、验证与运行说明：docs/E1_FIVE_ARM_100_20261012.md。
- 旧日志/产物保留；没有提交GitHub。以下为历史记录，旧状态不代表当前结论。

---

# 需要用户协助的事项（置顶，2026-10-08更新）

当前没有需要用户手动解决的下载或凭据问题。STRIVE官方权重已下载到工作区并校验，
本机多余权重副本已清理；无需重新提供权重或密钥。

# 当前任务

- STRIVE实现说明与复用命令：`docs/STRIVE_BASELINE.md`。
- DiffScene按用户要求暂不实现。
- 保留既有改动、日志与实验产物；本次不提交GitHub。
- 长实验每60分钟只读检查，无实质变化不报告。

# E1当前工作与科研门禁

- 三种基线各一场闭环冒烟：`experiments/e1_three_baseline_single_scene_smoke_20261008_r1`。
- SAFE-SIM实际执行不变，公共可达性可选择`ctrl_sim_rollout`或原`native`路径；不静默回退。
- STRIVE仍需完成闭环有效性与Waymo地图/CV初始化跨域审计；不能将适配版标为原论文完整复现。
- 旧的网络失败日志保留，STRIVE源码说明：`docs/STRIVE_BASELINE.md`。
- 统一单场冒烟与正式门禁：`docs/E1_BASELINE_WORKFLOW.md`。


## 2026-10-08基线修复与独立重测

- 原R1已结束：SAFE-SIM生成与执行成功，但64帧联合预测因FOV不完整、65帧EA因空近场集合无效；STRIVE初始化cuDNN反向失败。
- 修复详情与组件验证：`docs/E1_BASELINE_RECOVERY_20261008.md`。
- EA新协议`ea_empty_set_zero_v2`统一应用；完整输入无近场威胁EA=0，真正缺失/求解失败仍无效，D无有效帧仍null。旧结果不覆写。
- `--forecast-tracking persistent`仅评估副本持续跟踪源帧车辆，默认`observation`与`native`回退保留。
- STRIVE保留官方优化目标/eval语义，隔离worker用可导原生RNN和统一工厂设备。小预算真实策略反馈链路通过，尚未验证完整闭环。
- STRIVE单场：`experiments/e1_baseline_recovery_single_scene_20261008_r2`，75次初始化/300次对抗优化。
- SAFE-SIM单场：`experiments/e1_safe_sim_recovery_single_scene_20261008_r3`。R2的SAFE-SIM启动参数错误日志保留，R3独立重测，不中断STRIVE。
- 两者同一冻结索引0、种子42、自然结束或最大400帧；每60分钟只读监控。没有通过闭环及跨域保真审计前不自动启动250场正式实验。
