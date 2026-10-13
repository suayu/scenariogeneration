# E1 五组100场正式实验启动记录

日期：2026-10-12（北京时间）。服务器：zhaoyx@172.18.36.182。分支：RiskWeaver-dev。本轮不提交GitHub。

## 实际启动

- 目录：`/home2/zhaoyx/scenario-dreamer/experiments/e1_five_arm_idm_100_modelpool_20261012_r4`。
- Launcher PID：53420。PID文件：`experiments/e1_five_arm_idm_100_modelpool_20261012_r4.pid`。
- 主日志：`work/e1_five_arm_idm_100_modelpool_20261012_r4_launcher.log`。
- 已检查 launcher 存活、phase=formal；画像关/开首分片分别在 GPU0/GPU1执行，其余48个分片任务排队。总共50个分片、每组100场，共500次预定尝试。
- 正式轨迹中已记录 qwen3.8-flash 调用成功及实际状态转移；这不是额外模型切换冒烟。

## 固定配置

五组：RiskWeaver纯动态画像关、RiskWeaver纯动态画像开、Scenario Dreamer、SAFE-SIM适配、STRIVE适配。

同一冻结100场（来源配额30/22/48）、IDM、基础seed42，按冻结编号导出每场种子；自然结束或最多400帧。清单与R2准备的100场完全一致，未按性能筛选。

两组RiskWeaver仅profile_enabled不同：静态障碍关闭、难度控制关闭、单agent高级规划、规划窗口最多4次、扩散求解器固定inner_lr=0.2/inner_beta=0.5/n_guide_steps=2；保留两组相同的动力学修正和锚点引导。每场画像冷启动，使用该场全部已观测历史，禁止跨测试场景/分片记忆泄漏。

## 模型池与故障处理

- 你新增的模型列表已从服务器最新源码读取，原39个活动条目（含重复）在修改时经AST逐项比较保持一致，随后仅允许故障驱动自动注释。
- 首选：qwen3.8-flash。冻结文本规划候选：26个，候选是否真正可调用由服务返回判定，并非预先宣称全部有效。
- OCR、翻译、数学、音频/实时等10个专用接口不进入本轮文本JSON规划池，原名单保留；详情见model_pool.json。
- HTTP404：在FREE_MODEL_POOL中注释该模型的所有重复活动条目，带“百炼模型404”永久标记；下个请求/新进程均跳过。
- 明确免费额度耗尽：保留原永久停用规则，注释所有重复活动条目并回退。普通429限流和临时服务错误不永久注释。
- 多GPU共用文件锁和原子替换，防止并行进程覆盖彼此停用记录；模型池只允许冻结池内回退，不因全部停用偷偷恢复整池。
- 单次HTTP请求超时60秒，关闭SDK隐式重试。每个规划窗口的实际API尝试数可能大于1，全部记录于planner_trace；每次尝试成功模型和耗时继续保存。
- 如果所有候选都不可用，按planner_service_failure停止后续调度，不当成正常“不攻击”。失败和部分执行结果完整保留。
- 每次LLM输出trace记录attempted_models/successful_model/model_failures；attempt_result汇总成功和尝试模型频数，最终按组给出成功模型分布。
- 免费额度保护沿用百炼平台设置；代码的故障识别与回退不替代平台的免费保护开关，不自动启用付费。

本轮是**共享候选池比较**，不是固定单模型比较。模型切换可能与运行顺序/额度相关；最终应披露模型分布并做模型匹配的敏感性分析，不能把全部差异仅归因于画像。

## 验证与复用冒烟

93项相关测试通过：模型404回退、免费额度耗尽回退、普通限流不永久停用、重复条目、多并发写入、跨进程永久标记、池耗尽服务失败、公共几何/指标/分母/执行测试。py_compile与git diff --check通过。

没有重复三基线冒烟；复核R2的场景哈希、关键组件哈希、实际解析配置、退出码、视频和trace完整性后复用。SAFE-SIM可达性继续使用CtRL-Sim代理未来，执行为diffusion；STRIVE仍为Waymo适配而非原论文完整复现。不同预测来源的D不能直接混合排名。

R3在执行任何场景前，因为复用目录的相对路径与旧配置绝对路径不一致退出。保留R3主日志和终态launcher_result；修复Path.resolve后重新通过复核，再在全新R4目录启动。没有删改旧实验。

## 代码修改

1. `policies/llm_adversarial_planner.py`：404永久停用、重复项注释、并发写回保护、其他worker停用项刷新、显式请求超时、全部停用服务失败。
2. `experiments/comparison/e1_protocol.py`：公共候选池、专用接口过滤、两组一致配置与模型池协议。
3. `experiments/comparison/run_e1_group.py`：接受冻结池、限定池内回退、每次尝试模型频数。
4. `experiments/comparison/launch_e1_five_arm.py`：传递共同池、存档源代码快照、复核复用冒烟、最终模型分布、启动器异常落盘。
5. `tests/test_e1_model_pool_fallback.py`：故障与并行更新验证。

修改前备份：`work/e1_formal_20261012_r3_backup`。运行时初始源码快照：实验目录source_snapshot。已有用户修改保留。

## 本机入口

- [冻结100场](frozen_100_scenes.json)
- [模型池](model_pool.json)
- [协议](protocol.json)
- [启动参数与代码哈希](launch.json)
- [复用冒烟验收](smoke_acceptance.json)
- [预检记录](e1_formal_20261012_r3_preflight.json)

每60分钟只读监控，仅在组完成、终态服务失败、运行故障或需要用户操作时通知。终态收集全部视频、JSON、CSV与trace到本目录并核验哈希；不筛选失败尝试。QQ未配置，不请求、不发送。
