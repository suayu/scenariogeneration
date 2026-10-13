# CommonRoad-Reach、反应余量与 E1 运行恢复（2026-10-13）

## 已交付与使用方式

- 原 `RouteReachability` 仍保留，默认 `sim.evaluation.reachability.backend=route_grid`。
- 新增 `commonroad` 后端调用官方 CommonRoad-Reach 2025.2.1 的 C++ 多面体传播实现，并非只复刻接口。主仿真进程通过带 request_id 的 JSON 与独立 worker 交换输入、输出。
- 独立环境：`/home2/zhaoyx/venvs/riskweaver-commonroad-reach-2025.2.1/bin/python`。依赖锁：根目录 `requirements-commonroad-reach.lock.txt`。commonroad-io 固定为 2024.3；最新 IO 的形状 API 与该 Reach 版本不兼容。
- 显式切换：在现有 Hydra 仿真命令后追加 `sim.evaluation.reachability.backend=commonroad`。独立 Python 路径和 timeout 已在 `cfgs/sim/base.yaml` 配置。worker 失败、缺失联合预测、身份/掩码/时域校验失败不静默回退，也不计作有效零危险度。
- Scenario Dreamer 的 E1 runner 在 `--evaluation-source method_default` 下启用隔离 CtRL-Sim 滚动联合预测。`--evaluation-source native` 可恢复旧分支。SAFE-SIM 的实际 diffusion 执行与 CtRL 代理评估未来仍明确区分。
- STRIVE 默认每场景累积优化预算 300 秒，跨初始化、对抗优化、解优化和重规划共享；预算在下一次迭代前检查，完成当前更新与最终解码。它不是整个场景的严格 300 秒墙钟超时。原进程 watchdog 仍保留。
- STRIVE 分片异常只停止该组待启动分片，其他方法继续，STRIVE 冒烟失败也不阻断其他已通过组的正式阶段；失败条目、未尝试条目保留，不能算正常成功。共享 LLM 服务全局故障与其他运行器故障仍按原规则处理。

## 环境与代码保护

原 scenario-dreamer 环境从未安装、升级或移除包，因此不需要修改或完整复制该环境。修改前后 `pip freeze` 字节一致，记录位于 `work/commonroad_integration_20261013/original_environment_before.txt` 和 `original_environment_after.txt`。这两个记录是依赖审计，不是完整环境备份。如果后续必须修改原环境，仍须先做完整备份。

原代码在 `work/commonroad_integration_20261013/code_before/` 保留。原 STRIVE 官方仓库与优化损失代码没有修改；worker 仅在自身子进程替换官方函数模块的进度迭代器以实施预算。原日志、失败尝试、权重和实验均保留。

## 可达面积的几何语义与限制

两后端均以路线坐标中的位置可达面积作为指标，但不是相同数值协议：

1. 原方法枚举离散纵横加速度，以位置/速度网格合并状态，使用现有道路与碰撞规则。
2. CommonRoad 采用连续加速度区间、多面体状态传播及障碍裁剪。网格大小 1 m；保留全部中间时刻而不剪掉无法连接终点的中间节点。
3. 现有场景没有完整真实道路多边形，适配器以中心线及固定半宽构造道路代理；不是恢复了真实 Waymo 路面边界。
4. CommonRoad 自车采用外接圆膨胀，原方法的道路中心约束和 OBB 障碍处理与它不完全相同。新面积是曲线坐标面积，未作曲率 Jacobian 校正。
5. 添加道路外侧厚禁行带：避免薄边界跨投影域时被官方坐标变换遗漏。障碍裁剪至相关路线邻域，以防全部顶点在域外时丢失实际封闭区域；多边形包装成官方支持的 ShapeGroup。
6. 采样动态占据、近似道路、状态外近似和离散时间意味着面积不是连续时间无碰撞保证，更不是某个具体自车策略一定能规避的见证。

因此新旧面积、D 不可不加标记混用。下一步应在相同道路足迹、碰撞膨胀、加速度边界、坐标与分辨率下做收敛和准确性验证，再决定是否替换主评估器。当前默认不变。

## 两场景比较（全部采样点保留）

目录：`experiments/commonroad_two_scene_diagnostic_20261013_r2`。使用既定冻结清单最前两个场景，全局索引 1 和 3，各执行 21 步，在 0/10/20 步比较。不是完整路线危险性对比实验，也没有挑选较好场景。

| 场景 | 帧 | 两计算器共同输入 | 原面积 m² | CommonRoad 面积 m² | 原耗时 s | CommonRoad 耗时 s | D |
|---|---:|---|---:|---:|---:|---:|---:|
| 1 | 0 | CV，仅计算器诊断 | 59.5 | 0 | 3.033 | 1.962 | null |
| 1 | 10 | 完整联合滚动未来 | 123.5 | 0 | 7.260 | 0.249 | 0.298 |
| 1 | 20 | CV，仅计算器诊断 | 112.0 | 0 | 9.491 | 0.194 | null |
| 3 | 0 | CV，仅计算器诊断 | 66.5 | 73.582 | 1.373 | 0.191 | null |
| 3 | 10 | CV，仅计算器诊断 | 67.5 | 86.049 | 1.645 | 0.332 | null |
| 3 | 20 | 完整联合滚动未来 | 97.0 | 89.924 | 2.403 | 0.192 | 0.047 |

四个采样点联合未来掩码不完整，D 保持 null。共同 CV 只用于比较计算器，绝不冒充有效动态 D。表中有效 D 来自保留的原方法，而非 CommonRoad。首个 CommonRoad 耗时含 worker 启动。零面积导致提前终止，因此不能由本表宣称通用加速或准确性提升。场景 1 的分歧尚未定位到唯一原因，需要进一步校准上述不同约束。

两个场景均保存视频与状态/预测 trace。R1 的缺失预测失败记录也保留，R2 明确记录诊断输入类型。

## 闭环支路验证

### Scenario Dreamer

`experiments/scenario_dreamer_rollout_branch_smoke_20261013_r1`：20 个实际执行步，20/20 帧有效 D；D=0.026587，平均三秒终端可达面积 137.9 m²，无生成/运行/预测服务失败。隔离未来不改变实际环境、策略历史或 RNG；相关回归验证状态、模型/RNG 隔离。原 R4 缺失联合预测的旧结果不能直接补写 D，应重新评估并区分协议版本。

### STRIVE

`experiments/strive_budget_closed_loop_smoke_20261013_r1`：原故障场景 0_7.pkl，13 个背景车辆；20 步测试。

- 75 次初始化、32 次对抗迭代、0 次解优化；预算触发截断，优化计时 315.593 秒，整次场景耗时 582.016 秒。
- 1 个计划，20 帧实际攻击；20/20 有效 D，D=0.231926，最小 OBB TTC=2.271 秒，平均 EA=0.267681 m/s²。
- 自车碰撞 0、背景车间与背景静态碰撞均 0；生成/运行/预测服务失败均 0；视频和 trace 齐全。
- 官方 adversarial success=false，不宣称本次制造了碰撞，也不宣称完整复现原论文。仍有 Waymo 道路代理、CV 初始化、nuScenes 学习分布迁移和解优化未运行等限制。
- 剩余预算为零时仅使用尚在有效预测时域内、车辆集合一致的缓存。缓存过期或参与者变化按 generation_failure 记录，不能外推拼出未验证未来。任何截断候选仍须通过背景联合几何安全筛选。

另有 `strive_budget_diagnostic_20261013_r2` 仅优化候选，实际执行为零，不与上述闭环计数合并。

## 反应余量记录

原与新后端均输出从 0.5 秒到预测终点的 `area_curve`、`minimum_area_m2`。不把 t=0 单点的自然零面积计入最小值。

每条 reachability event 增加原/攻击面积曲线、未来最小面积、backend、同未来时刻剩余面积比与压缩度 `1-A_attack/A_baseline`。基线为零时比值为 null，不人为设置成零或一。

按同一计划的 source_step/target/strategy 聚合，记录两种不同时间轴：

- 攻击执行时间上的三秒终端面积变化；取 `minimum_terminal_area_during_attack_m2`。
- 每个执行帧的未来时间轴面积曲线；取 `minimum_future_area_during_attack_m2`。

面积有效与 D 有效分开：EA 无效不抹去已经成功计算的面积；缺失面积保留 null，缺失窗口不丢弃。面积压缩描述理论响应空间，不能单独命名为测得反应时间。后续可联合反应延迟、EA、TTC、余量持续时间作完整反应余量向量。

本轮用户展示图另用历史 R4 RiskWeaver 画像关的场景 003 首个真实接受计划（source_step 0..10）演示攻击执行时间曲线；它是历史观测，不是新运行的攻击对照。英文 PNG/SVG 同目录保留。

## 关键源码与验证

- `policies/commonroad_reachability.py`：隔离进程桥与输入验证。
- `experiments/comparison/commonroad_worker.py`：官方后端适配。
- `policies/risk_metrics.py`、`formal_metrics.py`、`reaction_margin.py`：面积曲线/最小值/逐攻击聚合。
- `run_e1_group.py`：Scenario Dreamer 默认预测支路、STRIVE 累计预算缓存处理。
- `optimization_budget.py`、`strive_official_worker.py`、`policies/strive_adapter.py`：完整迭代截断与审计。
- `launch_e1_five_arm.py`：STRIVE 组级失败隔离。
- `tests/comparison/test_reaction_margin_and_budget.py`：预算、面积、null、组隔离、累计预算。
- `tests/comparison/test_commonroad_native.py`：实际官方后端空走廊、静态封闭、动态封闭。

54 项相关原环境回归通过；独立环境 3 项实际 CommonRoad 测试通过；独立环境 pip check 通过。所有修改源码编译及 git diff --check 后再提交。本轮不自动启动 100 场正式实验。

本地完整入口：`D:/RiskWeaver/outputs/commonroad_reach_integration_20261013/index.html`，含四个视频、原始 JSON/trace、CSV、中文图、英文备用图和逐文件 SHA-256 校验清单。

## 上游参考

- https://github.com/CommonRoad/commonroad-reachable-set
- https://commonroad.in.tum.de/docs/commonroad-reach/

项目新增环境与上述适配不构成新可达性算法贡献；论文应明确引用上游，并说明道路/预测/模型假设与共同协议。
