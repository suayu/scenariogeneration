# STRIVE接入记录（2026-10-08）

官方源码：<https://github.com/nv-tlabs/STRIVE>，固定commit
`b708951f8665c97a1de9ed93b4ed3f58dd8cbf5d`。
服务器独立源码目录：`/home2/zhaoyx/STRIVE-riskweaver`。
不把SAFE-SIM内的StrivePlanner（自车规划器）误认成STRIVE危险场景生成器。

## 状态与正式实验门禁

当前实现包含官方模型、原始初始化/对抗/解优化函数的调用入口、nuScenes原生入口、
Scenario Dreamer输入转换、真实自车策略反馈、联合轨迹执行和审计。
只有真实官方权重严格加载、原生网络兼容性诊断和冻结场景闭环冒烟通过后，才可称STRIVE可用。
`formal_ready=false`是有意保留的审计标记；不得用随机权重或任意其他checkpoint替代。
2026-10-08已通过本机代理下载官方权重，ZIP CRC和上传SHA-256校验通过。
权重位于工作区`checkpoints/strive_official_20261008/traffic_model.pth`；真实模型诊断通过。

## 与原方法的区别

官方网络和损失函数源码保持原样，运行子进程隔离顶层models/datasets/utils包。
官方源码使用已移除的NumPy别名，兼容别名仅在子进程中注入。
使用当前Torch2.2/PyG2.6运行兼容环境；仍需用官方权重验证数值兼容性。
官方步长0.5秒、4个历史点、12个未来点（6秒）保持原值。执行时在真实当前状态与未来点间
插值至0.1秒，航向先展开，速度从位置差分。插值可能引入速度折点，不能宣称保持官方连续动力学。
背景OBB联合筛选检查预测端点/中点，执行前还检查真实起点/首点/中点，不能宣称连续安全保证。

Waymo适配使用3.5米宽车道走廊作为可行驶层；停车区、道路分隔线、车道分隔线为空，
不是等价的nuScenes语义地图。当前未来初始轨迹采用CV，不读取未来真值；不同于原版从
真实未来拟合初始化。以上差异必须随结果披露并通过跨域验证，不允许把该适配结果标成原论文复现。

官方闭环优化中的planner回调被接入当前自车策略。每轮对候选背景轨迹执行独立环境/策略副本，
返回真实策略响应供官方优化器拟合。新进入视野车辆在回调时使用CV。回调不会写入真实执行日志。
未来大模型策略可提供`clone_for_rollout(env)`接口以共享只读网络权重并复制策略状态。

保留官方可行攻击种子检查、对抗优化和成功后的solution优化；solution结果作为诊断，
不筛除失败尝试，不作为额外执行放行门槛。

## 可复用入口

真实权重位于服务器`/home2/zhaoyx/scenario-dreamer/checkpoints/strive_official_20261008/traffic_model.pth`。
确认其来自官方README链接，记录本地SHA256（作者未提供的哈希不能作为作者真实性证明）。

冒烟仅对Scenario Dreamer、SAFE-SIM和STRIVE各运行一场冻结场景，直到自然终止或400帧。
原生网络、严格加载和反向传播采用合成张量诊断，不额外执行原生nuScenes场景冒烟。
统一入口（输出目录必须不存在）：

```bash
cd /home2/zhaoyx/scenario-dreamer
/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python \
 experiments/comparison/run_e1_smoke.py \
 --manifest experiments/scenario_dreamer_ldm_expand250_20260928_r1/riskweaver_frozen_250_scenes.json \
 --checkpoint checkpoints/strive_official_20261008/traffic_model.pth \
 --output-dir experiments/e1_single_scene_smoke_NEW_RUN
```

SAFE-SIM默认使用CtRL-Sim隔离滚动预测评估可达性；实际执行仍为diffusion。
加入`--safe-sim-reachability-source native`即回退至原评估支路。不得在同一次运行中静默回退。
Scenario Dreamer冒烟保留原预测缺失口径，D=null；可以另用被动基线入口显式选择联合滚动预测。

`--indices all`复用250场；多GPU通过互斥索引分片，同一配置和种子。
`--policy rl`复用RL策略；其他策略通过统一Policy接口及clone_for_rollout接入。
每次新建产物目录，保存配置、输入/输出、官方优化日志、预测、实际执行trace、视频、单场结果、
生成失败和exit_status。其他异常显式终止该运行，不自动重试或覆盖。

单场闭环冒烟未通过时禁止启动250场。正式对比需要事后公共指标、碰撞相对速度和道路偏离口径的
统一审计，本入口目前记录自车碰撞速度，不能将其冒称相对碰撞速度。

## Scenario Dreamer的联合预测缺口

CtRL-Sim目前在Simulator.step中采样下一步动作，prepare_background_traffic返回None。
瞬时EA仍可用，但FormalRiskObserver将可达性置缺失。D要求EA和可达性同时有效，因此D=null。
所需联合预测必须从当前真实状态和共同历史出发，包含所有当前活跃非自车智能体的稳定ID、
全局位置/航向/速度、有效掩码、source_step，以及共同未来时标。至少覆盖现有3秒可达性时域，
当前0.1秒执行需至少30点；首点是t+dt。不能把各车辆来自不同联合样本的轨迹拼接成一份场景。
可以对CtRL-Sim环境与模型状态做隔离副本，连续生成整个联合未来；重规划时仅在当前时刻计算
一次，然后只执行首步。必须记录自车条件策略、采样ID和种子，以及预测与实际执行差异。
新增`ctrl_sim_forecast.py`实现隔离滚动预测；不更改危险度定义，不为缺失预测填0。
完整工作流程与适用限制见`docs/E1_BASELINE_WORKFLOW.md`。

## 本次验证结果

27项相关pytest通过（20项适配与公共指标测试、7项后端相关回归），py_compile和git diff --check通过。
官方网络采样、初始化优化和闭环对抗优化反向传播通过随机参数结构诊断；不属于预训练模型实验。
冻结场景0上的10帧IDM反事实副本运行通过，真实状态和策略计数未改变。
原生预检查因官方权重缺失退出码1。尚未完成真实权重原生冒烟、真实闭环攻击或250场正式评估。
验证报告：work/strive_implementation_validation_20261008.json。

## 2026-10-08更新验证

31项公共评估、适配和冒烟门禁测试通过，py_compile与diff检查通过。
官方真实权重严格加载、采样及初始化/对抗反向传播诊断通过。
CtRL-Sim真实30帧联合未来隔离诊断通过，首次2.59秒，执行环境/模型/策略/RNG保持不变。
三个方法各一个场景的闭环冒烟已启动；尚不能据此宣称STRIVE正式可用或完成250场。
服务器校验回执：`work/strive_official_transfer_verified_20261008.json`。
