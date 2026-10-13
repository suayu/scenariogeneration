# E1基线评估流程（2026-10-08）

## 固定协议

Scenario Dreamer、SAFE-SIM、STRIVE各只执行一个冻结场景的闭环冒烟；自然终止或最多400帧。
不另开四场冒烟。合成张量、模型加载、环境隔离属于组件测试，不计入场景实验。
DiffScene按用户要求不实现。正式250场不由冒烟或监控自动启动。
当前冒烟选冻结清单索引0，三个方法使用同一场景内容和种子42；不得依据输出换场景。

## SAFE-SIM可达性评估的新支路

1. 真实环境在源帧计算EA和公共CV基线可达面积。
2. SAFE-SIM照常生成、筛选并暂存diffusion轨迹，实际执行后端保持不变。
3. 复制环境动态状态、自车策略和CtRL-Sim运行状态。共享只读参数，复制模型模块与缓冲区。
4. 从同一历史与当前状态开始，每0.1秒采样所有活跃参与者的动作，在同一副本世界连续滚动30步。
   不能将各车辆的不同联合样本拼接成一份预测。
5. 全部预测带稳定ID、全局位置/速度/航向、有效掩码、源帧、种子、采样ID、条件自车策略和耗时。
6. 公共评估器仅在浅副本中替换待评估的联合轨迹；真实环境继续消费原diffusion轨迹。
7. 源帧、活跃ID、至少3秒时域、有效掩码和有限值全部通过才计算可达性。
   短时域、离开视野导致未来缺失、模型异常等保留具体无效原因，不用CV补成有效D。
8. 仅EA和可达性同时有效的帧参与D；没有有效帧严格D=null。

该预测是SAFE-SIM未来的代理模型估计。diffusion历史没有CtRL动作/RTG token，
动作通过已发生状态的逆运动学重建，历史RTG按最大值占位；这一差异在metadata披露。
采样结束恢复CPU/CUDA、NumPy和Python随机数流，不改变真实运行的随机样本序列。
记录真实diffusion联合预测及CtRL预测，额外记录两者首步位置差，不能把代理D当作已验证真值。
理论可达性需要完整时域，副本在假想终止后继续传播；这些帧不计为真实执行。

## 参数与回退

原实现保持为默认`native`，仅统一冒烟的SAFE-SIM显式选新支路：

```bash
python experiments/comparison/run_passive_baseline.py \
 --method safe_sim --manifest YOUR_FROZEN_MANIFEST \
 --steps 400 --max-scenarios 1 --reachability-source ctrl_sim_rollout \
 --output-dir experiments/SAFE_SIM_NEW_RUN
```

将`ctrl_sim_rollout`改为`native`即可显式回退。使用新产物目录，不覆盖旧结果。
Scenario Dreamer也可显式选新预测支路；原`native`没有联合未来，可达性和D保持缺失。
正式比较D时需要统一预测协议，不能混用代理D和原模型D而不披露。

## 单场统一入口和产物

`experiments/comparison/run_e1_smoke.py`默认在GPU1/2并行运行Scenario Dreamer和SAFE-SIM，
Scenario Dreamer结束后STRIVE复用GPU1。每方法仅一场；保存命令、清单哈希、配置、PID、
日志、视频、attempt_result、execution_trace、分阶段和最终门禁结果。
子进程退出0但零实际执行、生成失败、SAFE-SIM新支路预测失败或无有效D、视频/trace缺失，
均不能通过冒烟。失败尝试保留，不能筛掉再声明成功。

STRIVE使用官方固定源码、真实严格加载权重、原优化函数与真实自车策略反馈。
Waymo车道走廊语义代理和CV未来初始化是适配差异，需要跨域验证，formal_ready仍为false。

## 正式实验前仍需确认

- 三种方法单场闭环门禁结果与STRIVE跨域保真性。
- CtRL评估支路的真实闭环耗时、预测误差和有效覆盖率。
- 更换自车策略时提供可隔离策略副本，使用同一清单、预算与种子。
- 正式指标统一EA、TTC、可达面积、道路偏离/完成、碰撞速度口径；披露失败和覆盖率。
- 所有长实验每60分钟只读检查；没有实质变化不通知。


## 2026-10-08基线修复与独立重测

- 原R1已结束：SAFE-SIM生成与执行成功，但64帧联合预测因FOV不完整、65帧EA因空近场集合无效；STRIVE初始化cuDNN反向失败。
- 修复详情与组件验证：`docs/E1_BASELINE_RECOVERY_20261008.md`。
- EA新协议`ea_empty_set_zero_v2`统一应用；完整输入无近场威胁EA=0，真正缺失/求解失败仍无效，D无有效帧仍null。旧结果不覆写。
- `--forecast-tracking persistent`仅评估副本持续跟踪源帧车辆，默认`observation`与`native`回退保留。
- STRIVE保留官方优化目标/eval语义，隔离worker用可导原生RNN和统一工厂设备。小预算真实策略反馈链路通过，尚未验证完整闭环。
- STRIVE单场：`experiments/e1_baseline_recovery_single_scene_20261008_r2`，75次初始化/300次对抗优化。
- SAFE-SIM单场：`experiments/e1_safe_sim_recovery_single_scene_20261008_r3`。R2的SAFE-SIM启动参数错误日志保留，R3独立重测，不中断STRIVE。
- 两者同一冻结索引0、种子42、自然结束或最大400帧；每60分钟只读监控。没有通过闭环及跨域保真审计前不自动启动250场正式实验。
