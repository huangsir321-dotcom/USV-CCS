# PPO 战术权限门控与航线回归保护（v1.6.2）

## 1. v1.6.1 的问题

此前 `SimulationEngine.step()` 在每个策略决策时刻，只要 PPO 权重已加载且 ODD 未降级，就调用 `policy.predict()`。`prepare_decision()` 的动作掩码主要依据 MPC 可行性和 COLREGs 规则筛选，不区分 `monitor`、`maneuver` 和 `route_recovery` 阶段。

因此存在两个通道问题：

1. **过早偏航**：即使风险触发器尚未启动 `maneuver_active`，PPO 也能从所有可行候选中选择非零航向偏移。风险评估触发时间与 PPO 实际改变航向的时间被混为一谈。
2. **回航被覆盖**：回航候选的 MPC 评分虽已加入横向误差收敛项，但 PPO 可以忽略候选分数，只要该候选通过硬可行性掩码就能选中，导致不能沿 MPC 设计的回航方向持续收敛。

## 2. v1.6.2 的控制权限

| 仿真阶段 | 控制选择者 | PPO 权限 |
|---|---|---|
| 正常监视 `monitor` | 规则 + DWA-MPC 基线 | 关闭，只允许基线候选 |
| 避碰机动 `maneuver_active` | PPO + COLREGs 掩码 + MPC 可行性检查 | 开放 |
| 航线回归 `route_recovery_active` | 回航目标函数 + DWA-MPC 基线 | 关闭，只允许回航基线候选 |
| 已回归后的巡航 | 规则 + DWA-MPC 基线 | 关闭；新风险事件触发后重新开放 |
| ODD 降级 | 确定性规则/安全回退 | 关闭 |

当 Rule 17 直航船达到已配置的干预风险阈值，或风险评估进入紧迫局面时，发动机将其纳入 `maneuver_active` 机动窗口。正常监视阶段仍按共享风险触发条件保持基线行为。避碰启动、避碰结束并进入回航、以及回航期间有新危险目标加入时，均会清除旧的 10 s 策略保持指令并立即重新规划，不会让旧 PPO 航向继续保持到下一个策略周期。

## 3. 训练与推理一致性

`prepare_decision()` 在得到 MPC 的 `baseline_action` 后生成候选掩码。若 PPO 当前没有战术权限，掩码仅允许 `baseline_action`；在活动避碰机动窗口内，继续使用每个候选原有的 COLREGs/几何/安全可行性掩码。

训练脚本调用相同 `SimulationEngine.prepare_decision()` 和 `advance_decision()`，因此监视/回航阶段不会采样一个部署时被门控禁止的动作。

## 4. 1T-003 初始阶段的对照验证

上传的 1T-003 场景初始风险评分约 0.06，低于当前规则策略机动触发阈值 0.12。v1.6.2 用一个持续请求“左/右 60°”的假 PPO 执行测试：

- PPO 在该阶段没有被调用；
- 唯一允许的候选为 MPC 基线动作（保持航向/基准航速）；
- PPO 测试引擎与未加载 PPO 的规则引擎连续 5 个 2 s 控制步的位置和航向完全一致。

这证明在该案例的初始低风险窗口内，加载 PPO 不会提前引入额外偏航。实际用户权重文件未包含在交付包中，因此不能据此声称已经测试了用户特定 checkpoint 的完整行为。

## 5. 航线回归保护

驶过让清并满足避碰解除条件后，发动机进入 `route_recovery_active`。此时动作掩码只允许 `DWAMPCPlanner` 通过回航目标函数选出的基线候选，PPO 网络不参与动作选择。原有回航完成条件保持不变：横向偏差不超过 `0.1L`、航向误差不超过 `8°`、航速误差在容差内，并连续稳定 10 s。

如果回航期间又出现新的紧迫/让路/直航干预事件，新的风险事件可以中断回航并重新开放避碰机动；待新目标驶过让清后再次进入回航。

## 6. 日志与 GUI 诊断字段

- `policy_source`：`ppo_active_encounter`、`rule_mpc_monitor`、`rule_mpc_route_recovery`、`rule_mpc_post_recovery`、`rule_baseline_odd_degraded` 或 `training_override`。
- `policy_action_authorized`：当前风险/阶段是否开放 PPO 战术动作。
- `policy_gate_reason`：关闭/开放权限的具体原因。
- `maneuver_start_time_s`：最近一次避碰机动启动时刻。

这些字段可用于 PPO 与非 PPO A/B 评估，比较风险触发时刻、首个实际改向指令、回航启动时刻和回航完成时刻。

## 7. 权重兼容性

本次没有更改 `candidate-v4` 动作数量、`candidate-observation-v4` 观测维度或 Actor-Critic 网络层结构。v1.6.1 的 candidate-v4 `.pt/.pth` 文件仍满足结构校验，可直接加载；无需仅因本次门控修改而重训。训练时会自动采用新的阶段掩码。更早 candidate-v3 或旧五动作权重仍然不兼容。

本实现是软件在环研究策略，不构成实船安全保证或 COLREGs 完整合规认证。
