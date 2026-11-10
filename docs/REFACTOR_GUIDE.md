> **版本提示（v1.5.0）**：本文保留 v1.4.0 候选规划架构及奖励/训练设计的历史说明。当前正式动作集合、5°航向候选、75动作/739维观测契约、会遇临界干预窗口与各会遇策略，以 [COLREGS_TACTICAL_WINDOW_V150.md](COLREGS_TACTICAL_WINDOW_V150.md)、`usv_sim/actions.py`、`usv_sim/risk.py` 和 `usv_sim/planner.py` 为准。本文中的 21 候选、253 维观测和 `candidate-v2` 数值不代表 v1.5.0 当前配置。

# v1.4.0 科研避碰框架：实现、公式与实验入口

2026-11-04。对应已实施代码。早期问题与设计推导见 FRAMEWORK_AUDIT_AND_REDESIGN.md；论文与工业一手来源见 COLLISION_AVOIDANCE_RESEARCH.md。本文不把论文结果当作本项目已复现结果。

## 1. 实际运行链

`风险/责任记忆 → 候选轨迹 → 全目标规则及距离检查 → PPO或确定性选择 → 短周期监控 → 执行 → 日志`

`actions.py` 定义21个固定候选：航向偏移 `[0,10,20,35,-10,-20,-35]°` × `[hold,slow,accelerate]`。减速目标为当前速度减1.5m/s，加速为加1m/s，经配置上下限裁剪；达到限速后无效果的重复速度候选被剔除。索引0始终是当前候选基准航向与保持速度，不再隐式表示任何一条锁存策略。

普通避让时以会遇开始的航向为参考，驶过让清后用航线LOS恢复；普通直航情景以责任建立时航向/航速为参考。`rules.py`对每个目标比较同一候选的绝对命令，多船约束冲突会明确无解。会遇关系、让路责任、风险阶段和动作分开显示。

`engine.prepare_decision()`返回当前观测和mask并缓存候选；`engine.advance_decision(action)`调用同一执行链推进最多10秒；`engine.step()`执行2秒控制。目标和本船动力学均使用不超过1秒的积分子步。预测全程每2秒更新航向控制，与实际执行一致，不以10秒粗积分替代控制。

## 2. 规则、硬距离与监控

- 对遇R14：非紧迫时要求相对该目标会遇参考至少右转10°；10°是可调工程判据，不是COLREG统一数值。
- 交叉R15：不强制所有场景右转；预测相对轨迹穿越目标纵轴时，检查是否从船首前方穿过。仍是有限模型下的几何近似。
- 追越R13：保持让清责任，允许左右候选与加速；任何加速候选仍须通过全目标距离检查。
- 直航R17：普通阶段保持责任建立时航向航速；风险分数达到0.65进入可干预近似阶段；紧迫阶段优先防碰。阶段阈值属于工程近似，不冒充法律定义。

候选每个区间计算相对运动线段到原点的最小距离，而非仅检查端点；静态多边形同样检查运动线段，并增加基于艏摇上限的曲率余量。动态硬边界为当前配置的SDA；`SDA_pref`为舒适/操纵余量偏好，不再混为同一safe标志。默认偏好为 `(visibility_factor+0.274)*adjusted_SDA`，这是明确命名的工程策略，不声明为附录A的逐项等价实现。

规划器提供 `position_uncertainty_m` 和 `velocity_uncertainty_mps`，按 `e_p+t e_v` 膨胀目标位置包络；默认0，只表示确定性对照，不能称鲁棒保证。`oracle_prediction=False`默认仅使用目标当前运动；oracle是显式对照开关，不用于默认研究结果。

`safety.py`用最新状态独立检查即将执行的2秒，包含所有当前目标和障碍。它与规划共享动力学，有共因失效可能，不是经过形式证明的冗余安全系统。无可行候选时，恢复动作以 `(预测距离违反量,原因数,代价)` 排序并标记 `minimum_risk`、`safe=False`，不可把恢复动作计作已证明安全。

## 3. 采样式MPC公式

对候选c，控制器把目标航向转换为舵角，动力学滚动预测 `x[k+1]=f(x[k],δ_c[k],u_c,2s)`，只执行首段。

可行集为全目标规则约束、动态SDA与静态障碍约束的交集。可行时确定性基线选择最低J：

\[
J=.4\overline{e_\perp}/1000+.002e_{\perp,N}+.035|\Delta\psi|_{deg}
+.020\sqrt{\overline{\delta_{deg}^2}}+.025|\psi_c-\psi_{prev}|_{deg}
+.08|u_c-u|-.0004\max(0,\Delta s_N)
+2\max(0,-m_{pref,min}/1000)^2.
\]

有限21候选不保证找到所有真实可行轨迹。预测时域默认1000秒，按目标TCPA扩展到最多1500秒；不确定目标意图使长时域结果具有条件性。参数保存在代码的明确配置对象/属性中，训练与推理默认一致。

## 4. PPO观测、奖励与更新

动作空间为21候选，观测253维：本船9维、3个按目标ID排序且含存在标记的目标槽共36维、阶段/实际候选基准等14维、全目标聚合5维、21个候选各9维。候选特征包含全目标和静态障碍检查结果。因此未单独占槽的目标仍影响可行性与候选代价；这不等于完整编码任意目标意图、全部责任历史的完全可观测状态。

网络共享两层128 Tanh主干，actor21维，critic1维；训练与推理复用一个工厂。schema为 `candidate-v2 / candidate-observation-v2`，旧42维/5动作权重明确拒绝，UI显示原因。现有模型文件没有自动迁移或覆盖。

以每10秒为运行成本尺度，令 `q=dt/10`、`p=Δs/(u_nom*10)`、`e=横偏米数`、`d=max_i max(0,1-separation_i/SDA_i)`，separation对于TCPA>0取当前距离与DCPA的较小值，其余取当前距离。

\[
\begin{aligned}
r={}&.08p-q[.02+.10\min(3,e/1500)+.06|\delta|/\delta_{max}
+.10|\Delta\delta|/(\dot\delta_{max}dt)\\
&+.04|r_{yaw}|/r_{max}+.90d+1.5I_{rule}+.25I_{intervention}]
-100I_{collision\ event}+3I_{route\ return\ event}+10I_{goal\ event}.
\end{aligned}
\]

实际代码对分母加小正数保护；碰撞事件包括船舶和静态障碍。整体奖励不裁剪，回归和终点仅发生沿触发。规则成本来自独立审计字段，关闭规则mask时也计算，避免消融同时改变目标函数。分量见 `rewards.py`，累积统计写入checkpoint。

一个策略决策内累积2秒分量奖励，并按时间折扣。PPO的γ=0.99、GAE λ=0.95均对应10秒；不足10秒的末步使用 `γ^(elapsed/10)` 和 `λ^(elapsed/10)`。PPO ratio使用保存的同一mask，clip为0.8～1.2，价值损失系数0.5，熵系数0.01，Adam 3e-4，梯度范数上限0.5。

全部候选无解时仅开放一个明确的恢复槽，actor在该状态没有选择梯度，这是真实无解状态的记录，不能隐藏成PPO能力。训练应同时检查可行候选数、单动作比例和干预率；总reward上涨不足以证明PPO有效。

## 5. 数据与可复现性

`data_split.py`按相对位置100m、航向5°、速度0.5m/s量化生成几何指纹，忽略平移与目标行排序，同指纹案例始终同组。seed7本地268案例划分为train206、val36、test26。近邻落在量化边界两侧仍可能被分开；这是一种明确的近重复启发式，不是彻底消除所有场景族泄漏的证明。

预热默认0轮；可选预热只读train，以相同候选基线为教师，不使用测试标签作真值。checkpoint包含split与几何指纹、schema、CLI配置、控制时钟、网络/优化器、RNG、案例顺序/游标、奖励统计；恢复校验数据与消融配置并跳过预热。保存发生在rollout更新后，不保存船舶回合中间状态，恢复是继续训练而非逐步完全复现中断回合。

## 6. 运行与消融

在项目根目录：

```powershell
# CPU启动与测试；CUDA请在所选解释器配置对应PyTorch
.\.venv\Scripts\python.exe run.py
.\.venv\Scripts\python.exe -m pytest -q

# 保存到新文件，不覆盖原五动作权重
.\.venv\Scripts\python.exe train_ppo.py --device cpu --steps 150000 --out models/candidate_policy.pt

# 相同独立测试集：确定性基线与PPO
.\.venv\Scripts\python.exe evaluate_ppo.py --baseline --seed 7 --split test --out results/baseline_test.csv
.\.venv\Scripts\python.exe evaluate_ppo.py --checkpoint models/candidate_policy.pt --split test --out results/ppo_test.csv

# 两个独立开关，仅用于离线仿真消融；默认均启用
.\.venv\Scripts\python.exe evaluate_ppo.py --checkpoint models/candidate_policy.pt --no-rule-mask --out results/no_rule_mask.csv
.\.venv\Scripts\python.exe evaluate_ppo.py --checkpoint models/candidate_policy.pt --no-monitor --out results/no_monitor.csv
```

训练也支持同名独立开关；对比需固定种子、网络、数据、候选与训练预算。`--split all`是全部案例回归，不能标作独立泛化测试。`--limit`和`--max-steps`支持流程烟雾测试，未到终点会如实标为未完成。

CSV分开报告碰撞事件、安全域侵入时长/裕度、规则审计、到达/超时、无解、干预、单动作决策比例及规划延迟p95。评估成功要求到达且无碰撞/接管/安全域侵入/已实现规则违规，仍仅对应本模型及当前规则实现。

## 7. 仍需后续研究验证的范围

本轮没有做长期PPO收敛或证明其优于确定性基线；没有完整复现标准附录算法，也没有完成正文全部测试要求。当前规则主要针对普通机动船互见开阔水域，特殊船舶责任、完整Rule19、狭水道/分道通航及合作目标行为均不完整。目标目前有常速/脚本运动，尚非会遵守规则的智能目标驾驶员。

动力学为未标定的3DOF替代模型；最小速度默认2m/s，不支持宣称停车/倒车能力。碰撞使用船体包络圆及区间扫掠，未完成精确船体姿态接触模型。无解恢复不具备递归可行性或最小风险最优性证明。计算耗时仅测量，无硬实时调度保证。这些边界需在论文中保留。
