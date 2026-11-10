# v1.6.0 速度包络与 1972 COLREGs 规则策略说明

## 1. 速度策略

任务启动时，`SimulationEngine.cruise_speed_mps` 保存本船初始纵向速度 V0。每次 MPC 候选速度固定锚定到本次 encounter reference speed，不随重规划结果累计变化。战术速度目标按 `V0` 计算，并受 `ShipConfig.min_speed_mps` / `max_speed_mps` 二次裁剪：

- lower = max(config.min_speed, 0.5 × V0)
- upper = min(config.max_speed, 1.5 × V0)
- candidates = hold, slow_25, slow_50, accelerate_25, accelerate_50

当本船初始速度为 8 m/s，且物理速度上下限不进一步收窄时，五个目标分别为 8、6、4、10、12 m/s。该 ±50% 是航速设定值的相对包络，并不意味着 2 秒控制周期允许瞬时改变 50%。

真实纵向运动由 `Simplified3DOFDynamics` 处理：一阶速度滞后 `speed_tau_s`，并使用 `max_accel_mps2`、`max_decel_mps2` 限制每秒速度变化。`ShipConfig.for_length()` 按 50–500 m 船长调整响应时间与加减速度上限；这些是可调的研究型代理参数，不能替代具体船舶试验标定。

## 2. 1972 COLREGs 行为映射

### Rule 13 — Overtaking

- 追越识别采用从被追越船正横后大于 22.5° 的几何近似；对是否追越有疑问时保守维持追越身份的状态记忆，直到驶过让清。
- 追越船保持让清义务；不会因方位变化自动转换为 Rule 15 交叉相遇。
- 规则本身没有规定统一的左舷/右舷追越侧。工程采用相对几何产生 pass-side preference，再由 MPC 对双侧轨迹和速度组合做安全筛选。

### Rule 14 — Head-on

- 近似正前方且航向近反向时识别对遇；对局面有疑问时采取保守识别。
- 碰撞风险窗口激活后，规则筛选要求本船显著向右转，禁止零转向或左转候选。
- 开阔水域 20° 仅是 GB/T 46851-2025 Annex A Table A.1 初始改向参考，不是 Rule 14 写明的强制角度。精确角度仍由候选轨迹与安全距离约束确定。

### Rule 15 — Crossing

- 他船在本船右舷并构成碰撞危险时，本船按让路船处理。
- 预测相对轨迹检查是否从对方船首前通过；在有安全船尾通过候选时，船首通过候选被排除。
- 条文中“如当时环境许可”意味着并非无条件固定向一侧转；当无法找到安全可行候选时，规划器会报告最低风险回退和规则原因。

### Rule 16 — Give-way action

- 激活的交叉让路阶段要求采取有意义的操纵：航向改变量至少达到配置的 10° 可见性门槛，或选择 25%/50% 的明确减速档位；5° 网格用于搜索实际改向角。
- 交叉让路不强制右转。Rule 15 要求右舷来船让路，并在环境许可时避免横越其船首；Rule 16 要求及早、大幅度、宽裕让清，但未指定转向方向或固定角度。
- 10° 门槛是软件参数，不是 Rule 15/16 的原文要求。预测相对轨迹的船首/船尾通过检查负责判断所选方案是否满足“环境许可时避免横越船首”。

### Rule 17 — Stand-on

- 普通阶段保持航向和航速。
- 让路船显然未采取适当行动，或单靠让路船行动无法避免碰撞时，放开直航船的必要避碰动作。
- Rule 17(c) 仅在相应交叉局面、对本船左舷船舶行动时施加“条件允许时避免左转”的偏好，不应泛化到本船作为追越让路船等其他局面。

## 3. GB/T 46851-2025 参考项

- Appendix A Table A.1：开阔水域对遇 20°、追越 10°、右舷小角度交叉 30°、右舷大角度交叉 60°；这些仅作为首轮规划的软参考，不代表 Rule 14/15 的固定强制角度；水域切换后应使用对应行。
- Appendix C：区分左/右舷对遇、右/左舷交叉、右/左正横以及左右舷追越/被追越。当前 beam sectors 使用 ±15° 的仿真分箱，并将 SDA 映射到相应交叉样例表，标记为估计值。
- Appendix D：应将无碰撞危险、碰撞危险、紧迫局面和目标船行动不协调场景用于回归测试。当前交付增加核心条款测试，完整的全部附录 D 场景自动化验证仍需继续扩充。

## 4. 明确未覆盖/待完善

当前版本重点覆盖开阔水域、互见、一般机动船的 Rule 13–17。仍需补充 Rule 9 狭水道、Rule 10 分道通航制、Rule 18 特殊船舶及其实际状态输入、Rule 19 能见度不良、Rules 34/35 声号以及 Rule 2 特殊情况和良好船艺判断。当前相对方位分类和 DCRC/DCQS 为工程近似，不能替代完整 COLREGs 法律判断、标准认证或实船安全验证。


## 5. Minimum-risk fallback and known mixed-traffic limitation

When no candidate satisfies all predicted SDA and obstacle constraints, the planner does not call the selected path safe. It first chooses among candidates that preserve the core Rule 14 starboard direction and Rule 16 substantial-action requirements, then ranks the remaining candidates by predicted geometric violation and conditional Rule 15 / Rule 17(c) preferences. If no candidate can preserve those core directions, it uses an explicitly logged last-resort minimum-risk fallback that requires review.

The bundled synthetic default 1V2/1V3 mixed-threat smoke setup can begin in a state where no candidate satisfies all SDA constraints for every target. The fallback avoids silently selecting a port turn against a head-on Rule 14 duty where a starboard candidate exists, but may still have a predicted SDA shortfall. It is reported as `minimum_risk`, not as safe passing. This scenario requires additional multi-objective deconfliction / route-priority logic and is a known limitation, not a COLREGs-compliant guarantee.
