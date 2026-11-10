## v1.6.2 PPO 战术权限门控与回航保护

- 修复 PPO 在 `monitor` 阶段也可对所有可行航向动作进行 argmax 的问题：监视阶段动作掩码只允许规则/MPC 基线候选，PPO 不得提前发起偏航或加减速。
- 修复 PPO 在 `route_recovery_active` 阶段可覆盖回航引导的问题：回航阶段只执行 route-recovery MPC 基线动作，保持原有 0.1L/8°/速度/10 s 回航完成判据。
- 仅在规则/风险判断触发的避碰机动窗口（含 Rule 17 直航船需要干预的高风险窗口）内开放 PPO 战术候选。COLREGs 动作掩码和 MPC 可行性检查继续生效。
- 将同一策略权限掩码用于训练采样与部署推理；PPO 训练不能在监视或回航阶段采样部署时必然被拦截的动作。
- 避碰启动、避碰结束进入回航、回航期间新目标加入时，立即清除旧的 10 s 策略保持指令并重新规划，避免旧 PPO 航向在权限切换后继续执行最多 8 s。
- GUI/日志显示决策来源、PPO 权限、门控原因和最近一次机动触发时间。
- 不修改 candidate-v4 动作/观测契约或 Actor-Critic 网络结构；v1.6.1 candidate-v4 权重仍可加载，无须因本次逻辑保护强制重训。
- 验证：128 项非长航程测试通过；6 项回航/控制器测试分组通过（1T、2T、3T 上传 Scene 001 和 3 项风浪/控制器回归）。合计 134 项通过。

## v1.6.1 标准 SDA 风险圈与失败判定


- Map risk circles now use the Appendix-B SDA radius itself; they no longer use the larger SDApref.
- Completed the Appendix-B SDA example lookup for head-on, crossing, beam-on, overtaking, being-overtaken and rear-quarter rows, including bow/stern variants and Table B.4 vessel-type factors.
- Use linear interpolation/extrapolation for ship lengths between/outside tabulated points and mark non-table values as estimates. Crossing/beam risk boundaries use the conservative larger of Appendix-B bow/stern rows until a passage-side-specific threshold is selected; unclassified/following targets use a visibly marked conservative estimated envelope.
- Detect per-control-interval actual swept centre-to-centre separation and latch an SDA failure whenever it falls below the active SDA. The run continues for trajectory analysis, but GUI and evaluation/CSV report failure.
- Add per-target minimum actual separation, SDA margin, and violating target IDs to simulation metrics and evaluation CSV.
- Include SDA circle extents in automatic chart limits and label each ring with its SDA value.

## v1.6.0 速度边界与 COLREGs 核心规则约束

- Tactical speed commands are anchored to the mission's initial cruise speed V0 and bounded to [0.5 V0, 1.5 V0], narrowed by the ownship configured minimum/maximum speed. Added 25% and 50% slow/accelerate action levels; actual surge response remains rate-limited by length-dependent acceleration/deceleration and speed lag.
- Expanded candidate set from 75 to 125 (25 course offsets at 5-degree resolution × five speed levels). Replanning no longer ratchets speed targets.
- Rule 14 head-on active-risk candidates must make a visible starboard alteration; Table A.1's open-water 20-degree initial angle is a soft preference, not a fixed treaty angle.
- Rule 15 crossing-ahead predicted trajectory check now applies to both starboard-crossing and starboard-beam classifications; the planner prefers astern passing when feasible. Rule 15/16 do not prescribe a mandatory starboard turn: a substantial course alteration (either side, subject to the predicted passing geometry) or explicit speed reduction can satisfy the simulator action gate.
- Rule 16 visible-action course threshold is an engineering setting, not a treaty angle; the Rule 14 starboard-direction hard constraint is kept separate from crossing give-way logic.
- Rule 17 stand-on hold-course/hold-speed and conditional 17(c) port-turn restriction are scoped to port-side crossing/beam contacts; urgent fallback remains available when the close-quarters safety condition requires action.
- Encounter classification now exposes right/left beam sectors in addition to head-on, overtaking and crossing; SDA lookup maps beam cases to the crossing example tables while marking this as an estimated mapping.
- Versioned action/observation contracts to candidate-v4 / candidate-observation-v4; PPO checkpoints from candidate-v3 and earlier require retraining. Rule-only mode needs no checkpoint.
- Added tests for the ±50% speed envelope, vessel-size-dependent surge acceleration/deceleration, Rule 14 starboard constraints, Rule 17 stand-on behavior, Appendix C beam classifications and Appendix A indicative angles.

## v1.5.0

- Delay ordinary MPC intervention until a DCRC-style critical window estimated from relative-motion geometry and rudder/yaw response lag; distant 1T-003 at 8 NM now remains in monitor/hold-speed mode.
- Add encounter-specific tactical preferences: overtaking jointly evaluates pass-side course alteration and speed, with the supplied 1T-003 geometry preferring port-side turn/acceleration once the window opens; crossing/head-on give-way phases mask unjustified acceleration and prioritize course-based avoidance.
- Expand the action grid from 7 to 25 heading offsets at 5-degree intervals (0 and ±5 through ±60), combined with three speed modes for 75 candidates.
- Preserve Rule 14 starboard preference, Rule 15 crossing-ahead trajectory check, Rule 17 stand-on course/speed behavior and conditional Rule 17(c) port-turn preference.
- Version the action/observation contracts to candidate-v3 / candidate-observation-v3; prior PPO checkpoints require retraining. Rule-only deterministic operation remains available.
- Add focused tests for the 1T-003 distant/critical-window behavior, 1T-006 crossing behavior, 5-degree action grid and multi-ship route recovery.
- Validation: 93 non-route tests passed; the 1T/2T/3T uploaded Scene 001 route-regression cases passed in targeted runs, and the remaining route/dynamics regression tests passed separately. All 99 collected tests passed across those test groups. Compile checks and package extraction checks are performed for this delivery. Combined route-file execution may take longer than the execution environment's test timeout.

## v1.4.3



- Preserve the mission cruise setpoint captured from ownship's initial scenario speed; avoidance speed changes no longer redefine the cruise target.
- During route recovery and after confirmed route return, all candidate headings use the fixed initial cruise-speed command; tactical slow/accelerate candidates are masked until a new avoidance manoeuvre starts.
- Route-return completion now also requires actual surge speed to be within `max(0.10 m/s, 2% of cruise target)` for the same five consecutive 2 s control steps as cross-track and heading alignment.
- Show mission target speed, current speed error and cruise-lock status in the GUI and simulation logs.
- Add tests for fixed-speed candidates, speed-inclusive recovery stability and post-recovery non-acceleration.
- Validation: `PYTHONPATH=. python -m pytest -q` — 93 tests passed; `compileall` passed. GUI source compilation passed; Qt interactive testing depends on the user's local PySide6 environment.

## v1.4.2

- Fix route-recovery local optimum: recovery scoring prioritizes predicted terminal and integrated cross-track convergence; it no longer awards along-track progress that could outweigh returning to the route.
- Couple recovery look-ahead distance to the actual MPC prediction horizon, bounded by vessel length, so the discrete action offsets do not repeatedly cancel a coarse intercept command.
- Reduce the illustrative wind/wave yaw-acceleration coefficients to keep the maximum default environmental disturbance within the yaw authority of 50–500 m ship configurations.
- Add model-based environmental disturbance feed-forward to the course controller, so the rudder can reject persistent wind/wave yaw bias during candidate prediction and actual execution.
- Add uploaded Scene 001 end-to-end regression coverage for 1T/2T/3T at 200 m and 1T at 500 m, plus planner and disturbance compensation tests.

## v1.4.1

- Rule-only recovery is now an explicit navigation phase with predictive cross-track guidance, rate-aware anticipation and a stable return gate (0.1L, heading error <= 8 deg for 10 s), with cross-track velocity feed-forward to prevent large-vessel overshoot.
- Reset the route-return event when a new avoidance manoeuvre starts; do not advance/end a route while avoidance or recovery is still active.
- Use analytic segment-to-circle clearance and AABB broad-phase rejection before exact polygon distance checks in MPC.
- Use exact swept segment obstacle checks in the safety monitor and contact metrics.
- Bound the local candidate-prediction horizon to keep every GUI planning update responsive; retain global multi-target CPA/TCPA assessment.
- Add regression tests for circle geometry, polygon AABB rejection, route alignment/stability and predictive recovery.

# 版本变更记录

## v1.4.0

- 21候选轨迹替代五个模糊意图；PPO与确定性基线共享候选，全目标规则及距离检查后选择，独立短周期监控记录干预。
- 修复近场TCPA漏判、横荡CPA、目标未来脚本泄漏、预测/实际积分不一致、尾随分类、责任记忆调用频率、初始超限速度瞬跳、重置目标初态及重复事件奖励。
- 2秒反馈/10秒决策统一训练推理；直航按实际会遇参考航向/速度检查，预测全程复用实际反馈时钟。
- 253维版本化观测与21动作网络，旧权重明确拒绝；训练集206/验证36/测试26(seed7)，不再以测试案例参与预热或训练。
- 增加奖励分量、权重/优化器/RNG恢复、时间折扣、独立掩码/监控消融及科研CSV指标。文档见REFACTOR_GUIDE.md。
- 新训练仅完成流程烟雾验证，未完成长期训练或证明PPO优于基线；标准与模型边界见指南。

## v1.3.5

### 地图显示
- 将图例放到左侧海图画布内、绘图区右上方外侧的独立窄栏，紧贴海图且不遮挡坐标区。
- 本船船型符号改为电光青色填充、白色描边和暗色外缘；显示尺寸放大到 1.48 倍，纯属可视化变化，不改变动力学船长和碰撞几何。

### 右侧操作布局
- 将右侧主要操作拆分为“场景/仿真”“PPO/GPU”“批量评估”三个标签页。
- 将实时状态、目标船信息、碰撞风险和运行日志分组；后面三项改为分页查看，减少右侧滚动长度。
- 调整左右分栏宽度，收窄右侧控制区域，优先保留左侧海图空间。
- 保留圆形/多边形障碍物编辑、缩放/拖动、PPO 加载/卸载与 CSV 批量评估功能。

### 验证
- `python -m py_compile ui/main_window.py` 通过。
- `python -m pytest -q`：26 项测试全部通过。
- 构建环境未安装 PySide6，未进行真实 Qt 窗口启动及鼠标交互验收。


## v1.2.0（本次交付）

### 路径规划
- DWA 候选最终航向不再被每周期航向指令变化率错误压缩；MPC 预测使用指令爬升过程，实际每次只下发受限的第一步控制。
- 将安全约束比较改为分层选择：安全可行优先；若无安全候选，先最小化约束违约量，再比较规则和其他代价。
- 加强目标驶过后的清除判据，避免 CPA 刚过去就解除避让和过早恢复航速。
- 保留本船对所有目标的多船约束，同时区分最高风险目标和让路责任目标。

### PPO pipeline
- 增加规则动作掩码；训练采样、环境执行和 PPO 更新时采用同一个动作掩码。
- 活跃避让期间“保持/维持当前策略”作为保持动作；紧迫状态允许策略升级。
- 修复训练 rollout 中采样动作与安全层最终动作可能不一致的问题。

### GUI
- 主窗口默认最大化；场景和训练参数放在顶部标签页，为地图留出更大空间。
- 地图标题、坐标轴与图例使用英文；其余 UI 使用中文。
- 地图自动缩放根据当前船舶和最近航迹，不再让完整远期 MPC 预测路径将绘图区缩小。
- 横纵轴物理比例保持 1:1。
- 所有 `last_plan` 读取均先检查空值，修复加载 PPO 后的 `NoneType` 异常。

### 数据与 GPU
- 保留用户上传的 268 个 1T/2T/3T CSV 场景。
- 回合时长按源 TCPA 扩展。
- 增加 CUDA PyTorch 安装脚本、GPU 检查脚本和明确的 CUDA 可用性错误处理。
- `run_windows.bat` 使用当前激活的 Python 解释器，避免训练安装在另一个本地 `.venv`。
- 启动器改为先检查 CUDA，已可用时跳过 PyTorch 重装；缺失时询问是否安装。CUDA 安装失败不会自动卸载现有 Torch 并替换成 CPU 版本。

## 仍然存在的研究限制

原始 CSV 不是逐时刻轨迹；船长/船型/环境及目标船未来操纵行为需要用户补充或设定。当前运动模型、COLREGs 分类、SDA 数据外推和 MPC 仍需针对具体船型标定与更完整标准测试。

## v1.3.0

- Fixed premature encounter release: a target must have passed CPA (TCPA < -60 s), be receding, and be outside an expanded clearance range before the ownship can return to its route.
- Preserved the initial COLREGs encounter classification and give-way responsibility through the active encounter; DCPA temporarily improving no longer clears encounter memory.
- Changed the map to a landscape plotting canvas and removed the square axes constraint. The axes now use their actual physical aspect ratio to calculate data limits, preserving 1:1 metre scale while filling the left panel horizontally. The map focuses on the current encounter and recent track tail for a closer view.
- Added regression coverage for a head-on scenario that previously released avoidance at about 430 s while the target was still ahead.


## v1.3.1

- Preserve and draw complete ownship and target vessel tracks without fixed point truncation.
- Add a clear-tracks action that resets track history at the current vessel positions without resetting the simulation.
- Add background PPO batch evaluation for 1T, 2T, 3T, or all 268 cases.
- Write one CSV row per case immediately after evaluation; include outcome, collisions, minimum separation, unsafe planner count, route return, terminal arrival, and runtime metrics.
- Keep the selected PPO checkpoint active when resetting or reloading a scenario.


## v1.3.2

- Add an explicit PPO unload button to the PPO training/GPU settings tab. Unloading switches the active simulation policy to the rule-based fallback, clears the active checkpoint selection so later scenario resets do not reload it, and leaves the checkpoint file untouched.
- Disable batch evaluation when no PPO checkpoint is loaded; re-enable it after a valid checkpoint is loaded. Unloading is blocked while a batch evaluation subprocess is running.
- Move the map legend and vessel/track/obstacle symbol samples into a dedicated area outside the map axes to prevent obscuring map content. True vessel positions and tracks remain plotted on the map.


## v1.3.3

- Change the ownship symbol to high-visibility yellow with a dark outline.
- Add pointer-centred wheel zoom, left-button drag panning, view reset, and preserve the user's map extent while the simulation redraws.
- Add GUI polygon drawing with vertex undo/cancel/finish, simple-polygon validation, click-to-select, delete-selected, and clear-manual-obstacles actions.
- Insert manually drawn obstacles into the active scenario obstacle list so DWA-MPC evaluates their polygon clearance during predicted trajectories.
- Add static-obstacle contact counting to simulation metrics and PPO batch-evaluation CSV output.


## v1.3.4

- Reorganize the main window into a large left-side chart and a right-side scrollable panel containing scene controls, PPO/GPU settings, batch evaluation, map tools, legend, and live status information.
- Expand the chart axes to use nearly the full left canvas; remove the in-figure legend so it no longer consumes map space.
- Add a user-defined circular obstacle tool: enter a radius in metres, preview the circle while moving the cursor, and click the chart to set its centre. Right-click or the cancel button exits placement mode.
- Store circle centre/radius metadata and generate a 64-segment circumscribed polygon boundary, allowing the existing MPC polygon-clearance and engine contact-counting paths to handle circles conservatively.
- Preserve manually placed circle geometry when resetting the same scene; circles remain selectable and deletable like hand-drawn polygons.
- Add circle-geometry and obstacle-copy regression tests.


## v1.3.6

- Route recovery now requires cross-track error <= 0.1 * ownship length (no 150 m floor).
- Live metrics show route cross-track error and the active recovery threshold.
- Removed the "本船" text drawn over the ownship marker; legend label remains.
- Added regression tests for 50 m and 200 m ownship lengths and route-direction-independent cross-track distance.
