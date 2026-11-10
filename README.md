# USV 智能船舶避碰仿真系统（USV COLREGs Simulator）

> **版本**：v1.6.2　|　**动作契约**：`candidate-v4`　|　**观测契约**：`candidate-observation-v4`
>
> 一个面向无人水面艇（USV）的研究型「软件在环」避碰仿真系统。它以 **1972 年《国际海上避碰规则》（COLREGs）** 与 **GB/T 46851—2025** 为参考，将
> **规则约束筛选 + DWA-MPC 滚动时域规划 + PPO 深度强化学习** 三层机制组合起来，在导入的 1T/2T/3T 会遇场景上完成会遇态势识别、碰撞风险评估、避碰决策、航线回归与批量评估。

---

## 目录

1. [项目简介](#1-项目简介)
2. [目录结构](#2-目录结构)
3. [环境安装与快速开始](#3-环境安装与快速开始)
4. [系统总体架构与运行流程](#4-系统总体架构与运行流程)
5. [坐标系与角度约定](#5-坐标系与角度约定)
6. [船舶运动学模型（3-DOF）](#6-船舶运动学模型3-dof)
7. [会遇态势识别（COLREGs Rule 13–17）](#7-会遇态势识别colregs-rule-1317)
8. [碰撞风险评估](#8-碰撞风险评估)
9. [安全会遇距离 SDA 的计算](#9-安全会遇距离-sda-的计算)
10. [动作空间：125 个候选动作](#10-动作空间125-个候选动作)
11. [DWA-MPC 规划器](#11-dwa-mpc-规划器)
12. [PPO 强化学习策略](#12-ppo-强化学习策略)
13. [策略权限门控（v1.6.2）](#13-策略权限门控v162)
14. [航线回归控制](#14-航线回归控制)
15. [奖励函数](#15-奖励函数)
16. [安全监控与 ODD](#16-安全监控与-odd)
17. [数据集与数据划分](#17-数据集与数据划分)
18. [训练与评估](#18-训练与评估)
19. [测试](#19-测试)
20. [局限性与安全声明](#20-局限性与安全声明)

---

## 1. 项目简介

本系统解决的核心问题是：**在开阔水域的多船会遇场景中，本船（ownship）如何遵守 COLREGs 的让路/直航责任，同时避免碰撞、避让静态障碍物，并在避让结束后安全回归原航线。**

系统把避碰决策拆解为一条**共享的执行链**（训练与部署复用同一套代码，见 [engine.py](usv_sim/engine.py)）：

```text
┌─────────────┐   ┌──────────────┐   ┌─────────────────┐   ┌──────────────┐   ┌────────────┐
│ 会遇态势识别 │ → │ 风险评估      │ → │ DWA-MPC 候选生成 │ → │ 规则/安全掩码 │ → │ 策略选动作   │
│ (colregs.py)│   │ (risk.py)    │   │ (planner.py)    │   │ (rules/safety) │   │ PPO / 规则  │
└─────────────┘   └──────────────┘   └─────────────────┘   └─────────────────┘   └────────────┘
                                                                                        │
                                            3-DOF 船舶动力学 (dynamics.py)  ←────────────┘
```

三层机制的分工如下：

| 层次 | 模块 | 职责 |
|---|---|---|
| **规则层** | `colregs.py` / `rules.py` | 判定会遇类型、让路/直航责任，生成**硬性动作掩码**（Rule 13–17） |
| **规划层** | `planner.py` | 生成 125 个候选动作，滚动预测轨迹，用确定性代价函数选出基线动作（DWA-MPC） |
| **学习层** | `ppo_agent.py` / `train_ppo.py` | 在规则掩码允许的候选子集内，用 PPO 策略选择战术动作 |

---

## 2. 目录结构

```text
usv-ccs/
├── configs/                     # 自定义 JSON 场景配置
├── datasets/cases/              # 用户上传的 268 个会遇案例 CSV（1T/2T/3T）
│   ├── 1t/                      # 1 目标船 × 138 场景
│   ├── 2t_0720/                 # 2 目标船 × 99 场景（含预览图）
│   └── 3t/                      # 3 目标船 × 31 场景（含预览图）
├── docs/                        # 设计说明、标准追溯、验证结果
├── models/                      # PPO 权重输出位置
├── logs/                        # 无界面仿真的 JSON 日志输出
├── tests/                       # pytest 测试
├── ui/main_window.py            # PySide6 中文 GUI
├── usv_sim/
│   ├── models.py                # 数据结构（船舶/目标/障碍物/场景/指标）
│   ├── geometry.py              # CPA、DCPA、线段-点/圆/多边形距离
│   ├── dynamics.py              # 简化 3-DOF 操纵模型
│   ├── colregs.py               # 会遇分类与规则逻辑（Rule 13–17）
│   ├── rules.py                 # 可执行的规则硬约束（候选掩码）
│   ├── risk.py                  # 多船 CPA/TCPA/SDA/DCRC/DCQS 风险评估
│   ├── sda.py                   # 安全会遇距离 SDA（GB/T 46851 附录 B）
│   ├── actions.py               # 125 个候选动作与速度包络
│   ├── planner.py               # DWA 候选 + MPC 滚动预测 + 代价函数
│   ├── ppo_agent.py             # PPO 观测构造、Actor-Critic、策略加载
│   ├── rewards.py               # 奖励函数
│   ├── safety.py                # 短时域执行监控
│   ├── odd.py                   # 运行设计域（ODD）监控
│   ├── obstacles.py             # 多边形/圆障碍物几何
│   ├── dataset.py               # 1T/2T/3T CSV 读取器
│   ├── scenario_io.py           # JSON 场景读写
│   ├── scenarios.py             # 内置合成场景
│   ├── data_split.py            # 几何指纹划分训练/验证/测试集
│   └── engine.py                # 仿真主循环
├── train_ppo.py                 # PPO 训练入口
├── evaluate_ppo.py              # 批量评估与 CSV 导出
├── run.py                       # GUI 入口
├── run_headless.py              # 无界面仿真入口
├── gpu_check.py                 # CUDA 可用性检查
├── run_windows.bat              # Windows 一键启动
├── install_cuda_torch.bat       # CUDA PyTorch 安装脚本
├── requirements.txt             # 通用依赖
└── requirements-validated-cpu.txt  # 已验证的 Windows/Python3.11 CPU 依赖
```

---

## 3. 环境安装与快速开始

### 3.1 依赖

核心依赖（[requirements.txt](requirements.txt)）：

```
numpy>=1.24
matplotlib>=3.7
PySide6>=6.6,<6.9
pytest>=7.4
torch>=2.3
```

已在本构建环境验证的 Windows / Python 3.11 **CPU** 环境（[requirements-validated-cpu.txt](requirements-validated-cpu.txt)）：

```
--extra-index-url https://download.pytorch.org/whl/cpu
numpy==1.26.4
matplotlib==3.8.4
PySide6==6.8.3
torch==2.15.0+cpu
pytest>=7.4
setuptools==78.1.0
```

### 3.2 启动

**GUI 方式（Windows）：**

```bat
conda activate yolo
python -m pip install -r requirements.txt
call install_cuda_torch.bat      # 可选：安装 CUDA 版 PyTorch
python gpu_check.py              # 检查 CUDA
python run.py
```

**无界面核心仿真：**

```bat
python run_headless.py --targets 3 --steps 120
```

**训练 PPO：**

```bat
python train_ppo.py --steps 150000 --device cuda --dataset-root datasets/cases --out models/ppo_policy.pt
```

> 注意：`train_ppo.py` 默认要求 CUDA，未检测到 GPU 时会**明确报错**而非静默回退 CPU；如需 CPU，请显式传入 `--device cpu`。

---

## 4. 系统总体架构与运行流程

仿真引擎 [engine.py](usv_sim/engine.py) 采用**双层时间节拍**：

- **控制节拍**：`CONTROL_DT_S = 2.0 s`（船舶动力学积分步长）
- **策略节拍**：`DECISION_PERIOD_S = 10.0 s`（PPO/规则决策周期，即 5 个控制步）

每个控制周期（[step](usv_sim/engine.py)）的流程如下：

```text
1. ODD 检查       —— 超出运行设计域则强制接管并停止
2. 终点检查       —— 到达航线终点则结束
3. 风险评估       —— 重新计算多船 CPA/TCPA/SDA 与风险等级
4. 会遇状态更新   —— 判断是否进入/退出避碰机动、航线回归
5. 决策（若到期） —— 生成候选 → 规则掩码 → 选择动作 → 更新保持的舵令/航速
6. 安全监控       —— check_control 短时域校验，必要时干预
7. 动力学推进     —— 本船与目标船各积分一步（2 s）
8. 接触/侵入记录  —— 碰撞、SDA 侵入、障碍物接触统计
9. 回归/终点更新  —— 航线回归稳定计数、终点判断
10. 写日志        —— 记录策略来源、门控原因、风险、SDA 裕度等
```

关键状态机（[engine.py](usv_sim/engine.py) 中的 `navigation_stage` / `phase`）：

```text
monitor（监视）
  │  风险触发（emergency / stand-on 干预 / give-way 激活）
  ▼
maneuver_active（避碰机动） ← PPO 在此阶段拥有动作选择权
  │  所有机动目标驶过让清（TCPA < -60s 且距离 > 1.5·1.274·SDA 且风险 < 0.08）
  ▼
route_recovery_active（航线回归） ← 确定性 MPC 控制，PPO 被门控屏蔽
  │  横向偏差 ≤ 0.1L 且 航向误差 ≤ 8° 且 航速误差达标，连续稳定 5 个控制步（10 s）
  ▼
post-recovery cruise（已回归巡航） ← 保持初始巡航航速，等待下一次会遇
```

---

## 5. 坐标系与角度约定

系统采用航海惯例（[models.py](usv_sim/models.py) 中 `ShipState`）：

| 量 | 约定 |
|---|---|
| 位置 `(x, y)` | 本船为原点的局部平面坐标：`x` 为东向距离（m），`y` 为北向距离（m） |
| 船首向 `ψ` | 弧度，`0 = 正北`，**顺时针为正**（右转为正） |
| 相对方位 | `relative_bearing = wrap(atan2(dx,dy) - ψ)`，**正值为右舷** |
| 航向改变 | **正值为右转（starboard）** |
| 速度 | 模型内部为 m/s；界面显示对地航速 SOG（kn）、对地航向 COG、船首向 Heading |

角度归一化使用 [wrap_angle](usv_sim/models.py)：`wrap_angle(a) = (a + π) mod 2π - π`，将任意角映射到 `[-π, π)`。

---

## 6. 船舶运动学模型（3-DOF）

[dynamics.py](usv_sim/dynamics.py) 实现了一个**简化 3 自由度（纵荡 surge、横荡 sway、艏摇 yaw）操纵模型**（Nomoto-like 一阶响应），带舵机、艏摇、航速滞后。它是研究用可调模型，**不是由实船试航数据标定的 MMG 模型**，接口保持与后续 MMG 实现兼容。

### 6.1 状态量

```text
x, y    位置（m）
ψ       船首向（rad）
u       纵荡速度 surge（m/s）
v       横荡速度 sway（m/s）
r       艏摇角速度 yaw rate（rad/s）
δ       实际舵角（rad）
n       归一化推进状态
```

### 6.2 舵机模型（一阶 + 速率限幅）

```text
δ̇_desired = (δ_cmd − δ) / τ_rudder
δ̇ = clamp(δ̇_desired, ±max_rudder_rate)
δ  = clamp(δ + δ̇·dt, ±max_rudder_deg)
```

### 6.3 艏摇模型（Nomoto 一阶 + 速率限幅 + 环境扰动）

```text
r_target = K_yaw · δ · max(0.25, u/7)
ṙ = (r_target − r) / τ_yaw + disturbance
r  = clamp(r + ṙ·dt, ±max_yaw_rate)
```

其中 `K_yaw = yaw_gain`（与船长相关）。环境扰动为**小幅度、受船舶艏摇能力约束**的模型扰动：

```text
wind = 2.5e-6 · wind_speed · sin(wind_dir − ψ)
wave = 1.0e-5 · wave_level · sin(0.17·t + 0.001·x)
disturbance = wind + wave
```

### 6.4 纵荡模型（一阶 + 加速度/减速度限幅）

```text
u̇_desired = (u_cmd − u) / τ_speed
u̇ = clamp(u̇_desired, −max_decel, +max_accel)
u  += u̇·dt   （并保证不越界超调指令）
```

### 6.5 横荡近似

```text
v_target = −0.12 · u · r
v += (v_target − v) · min(1, dt / max(3, 0.06·L))
```

### 6.6 位置积分（含风流）

```text
ẋ = u·sin(ψ) + v·cos(ψ) + current_x
ẏ = u·cos(ψ) − v·sin(ψ) + current_y
```

### 6.7 船长相关配置

[ShipConfig.for_length](usv_sim/models.py) 以 200 m 为基准，按比例 `scale = L/200` 生成一套**保守的示例配置**（非特定船型标定）：

- `beam = max(8, 0.16·L)`
- `rudder_tau = 5 + 2·scale`（s）
- `yaw_tau = 32·scale`（s）
- `yaw_gain = 0.075/scale`
- `max_yaw_rate = clamp(0.6/scale, 0.18, 1.8)`（°/s）
- `max_accel = 0.08/√scale`，`max_decel = 0.12/√scale`（m/s²）

> 大船（L 大）艏摇时间常数更大、艏摇能力更弱、加减速更慢——这直接影响了避碰时机与回航时的“过冲/摆动”行为，也是 v1.4.x 航线回归修正的物理背景。

目标船的推进（[TargetShip](usv_sim/models.py)）默认采用**恒速恒向**模型（`constant_velocity`），并支持脚本化的 `turn_after_delay`、变速等行为；默认预测**不读取未来脚本**（`oracle=False`）。

---

## 7. 会遇态势识别（COLREGs Rule 13–17）

[colregs.py](usv_sim/colregs.py) 的 [classify_encounter](usv_sim/colregs.py) 基于**方位/航向快照**给出会遇分类、让路责任与对应规则。它是有意保守的近似，不是完整的法律瞭望判定。

### 7.1 判定流程

```text
输入：本船 own、目标 target
1. 相对方位 rel_bearing（正=右舷）
2. 航向差 heading_diff = |wrap(ψ_target − ψ_own)|
3. CPA 计算 → DCPA / TCPA / range
4. 追越判定：
   - own_is_overtaking：本船从目标船正横后 > 22.5° 方向接近（|own_bearing_rel_target| > 112.5°）
   - target_is_overtaking：112.5° < |rel_bearing| ≤ 180° 且航向差 < 70°
```

### 7.2 分类规则

| 会遇类型 | 判据 | 本船让路？ | 规则 |
|---|---|---|---|
| `head_on`（对遇） | \|rel_bearing\| ≤ 10° 且 \|180°−航向差\| ≤ 15° | 是 | Rule 14 |
| `own_overtaking`（本船追越） | 本船从目标正横后 >112.5° 接近 | 是 | Rule 13 |
| `being_overtaken`（被追越） | 目标从本船正横后 >112.5° 接近 | 否 | Rule 13/17 |
| `following`（同向尾随） | 航向差 < 15° 且 \|rel\| < 90° 且本船未追近 | 否 | monitor |
| `starboard_beam`（右正横） | 90°±15° 右舷扇区 | 是 | Rule 15/16 |
| `starboard_crossing`（右舷交叉） | 右舷其余扇区 | 是 | Rule 15/16 |
| `port_beam`（左正横） | 90°±15° 左舷扇区 | 否 | Rule 15/17 |
| `port_crossing`（左舷交叉） | 左舷其余扇区 | 否 | Rule 15/17 |

其中正横扇区（±15°）是仿真器的分箱约定，用于把附录 C 的右/左正横接触单独暴露，而不是塌缩成通用“交叉”标签。

### 7.3 关键规则语义

- **Rule 13 追越**：追越船持续让清，不因方位后来改变就自动改判为交叉；通过侧由几何、安全域与障碍物共同决定，规则不强制左/右舷。
- **Rule 14 对遇**：双方均向右转、左舷对左舷通过。规则掩码**禁止向左或保持**；20° 是 GB/T 46851 附录 A 表 A.1 的开阔水域初始改向参考，不是公约固定角度。
- **Rule 15 交叉**：右舷目标且有碰撞危险时本船让路；条文**不强制交叉让路船右转**。规划器对预测轨迹做“船首通过”检查，优先选择从船尾通过的轨迹。
- **Rule 16 让路船行动**：要求尽早、大幅度、宽裕让清。工程上把**航向变化 ≥ 10° 或明确减速（25%/50%）** 作为“明显操纵”选项；10° 不是公约明文阈值。
- **Rule 17 直航船行动**：通常先保持航向航速；当让路船显然未适当行动或逼近到仅凭让路船无法避碰时，再采取必要行动。Rule 17(c) 的“避免向左转”仅用于本船左舷目标的相应交叉局面。

[assess_rule](usv_sim/colregs.py) 在会遇分类基础上，结合风险等级给出**首选转向角**（工程参考，非法律强制）：

| 会遇 | 首选转向角 |
|---|---|
| 对遇 | +20°（右转） |
| 右舷交叉/正横 | +30°（小角度交叉）/ +60°（大角度交叉或正横） |
| 本船追越 | 0°（不指定通过侧） |
| 直航类 | 0°（保持）；紧迫局面时按方位符号取 ±1° 提示 |

---

## 8. 碰撞风险评估

[risk.py](usv_sim/risk.py) 的 [MultiShipRiskAssessor](usv_sim/risk.py) 对每艘活动目标船**独立**评估风险，产生 [EncounterAssessment](usv_sim/models.py) 与 [RiskSnapshot](usv_sim/models.py)。

### 8.1 基础 CPA/TCPA

[geometry.py](usv_sim/geometry.py) 的 [cpa](usv_sim/geometry.py)：

```text
r = 目标位置 − 本船位置
v = 目标速度 − 本船速度
TCPA = −(r·v)/(v·v)          （v≈0 时取 0）
DCPA = |r + v·TCPA|
```

速度由 [velocity_from_state](usv_sim/geometry.py) 计算：`(u·sinψ + v·cosψ, u·cosψ − v·sinψ)`。

### 8.2 SDA 与 SDApref

- `SDA`（安全会遇距离）：从附录 B 表查得的**硬性安全距离边界**，也是地图风险圈半径与“实际最小中心距失败判定”的阈值。
- `SDApref`：用于**风险/干预时机**的偏好安全域：`SDApref = (λ_1 + 0.274) · SDA`，其中 `λ_1 = 2.0`（低能见度）或 `1.0`（正常）。

### 8.3 DCRC / DCQS 干预窗口

[ `_critical_distance_threshold`](usv_sim/risk.py) 受附录 A 启发，对某一候选转向角求解**改向前的相对轨迹进入改向后相对运动线两侧 clearance 带的距离**，得到干预距离/时间；再叠加舵机/艏摇响应延迟，把触发点提前：

```text
response_delay = max(20, 1.25·τ_yaw + 1.5·τ_rudder + |turn|/max_yaw_rate)
```

- `DCRC`（Distance to Collision Risk Commencement）：以 `SDApref` 为 clearance、以首选转向角求解。
- `DCQS`（Distance to Close Quarters Situation）：以 `SDA` 为 clearance、以 ±90° 求解。

> 这是工程近似，不是对附录 A 全部 DCRC/DCQS 公式的逐式复现。

### 8.4 风险分级与分数

风险分数 `score ∈ [0,1]` 按下述优先级判定（`risk_level` 为中文标签）：

| 条件 | risk_level | score 公式 |
|---|---|---|
| 重叠或 `rng ≤ 1.10·SDA` | 紧迫局面 | `min(1, 0.82 + 0.18·(1 − rng/SDA))` |
| `TCPA ≤ 0` | 无危险 | `0` |
| `DCPA > SDA`（标称安全） | 关注 / 无危险 | `0.05` 或 `0` |
| `DCPA ≤ SDA` 且 `rng > DCRC` | 关注 | `0.06` |
| `DCPA ≤ SDA` 且 `rng ≤ DCQS` | 紧迫局面 | `min(1, 0.78 + 0.22·(1 − rng/DCQS))` |
| 其余（进入 DCRC 带内） | 碰撞危险 | `0.12 + 0.56·penetration` |
| 退化/近平行几何 | 碰撞危险（保守 TCPA 回退） | 见源码保守式 |

其中 `penetration = clamp((DCRC − rng)/(DCRC − DCQS), 0, 1)` 表示对 DCRC 窗口的穿透程度。

### 8.5 会遇记忆

[ `_encounter_memory`](usv_sim/risk.py) 在风险首次 ≥ 0.05 且会遇非 `none/following` 时**锁定初始会遇类型与让路责任**，避免本船避让改变相对方位后被错误重判。当目标安全远离（`TCPA < −60 s`、`rng ≥ max(2·SDA, 2·max(L_own,L_target))` 且距离持续增大）并稳定 ≥ 6 s 后，才清除记忆。

---

## 9. 安全会遇距离 SDA 的计算

[sda.py](usv_sim/sda.py) 实现 GB/T 46851—2025 **附录 B** 的示例 SDA 数据库。

### 9.1 数据表结构

附录 B 给出散货船在**本船长 {100, 200, 300} m** × **目标船长 {50, 100, 200, 300} m** 网格上的 SDA（单位海里，NM）。会遇分 head_on / crossing / beam / overtaking / being_overtaken / rear 系列，交叉与正横行区分 **bow（船首）/ stern（船尾）** 通过侧。

### 9.2 插值/外推算法

对任意本船长 `L_own` 与目标船长 `L_target`：

1. 对目标船长在 `[50,100,200,300]` 上做**分段线性插值**，端点之外**线性外推**，得到每个本船长档位的 SDA；
2. 再对本船长在 `[100,200,300]` 上做同样的插值/外推；
3. 乘以表 B.4 的**船型比例系数**：
   ```text
   lng/chemical 1.50, container 1.33, tanker 1.36, passenger 1.20, bulk/other 1.00
   ```
4. 单位换算：`SDA_m = base_nm × factor × 1852`。

### 9.3 通过侧处理

- `generic`：首尾值相等的行（对遇/追越）。
- `bow` / `stern`：明确通过侧。
- `conservative_max`：交叉/正横在**尚未确定首尾通过侧**时，取 bow/stern 中较大者作为保守边界。
- `none/following` 等附录 B 无专门行的分类：取所有示例行的最大插值，并**显式标记为估算**。

返回 `(SDA_m, estimated)`，`estimated` 表示结果经过插值/外推或船型未知，UI 与日志会标记“SDA 按附录B表值插值/外推估算”。

---

## 10. 动作空间：125 个候选动作

[actions.py](usv_sim/actions.py) 定义版本化动作契约。

### 10.1 航向偏移

```text
OFFSET_ANGLES_DEG = {0, ±5, ±10, ..., ±60}   （共 25 个，5° 一档）
```

### 10.2 速度档位

```text
SPEED_MODES = {hold, slow_25, slow_50, accelerate_25, accelerate_50}
SPEED_MODE_RATIO = {hold:0, slow_25:−0.25, slow_50:−0.50, accelerate_25:+0.25, accelerate_50:+0.50}
```

### 10.3 组合

```text
25 个航向偏移 × 5 个速度档位 = 125 个候选动作（ACTION_COUNT = 125）
```

### 10.4 速度包络（锚定初始航速 V0）

战术目标航速的**硬边界**为 `[0.5·V0, 1.5·V0]`，并进一步受本船配置的最小/最大航速约束（[speed_envelope](usv_sim/actions.py)）：

```text
lower = max(min_speed, 0.5·V0)
upper = min(max_speed, 1.5·V0)
```

[speed_setpoint](usv_sim/actions.py) 保证**非棘轮**（non-ratcheting）——速度档位始终锚定 `reference_speed`（本回合会遇参考速度）与 `V0`（任务初始航速），不会因反复重规划而逐次累加速度：

```text
requested = reference_speed + ratio · V0
setpoint  = clamp(requested, lower, upper)
```

---

## 11. DWA-MPC 规划器

[planner.py](usv_sim/planner.py) 的 [DWAMPCPlanner](usv_sim/planner.py) 实现**有限候选 + 滚动时域预测**控制器。它为每个候选动作做**轨迹预测 → 规则/安全掩码 → 代价评分 → 候选选择**。

### 11.1 候选生成（generate_candidates）

对 125 个动作逐一执行：

1. **航向指令**：`course = base + offset_deg`，其中 `base` 为航线引导角（或本回合参考航向）。
2. **速度指令**：由 `speed_setpoint` 计算（回航/巡航阶段锁速到任务巡航目标）。
3. **规则掩码**：[candidate_rule_reasons](usv_sim/rules.py) 返回该候选违反的规则编码；被**全局速度模式锁**排除的候选跳过完整轨迹积分（保留固定动作槽位与审计原因）。
4. **轨迹预测**：以控制节拍 `control_period_s = 2 s` 推进动力学模型到局部预测时域 `horizon`，形成预测路径 `path`。
5. **几何碰撞检测**（对每步线段）：
   - 目标船：`swept_separation`（线段最小距离）减去曲率补偿 `curvature_pad` 与不确定度 `uncertainty`，再减 `SDA` 得 clearance；
   - 静态障碍物：先做 AABB 宽相快速排除（O(1)），接近时才做精确的线段-圆 / 线段-多边形距离。
6. **船首通过检查**（Rule 15）：对右舷让路目标，若预测相对轨迹横穿目标船首（`lat0·lat1 < 0` 且前方分量 > 0），追加 `R15_crossing_ahead` 违规。
7. **代价评分**：见 11.3。

### 11.2 局部预测时域

- 无活动目标时 `horizon = 60 s`。
- 有活动目标时**自适应**收缩：取最早的相关 CPA 时间 `min(tcpa + 120)`，下限 180 s、上限 `max_horizon = 600 s`。
- 风险评估（全局 CPA/TCPA）不受此局部时域限制。

### 11.3 代价函数

每个候选的分数为：

```text
score = route_cost
      + offset_weight · |offset_deg|
      + 0.02 · √(rudder_cost)
      + continuity_weight · continuity
      + 0.08 · |speed − own.u|
      + progress_reward
      + 2 · max(0, −pref_min/1000)²
      + overtake_preference
      + rule_angle_cost
```

各项含义：

| 项 | 说明 |
|---|---|
| `route_cost` | 航线偏差代价，权重**随阶段不同**（见下） |
| `offset_weight·\|offset\|` | 惩罚大角度转向（保舵效平滑） |
| `0.02·√(rudder_cost)` | 舵角平方的均方根惩罚 |
| `continuity_weight·continuity` | 与上次指令的连续性 |
| `0.08·\|Δspeed\|` | 航速变化惩罚 |
| `progress_reward` | 沿航线前进奖励（仅正常巡航阶段） |
| `2·max(0,−pref_min/1000)²` | 偏好安全域（`SDApref`）侵入惩罚，`pref_min` 为预测中的最小偏好 clearance |
| `overtake_preference` | 追越通过侧软偏好（±10° 参考） |
| `rule_angle_cost` | 对附录 A 表 A.1 初始改向角的软代价（对遇 20°、右舷交叉 30°/60°） |

`route_cost` 的**阶段相关**定义（v1.4.x/v1.6.x 关键）：

```text
正常巡航：  route_cost = 0.4·cross_cost + 0.002·terminal_cross，progress_reward = −0.0004·max(0, forward)
避碰机动：  route_cost = 0.12·cross_cost + 0.0005·terminal_cross，progress_reward = 0（不许为航线前进牺牲会遇）
航线回归：  route_cost = 1.50·cross_cost + 0.020·terminal_cross − 0.50·recovery_gain
           recovery_gain = (initial_cross − terminal_cross)/max(initial_cross, L)
           progress_reward = 0（回归优先收敛横向误差，而非沿航线前进）
```

其中 `cross_cost` 为整段横向偏差的积分均值，`terminal_cross` 为预测末端横向偏差，`forward` 为沿航线前进距离。回归阶段的 `recovery_gain` 显式奖励“横向偏差收敛”，避免大船横穿航线后反复摆动、陷入“平行航行”的局部最优。

### 11.4 候选选择（select_candidate）

```text
1. 存在可行候选（feasible）→ 取 (score, id) 字典序最小者
2. 全部不可行 → 先找仍满足核心规则方向（R14_starboard_required / R16_substantial_action）的
   “最小风险”候选，按 (violation, 条件违规数, score, id) 选择
3. 最后兜底 → 取最小物理违约的候选，显式标记 minimum_risk，不伪称安全/合规
```

---

## 12. PPO 强化学习策略

[ppo_agent.py](usv_sim/ppo_agent.py) 定义版本化观测、共享 Actor-Critic 网络与策略加载；[train_ppo.py](train_ppo.py) 实现 PPO 训练。

### 12.1 观测构造（build_observation）

观测由固定顺序拼接，总维度 `OBS_SIZE = 1189`：

```text
OBS_SIZE = 9（本船特征）
         + 3 槽位 × 12（目标特征）
         + 14（上下文）
         + 5（聚合）
         + 125 × 9（候选特征）
         = 1189
```

- **本船特征（9）**：`u/max_speed`、`v/max_speed`、`sin ψ`、`cos ψ`、`r/max_r`、`δ/max_δ`、横向偏差/5000、剩余航程/30000、船长/500。
- **目标槽位（3×12）**：按目标 ID 排序填入最多 3 个目标；每槽含存在标志、归一化距离、`sin/cos` 相对方位、相对速度、`sin/cos` 相对航向、归一化 DCPA、归一化 TCPA、风险分数、归一化 SDA、是否让路。
- **上下文（14）**：机动激活标志、参考航向 `sin/cos`、参考航速、已用时、阶段 one-hot（stand_on_intervene/give_way/emergency/passing/monitor/stand_on）、上一动作、干预标志、ODD 降级标志。
- **聚合（5）**：目标数/10、最大风险、最小 `range/SDA`、让路目标数/10、是否存在已过 CPA 且侵入 SDA 的目标。
- **候选特征（125×9）**：每个候选的得分（用 `sign·log1p(|score|)` 保留排序又不致无界）、偏移角/60、速度、最小 clearance、偏好 clearance、是否违约、是否可行、是否 minimum_risk。

> 候选得分直接来自规划器对每个目标/障碍的扫描，使网络能在“候选级”特征上学习排序。

### 12.2 网络结构

[make_actor_critic](usv_sim/ppo_agent.py) 返回共享骨干的 Actor-Critic：

```text
backbone: Linear(1189 → 128) → Tanh → Linear(128 → 128) → Tanh
actor:    Linear(128 → 125)           # 动作 logits
critic:   Linear(128 → 1)             # 状态价值 V
```

### 12.3 动作掩码

采样与推理都使用**同一动作掩码**：把不允许候选的 logits 置为 `−1e9`，再从 Categorical 分布采样（训练）或 `argmax`（推理）。这保证训练时学不到部署时被禁止的动作。掩码来源：

1. 规则硬约束（Rule 13–17，见 [rules.py](usv_sim/rules.py)）；
2. 几何可行性（SDA / 障碍物）；
3. **策略权限门控**（v1.6.2，见第 13 节）。

### 12.4 PPO 训练（train_ppo.py）

- **可选候选基线预热（warm-start）**：以确定性候选选择器生成教师标签（伪标签，非专家真值），交叉熵训练 `warmup_epochs` 轮。
- **PPO 交互**：每轮从训练集抽取场景，随机配置船长（`[50,75,...,500]` 十档），按 `advance_decision` 推进最多 `episode_limit` 个决策（默认由源 TCPA 延长，上限 1500）。
- **GAE 优势估计**（对非均匀时间步做了折扣修正）：

```text
discount = 0.99^(Δt/10)      trace = 0.95^(Δt/10)
δ_t = r_t + discount · V_{t+1} · (1 − done) − V_t
GAE_t = δ_t + discount · trace · (1 − done) · GAE_{t+1}
returns = GAE + V
```

- **PPO-Clip 更新**（`ppo_epochs=4`，batch=128，lr=3e-4，梯度裁剪 0.5）：

```text
ratio = exp(new_logp − old_logp)
policy_loss = −mean( min(ratio·Adv, clip(ratio, 0.8, 1.2)·Adv) )
value_loss  = 0.5 · mean( (returns − V)² )
loss = policy_loss + value_loss − 0.01 · entropy
```

- 缓冲区达到 512 步或训练结束时执行一次更新；`--resume` 会校验时钟、规则掩码配置、数据划分指纹与 RNG 状态。

### 12.5 权重校验

[checkpoint_metadata](usv_sim/ppo_agent.py) 写入契约版本、观测尺寸、动作数量与动作规格；[validate_checkpoint](usv_sim/ppo_agent.py) 在加载时逐项比对，不兼容则拒绝加载并提示 `retrain candidate-v4`。v1.6.2 未改变动作数/观测维度/网络结构，因此 v1.6.1 的 candidate-v4 权重仍可加载；更早的 candidate-v3 权重不兼容。

---

## 13. 策略权限门控（v1.6.2）

v1.6.2 的核心修复是：**PPO 不再在所有时刻自由选择动作**。PPO 仅在风险/规则触发的**实际避碰机动窗口**内拥有动作选择权；监视、回航、回航后巡航均由确定性规则 + MPC 基线控制。详见 [engine.py](usv_sim/engine.py) 的 [ppo_action_authorized](usv_sim/engine.py) / [prepare_decision](usv_sim/engine.py)。

```text
ppo_action_authorized = maneuver_active
                        AND (NOT route_recovery_active)
                        AND (NOT ODD degraded)
```

门控实现方式：当 PPO 未被授权时，把动作掩码收缩为**单例 `{baseline_action}`**（当前 MPC 评分最低的基线候选），而不是让 PPO 从整个可行集里自由选择：

```text
if not policy_action_authorized:
    mask = [i == baseline_action for i in range(ACTION_COUNT)]
```

不同阶段的策略来源（`policy_source`）与门控原因（`policy_gate_reason`）：

| 阶段 | 策略来源 | 门控原因 |
|---|---|---|
| 监视 monitor | `rule_mpc_monitor` | `monitor_phase_waits_for_rule_trigger` |
| 避碰机动 | `ppo_active_encounter`（或 `rule_mpc_active_encounter` 无 PPO 时） | `active_encounter_window` |
| 航线回归 | `rule_mpc_route_recovery` | `route_recovery_uses_rule_mpc` |
| 回航后巡航 | `rule_mpc_post_recovery` | `post_recovery_cruise_uses_rule_mpc` |
| ODD 降级 | `rule_baseline_odd_degraded` | — |

**训练采样、PPO 更新、部署推理共用同一掩码**，避免训练时学到部署时被禁止的动作。会遇开始、让清转入回归、新目标介入等事件会触发 [ `_force_policy_replan`](usv_sim/engine.py)，使保持的旧策略指令立即失效并强制重规划，而不是把旧舵令保留到下一个 10 s 决策点。

---

## 14. 航线回归控制

[planner.py](usv_sim/planner.py) 的 [ `_route_guidance`](usv_sim/planner.py) 实现**有界、横向误差感知的视线（pure-pursuit）引导**：

```text
lookahead（回归模式） = clamp(0.8 · u · recovery_horizon, 1.5·L, 8·L)
predicted_cross = signed_cross + cross_rate · anticipation_s   （anticipation ∈ [25, 150]s，随 τ_yaw）
correction = atan2(predicted_cross, lookahead)
correction 限制在 ±[22°, 36°]（随船长递减）
intercept_heading = route_heading + correction
```

关键设计：

- 回归模式的前视距离与 MPC 预测时域联动，使离散的 ±5° 动作候选能表达细小的回归角，避免选中“平行航行”的局部最优。
- 预测艏摇响应期间的横向位移并前馈修正，抑制大船因长艏摇滞后而横穿航线后的左右摆动。
- [ `_course_controller`](usv_sim/planner.py) 增加风浪艏摇扰动的前馈补偿，消除持续航向偏差。

回归完成判据（[ `_update_route_recovery`](usv_sim/engine.py)）**三项连续稳定 5 个控制步（10 s）**：

```text
1. 横向偏差 ≤ 0.1·L（船长）
2. 船首向与当前航段航向误差 ≤ 8°
3. |纵向速度 − 初始巡航航速| ≤ max(0.10, 0.02·V0)
```

任一条件失效即清零稳定计数；新避碰事件会清除上次回归完成标志。回航与回航后巡航锁定为任务初始巡航航速（`cruise_speed_locked`），只有新一轮避碰才重新开放战术加减速。

---

## 15. 奖励函数

[rewards.py](usv_sim/rewards.py) 的 [compute_reward](usv_sim/rewards.py) 提供**物理尺度的分项奖励**，交互仿真与训练共用（无 torch 依赖）。默认按 10 s 决策周期加权：

```text
scale = dt / 10
reward = time + progress + cross_track + rudder + rudder_rate + yaw
       + domain + rule + intervention + collision + route_return + destination
```

| 分量 | 公式（默认权重） | 含义 |
|---|---|---|
| `time` | `−0.02·scale` | 时间惩罚 |
| `progress` | `+0.08·Δs/(V_nom·10)` | 沿航线位移（位移型，非速度型） |
| `cross_track` | `−0.10·min(3, \|lat\|/1500)·scale` | 横向偏差 |
| `rudder` | `−0.06·\|δ\|/max_δ·scale` | 舵角惩罚 |
| `rudder_rate` | `−0.10·\|Δδ\|/(max_rate·dt)·scale` | 舵速惩罚 |
| `yaw` | `−0.04·\|r\|/max_r·scale` | 艏摇角速度惩罚 |
| `domain` | `−0.90·max(risk)·scale` | 安全域侵入（`risk = 1 − separation/SDA`） |
| `rule` | `−1.5·violation·scale` | 规则违反（含 R14/R15/R17 编码） |
| `intervention` | `−0.25·scale` | 安全干预惩罚 |
| `collision` | `−100`（**新发生**，不裁剪） | 船舶/障碍物碰撞 |
| `route_return` | `+3`（**单次事件**） | 避让后回归航线 |
| `destination` | `+10`（**单次事件**） | 到达航线终点 |

---

## 16. 安全监控与 ODD

### 16.1 短时域执行监控（safety.py）

[check_control](usv_sim/safety.py) 是**独立于候选评分函数**的执行监控：在下一个控制周期内以 1 s 步长做 swept 检查，若预测最小间距低于 SDA 或侵入障碍物安全带（`0.5·L + 100 m`），则触发干预、强制切换到规划器重新选择的基线候选。它共享同一动力学模型，**不构成形式化的独立冗余保障**。

### 16.2 ODD 运行设计域监控（odd.py）

[ODDMonitor](usv_sim/odd.py) 检查声明的运行包络：

| 条件 | 后果 |
|---|---|
| 感知系统无效 / 控制系统无效 | `takeover_required`（强制接管，停止仿真） |
| 能见度 < 1.0 NM | degraded（低能见度，SDApref 的 λ₁=2） |
| 风速 > 10 m/s / 海况 > 3 级 | degraded（禁止加速候选） |

ODD 降级时 PPO 被门控屏蔽，由确定性基线控制。

---

## 17. 数据集与数据划分

### 17.1 案例集

[dataset.py](usv_sim/dataset.py) 读取 `datasets/cases/` 下的会遇案例：

| 案例包 | 场景数 | 每场景目标数 |
|---|---:|---:|
| `1t` | 138 | 1 |
| `2t_0720` | 99 | 2 |
| `3t` | 31 | 3 |
| **合计** | **268** | — |

CSV 字段：`ownship_lat/lon`、`ownship_sog(KN)`、`ownship_cog(deg)`、`target_lat/lon`、`target_sog(KN)`、`target_cog(deg)`、`DCPA(NM)`、`TCPA(min)`、`meeting_situation`、`stage` 等。

**重要数据边界**：源 CSV 是**会遇初始状态快照**，不是逐时刻轨迹；不包含船长/船宽/舵机参数，也没有目标船的连续运动轨迹。因此：

- 经纬度被转换到本船为原点的局部平面坐标（等距圆柱近似，适合短时会遇）：`y = Δlat·111132`，`x = Δlon·111132·cos(lat0)`；
- 目标船后续运动由仿真器的恒速恒向模型推进；
- 船长由界面设定或训练随机化给出（50–500 m）；
- 风浪流与控制响应参数是仿真设定，非案例实测。

编码自适应：1T 为 UTF-8 BOM，2T/3T 为 GB18030/GBK，加载器依次尝试解码。

### 17.2 几何划分（data_split.py）

[split_cases](usv_sim/data_split.py) 使用**内容派生的几何指纹**做可复现划分，使近重复会遇几何归入同一集合：

```text
fingerprint = SHA256( 量化(本船航向 5°, 本船速度 0.5 m/s,
                         每目标相对位置 100 m, 航向 5°, 速度 0.5 m/s) )
分桶 = SHA256("geometry-v1:seed:key")[:16] / 2^64
  < 0.8 → train；< 0.9 → val；否则 → test
```

默认 `seed=7` 得到 **206 / 36 / 26**。仅训练集参与学习，验证/测试集保留在 checkpoint 中供评估。

---

## 18. 训练与评估

### 18.1 训练

```bat
python train_ppo.py --steps 150000 --device cuda \
  --dataset-root datasets/cases --out models/ppo_policy.pt
```

主要参数：`--steps`（交互步数）、`--episode-steps`/`--max-episode-steps`（回合长度）、`--warmup-epochs`（候选基线预热）、`--ppo-epochs`、`--seed`、`--resume`、`--no-rule-mask`/`--no-monitor`（离线消融）。

### 18.2 评估

```bat
python evaluate_ppo.py --checkpoint models/ppo_policy.pt --split test \
  --group ALL --out results/eval_test.csv
```

`evaluate_ppo.py` 复用已加载策略逐案例评估，并逐案例写入 CSV（避免中断丢失结果）。**评估成功**定义为：

```text
到达航线终点 且 无碰撞 且 无障碍物接触 且 无 ODD 接管
且 无 SDA 侵入 且 无规则违反步数
```

CSV 输出包含：结束状态、最小船间距离、SDA 违规目标、各目标最小会遇距离、最小安全域裕度、碰撞计数、不安全规划次数、舵指令变化次数、最大风险分数、单动作决策比例、规则违反步数、规划延迟 p95 等。

---

## 19. 测试

```bat
python -m pytest -q
```

本交付环境的验证结论（来自 [BUILD_INFO.txt](BUILD_INFO.txt)）：

- 核心/非航线测试：**130 项通过**（`--ignore=tests/test_route_recovery_regression.py`）；
- 航线/动力学回归：**6 项通过**（上传的 1T/2T/3T Scene 001 回航 + 风浪回航 + 500 m 艏摇能力 + 扰动前馈）；
- 合计 **136 项通过**（分组运行，非单次全量进程）；三个长上传案例单独运行；
- `compileall` 与 40 s 无界面冒烟通过。

---

## 20. 局限性与安全声明

- **运动模型**：当前为带舵机/艏摇/横荡/航速滞后的简化 3-DOF / Nomoto-like 研究模型，**不是由实船试航数据标定的完整 MMG**。
- **数据**：源 CSV 未提供船舶尺度，目标船长由用户配置或训练随机化给出；目标船后续运动由恒速恒向模型推进。
- **风险与规则**：风险分级、COLREGs 会遇识别、采样式 MPC、DCRC/DCQS 窗口均为**工程近似**，不声称完整标准符合性；附录 A–D 是资料性示例，未复现附录 A 全部 DCRC/DCQS 算法。
- **PPO**：warm-start 使用规则策略伪标签，不代表专家避碰决策真值；动作掩码只对本系统列出的会遇情况执行基本硬规则约束，不等于对 COLREGs 全部条款的法律合规证明；PPO 权重须经独立场景测试后才能用于研究结论。
- **未覆盖条款**：能见度不良 Rule 19、狭水道 Rule 9、分道通航 Rule 10、特殊船舶责任 Rule 18、声光信号与实际感知不确定性仍需相应输入模型与场景测试。

> **本工程仅供研究和仿真，不得直接连接真实船舶自动驾驶，也不得作为实船避碰安全保障依据。**

---

## 参考文件

- `1972国际海上避碰公约.pdf`（项目根目录，随附）
- `ccs标准.pdf`（GB/T 46851—2025，随附）
- [docs/](docs/) 下的设计说明：`COLREGS_RULE_LOGIC_V160.md`、`COLREGS_TACTICAL_WINDOW_V150.md`、`SDA_BOUNDARY_AND_FAILURE_V161.md`、`PPO_POLICY_GATING_V162.md`、`ROUTE_RECOVERY_AND_GEOMETRY_OPTIMIZATION.md`、`STANDARD_TRACEABILITY.md` 等