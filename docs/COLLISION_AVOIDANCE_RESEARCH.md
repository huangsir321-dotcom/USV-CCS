# 船舶避碰框架：规范、论文与工程证据

检索日期：2026-11-04。本文是重构设计的证据笔记，不是本项目的符合性结论。下列建议属于工程推导，必须通过本项目独立测试验证。论文作者报告的实验结果不代表本项目已复现；厂商产品声明不等于完整避碰算法的认证。

## 1. 规范边界

GB/T 46851—2025《智能船舶 避碰系统技术要求及测试方法》已发布且现行，发布日期2025-12-31、实施日期2026-07-01。官方公开目录足以确认这些元数据，不能替代标准正文。附录算法、表格、测试场景需与项目所持原文逐项核对，不能依据网上摘要补写成“标准公式”。[国家标准信息公共服务平台](https://std.samr.gov.cn//gb/search/gbDetailed?id=473EBB99D667455EE06397BE0A0ABB9A)

本地原文核对补充：项目的 `ccs标准.pdf` 实际是上述国家标准；第1章适用开阔水域，附录A～D均为资料性示例。正文7.1允许综合风险评估，附录A是一种右舷交叉示例算法，并非唯一强制算法；7.3.2要求避让最小距离不小于设定SDA。附录C提供比当前五类更细的会遇子类。8.2.1和8.3要求相应目标船具备避让能力，不能把常速目标实验视为全部标准场景。详见 [项目审计](FRAMEWORK_AUDIT_AND_REDESIGN.md)。

COLREGs相应规则不是五个互斥的操纵命令：13为追越责任，14为对遇，15为交叉相遇，16为让路船行动，17为直航船分阶段行动；这些规定的适用背景包括互见条件。受限能见度应考虑19，特殊船舶责任应考虑18。[IMO公约说明](https://www.imo.org/en/about/conventions/pages/colreg.aspx)

应将“会遇关系”“让路责任”“风险阶段”“操纵动作”分开建模。例如，本船追越 + 让路 + 存在风险 + 右转并加速，是四个维度，不能仅用第六个状态“加速追越”替代。追越义务应持续至驶过并让清；22.5°正横后对应从目标船船首方向测得超过112.5°，不是本船看目标的方位角。核对原文必须使用国际规则栏，不能混入美国内河规则例外。[USCG国际/内河规则手册](https://www.navcen.uscg.gov/sites/default/files/pdf/navRules/Handbook/CG_NAV_Rules_29APR2020.pdf)

规则8要求及时、明显、有效的行动；规则15要求右舷有他船的让路船在情况许可时避免横越他船前方；16要求尽可能及早大幅行动；17不能简化成直航船永远保持不动。COLREGs不提供统一的安全速度或通过距离，阈值应结合船型、操纵性、环境和预测不确定性标定。[USCG规则行动摘要](https://www.dco.uscg.mil/Portals/9/DCO%20Documents/5p/CG-5PC/CG-CVC/CVC3/references/Rules_of_Road_Quick_Reference.pdf)、[USCG安全速度/距离说明](https://www.navcen.uscg.gov/navigation-rules-faqs)

## 2. 原始论文证据与可借鉴部分

| 来源 | 证据与边界 | 对本项目的设计启发 |
|---|---|---|
| Johansen、Cristofaro、Perez，Ship Collision Avoidance Using Scenario-Based Model Predictive Control | 以有限候选操纵行为及预测危险评估进行滚动选择；属于场景式预测控制路线，并非必须在线求解连续非线性优化。[作者论文PDF](https://torarnj.folk.ntnu.no/colregs_cams.pdf) | 候选采样加滚动预测可以称采样式MPC，但必须公开候选、预测、代价和约束；不能仅凭名称声称是完整NMPC。 |
| Eriksen、Breivik，2019，Short-term ASV Collision Avoidance with Static and Moving Obstacles | BC-MPC加入基于占据栅格的静态障碍代价，改善操纵摆动；作者报告峡湾实尺度试验，静态障碍为虚拟，动态障碍为真实且雷达跟踪。[原论文](https://arxiv.org/abs/1907.04877) | 结合静态与动态障碍、减少反复改向；把“规则约束”与“偏好”区分，保留试验范围说明。 |
| Tran、Johansen、Negenborn，2024预印本，2025修订，Distributed MPC for autonomous ships on inland waterways with collaborative collision avoidance | 两层框架、规则与算法分离、通过ADMM协商意图；摘要报告的是仿真结果，且环境是内河交通。[原论文](https://arxiv.org/abs/2403.00554) | 规则生成约束，不散落在训练/推理/规划三个实现中；协同意图仅适用于确有通信和协议的船，不能假设所有目标会配合。 |
| Schulman等，2017，Proximal Policy Optimization Algorithms | 提出限制策略更新幅度的代理目标及多轮小批量更新；基础任务验证并非船舶避碰验证。[原论文](https://arxiv.org/abs/1707.06347) | PPO负责优化可行策略中的效率与舒适性；奖励大罚分不构成运行时安全保证。 |
| Huang、Ontañón，A Closer Look at Invalid Action Masking in Policy Gradient Algorithms | 对状态相关掩码后的策略梯度给出理论解释，并在游戏环境评估，不能直接推出海上安全。[原论文](https://arxiv.org/abs/2006.14171) | 采样、保存旧log-prob及PPO更新均使用同一时刻保存的掩码；不能采样时mask、训练时恢复全动作softmax。 |
| Alshiekh等，AAAI 2018，Safe Reinforcement Learning via Shielding | 区分动作前提供安全动作集与动作后纠正；其保证依赖安全规格和环境抽象。[原论文](https://ojs.aaai.org/index.php/AAAI/article/download/11797/11656) | 一个统一安全模型同时支持动作mask与末端校核；仅必要时干预，记录原因。几何启发式筛选不应宣称为形式验证shield。 |

## 3. 掩码、PPO与硬规则的接口建议

以下为本项目建议公式，不是COLREGs条文公式，也不是已经实装的代码：

\[
\pi_\theta^M(a\mid s)=\frac{M(s,a)\exp z_\theta(s,a)}{\sum_b M(s,b)\exp z_\theta(s,b)}.
\]

\[
L^{clip}=\mathbb E_t\left[\min\left(\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right],
\quad \rho_t=\frac{\pi_\theta^{M_t}(a_t\mid s_t)}{\pi_{old}^{M_t}(a_t\mid s_t)}.
\]

mask只剔除在当前适用规则、动力学及预测假设下不可接受的候选，不能将“通常偏好右转”无条件编码成“所有左转永不可用”。多个目标的约束应同时检查，若交集为空，进入显式应急/最小风险选择并报告不可行；不能悄悄开放全部动作或把停船当作任何场合都安全。

末端校核采用实际控制候选和更新的状态预测，覆盖执行器饱和、静态障碍、新出现目标等。应记录 `proposed_action`、`executed_action`、干预原因、有效动作数、干预率以及原始/执行后轨迹差异。PPO仍可在多个可行候选间学习；若多数状态只剩一个候选，或不同策略经过下层映射得到同一控制，PPO对闭环的边际影响会很小。这是需以消融验证的现象，不能只用训练reward上涨证明有效。

建议比较规则基线、规则+规划、PPO+同一规划、PPO+mask+校核，使用相同场景、动力学、随机种子与预算。分别报告碰撞、船舶领域侵入、规则阶段违规、到达率、超时、最小裕度、航程/时间/能耗、操纵反复、规划延迟及干预率。训练集与独立测试集按场景族/几何参数分组，避免只用同一268案例训练后又评估它们。

## 4. 工业实现能支持什么结论

DNV的SAFEMATE项目与Torghatten、Kongsberg和NTNU合作，涉及未标注障碍感知、驾驶员告警和保障方法；官方报告桥楼仿真与Bastø VI实船条件测试。它支持“感知—决策支持—人机监督—测试保障”的工程路线，不能作为本项目PPO已获认证的证据。[DNV项目页](https://www.dnv.com/research/maritime/ship-autonomy/safemate/)

Sea Machines的SM300-SP产品页列出GNSS/IMU、雷达、AIS、深度等传感器集成、远程指挥、危险响应和告警。这表明产品交付包含状态感知、接口和失效响应，不只有避碰策略网络。页面的class-approved表述针对硬件外壳，不能扩大为所有自主避碰软件完整COLREG认证；页面未公开证明采用PPO。[厂商产品页](https://sea-machines.com/sm300-sp/)

挪威向IMO提交的MASS项目情况报告描述情景感知、避碰计算机、专用人机界面、岸基监督及后备状态等功能分工。它是具体项目进展及监管沟通材料，不能推论所有船型已获无人化许可，也不能据此复制隐藏算法。[MSC 108/INF.16，挪威官方提交文件（IALA托管）](https://www.iala.int/content/uploads/2025/03/MSC-108-INF.16-Ongoing-MASS-projects-in-Norway-Norway.pdf)

## 5. 推荐的重构方向

本研究建议采用“统一规则语义 + 可验证规划基线 + 可替换学习层”：状态估计及不确定性 → 多目标会遇/责任/阶段管理 → 统一可行候选与原因码 → PPO选择高层操纵意图 → 采样式预测规划或NMPC → 独立末端安全监控 → 船舶控制。感知、规则、策略、规划、控制、监控应有清晰接口；训练与部署共用规则和执行链。

先建立可复现的规则+规划基线与场景测试，再判断PPO是否带来效率收益。若PPO不能优于该基线，应允许移除或降为可选实验模块。若后续需要实船研究，增加经过标定的操纵模型、时间同步/延迟、目标意图不确定性、传感器丢失、ODD退出和人工/远程接管；仅靠上述软件分层不能证明实船安全。
