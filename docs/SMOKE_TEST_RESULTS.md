# v1.2.0 自动化与回归测试结果

测试日期：2026-11-03

## 1. 自动化测试

命令：

```bash
PYTHONPATH=. pytest -q
```

结果：**15 passed**。

Python 语法检查：`run.py`、`train_ppo.py`、`ui/main_window.py` 和 `usv_sim/*.py` 均通过 `py_compile`。

## 2. 268 个上传案例导入检查

每个案例均执行一次仿真步进：

| 指标 | 结果 |
|---|---:|
| 案例 CSV 总数 | 268 |
| 成功导入并执行一步 | 268 |
| 解析/仿真异常 | 0 |
| 一步内船体碰撞计数 | 0 |
| 一步内规划安全标志为 false 的次数 | 32 |
| 一步后的全案例最小中心距 | 1829.2 m |

“规划安全标志为 false”表示在规划预测时域内没有采样到完全满足当前 `SDA_pref` 缓冲约束的候选，不等价于已经发生实际碰撞。

## 3. 代表性长回合回归

设置：本船 200 m、目标船默认 100 m、`dt=10 s`，每个场景推进 240 步（2400 s）。路径比为实际航迹长度除以起终点直线距离，用于观察绕圈/振荡情况。

| 案例 | 最小实际中心距 | 碰撞计数 | 不安全预测步数 | 回归原航线标志 | 航迹长度/直线距离 |
|---|---:|---:|---:|---|---:|
| 1T / Scene 001 | 1605 m | 0 | 0 | 是 | 1.041 |
| 2T / Scene 001 | 1591 m | 0 | 5 | 是 | 1.020 |
| 3T / Scene 001 | 1467 m | 0 | 19 | 是 | 1.039 |

以上是三个代表性场景的确定性软件仿真结果，不代表 268 个场景全部经过完整长回合安全认证。2T/3T 的部分重规划时刻曾报告预测裕度不足，实际回合记录的最小距离高于这些场景初始会遇所对应的 SDA 示例值；`unsafe_prediction_count` 仍应作为后续调参和独立验证指标。

## 4. 船长范围基本检查

本船长度分别设为 50、100、200、300、400、500 m；每个设置使用 3T Scene 001 和目标船长度 `[500, 200, 50] m` 推进 20 步。所有设置下状态均保持有限值，未出现船体碰撞计数。该检查只验证尺度配置和数值稳定性，不代表每种尺度组合都满足 SDA。

## 5. PPO 训练流程冒烟测试

使用 CPU 完成小规模端到端测试：

```bash
python train_ppo.py --steps 12 --episode-steps 2 --max-episode-steps 20 \
  --warmup-epochs 1 --ppo-epochs 1 --device cpu \
  --out models/smoke_mask_test.pt
```

测试遍历全部 268 个案例完成 warm start 数据生成，执行 PPO 更新，保存 checkpoint，并由 PyTorch 重新加载成功。该测试验证训练管线，不是正式模型训练结果。

## 6. GPU 与 GUI 环境限制

当前构建环境的 PyTorch 为 `2.10.0+cpu`，`torch.cuda.is_available()` 为 `False`；指定 `--device cuda` 会按设计明确退出，而不是静默回退 CPU。GPU 安装脚本已加入 Windows 工程，但本次没有可用 CUDA GPU 进行实际反向传播性能验证。

当前构建容器未安装 PySide6，因此界面完成语法检查，尚未在本环境实际打开 Qt 窗口。请在目标 Windows/Anaconda 环境运行 `run.py` 验证最终显示。

## 7. 不应误读的结果

- 没有碰撞计数不等于满足所有 COLREGs 条款或标准所有 SDA 要求。
- `SDA_pref` 是带操纵裕度的理想距离缓冲；`SDA` 是数据库示例值，两者要区分。
- CSV 没有连续时间轨迹、行为和船舶尺寸；目标船后续轨迹由仿真模型生成。
- 当前船舶模型仍是可调简化 3-DOF/Nomoto-like 模型，并非由实船试航数据标定的完整 MMG。


## v1.3.0 regression tests (2026-11-03)

The following representative uploaded scenes were rerun after fixing encounter release:

| Scenario | Simulated steps | Minimum actual separation | Collisions | Unsafe MPC predictions |
|---|---:|---:|---:|---:|
| 1T / Scene 001 | 1500 (3000 s) | 1299.8 m | 0 | 0 |
| 2T / Scene 001 | 550 (1100 s) | 1349 m | 0 | 0 |
| 3T / Scene 001 | 650 (1300 s) | 1464 m | 0 | 0 |

For 1T / Scene 001, the target is a head-on encounter with source TCPA 16 min and the configured SDA is 1018.6 m. The previous build incorrectly released the manoeuvre at about 430 s while TCPA was still positive; the v1.3.0 build keeps the original head-on classification and manoeuvre active until CPA has passed and the target is safely receding. In the deterministic run, the manoeuvre was released at about 1186 s and minimum actual separation was 1299.8 m.

The 2T / 3T values are representative software-in-the-loop runs only. They are not proof that all 268 scenes meet SDA or COLREGs requirements. PySide6 is not installed in this build environment, so the GUI layout code was syntax-checked but not opened in a native Windows Qt session.

## v1.3.1 delivery checks (2026-11-04)

- `python -m pytest -q`: **18 passed**. Added a regression test proving that clearing tracks restarts ownship/target history at the current positions without changing vessel state.
- Python syntax compilation passed for `ui/main_window.py`, `usv_sim/engine.py`, and `evaluate_ppo.py`.
- PPO evaluation CLI smoke test: generated a temporary compatible PPO checkpoint, evaluated a copied 1T Scene 001 with a 3-step cap, and verified that the CSV was written with one row and 21 result columns. This validates checkpoint loading and CSV export wiring only; it is not a full-length scenario evaluation or a trained-policy performance result.
- The build environment still has no PySide6 installation, so the Qt window itself could not be launched here. Validate the final visual layout in the target Windows/Anaconda environment.


## v1.3.2 delivery checks (2026-11-04)

- Added a GUI action to unload the PPO policy without deleting the checkpoint file; scenario reset paths now pass `None` after unload so the old checkpoint is not silently reloaded.
- Added a dedicated off-map figure legend with vessel, route, track, obstacle, and prediction symbols. The previous figure-level legend is removed on each redraw to avoid duplicate legend accumulation.
- UI syntax compilation and the existing automated test suite were run for this revision. The build environment still lacks PySide6, so final visual confirmation must be done in the target Windows/Anaconda environment.


v1.3.3: Added manual polygon validation, map input interaction code syntax checks, and static-obstacle contact metrics. Qt GUI runtime validation depends on PySide6 being installed in the target Windows environment.

## v1.3.3 delivery checks (2026-11-04)

- Python syntax compilation passed for all project Python files.
- Automated tests: 22 passed, including simple polygon acceptance, self-intersection rejection, live scenario obstacle-list linkage, and static-obstacle contact counting.
- Headless simulation smoke test completed 4 steps without errors.
- PPO reward helper smoke test confirmed that a static-obstacle contact triggers the collision penalty.
- `evaluate_ppo.py` now records `静态障碍物接触计数` and treats static-obstacle contact as an unsuccessful evaluation outcome.
- PySide6 is not installed in this build environment, so the Qt window, Matplotlib mouse events, and visual layout could not be launched here; validate those interactions on the target Windows/Anaconda setup.
