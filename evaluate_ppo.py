from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

from usv_sim.dataset import list_cases, load_case
from usv_sim.engine import SimulationEngine
from usv_sim.ppo_agent import PPOPolicy
from usv_sim.data_split import split_cases, geometry_fingerprint


def _fmt(value, digits=3):
    if value is None:
        return ""
    try:
        number = float(value)
        return f"{number:.{digits}f}" if math.isfinite(number) else ""
    except (TypeError, ValueError):
        return str(value)


def evaluate_case(spec, policy, max_steps: int, own_length_m: float, target_length_m: float,
                  use_rule_mask=True, use_monitor=True):
    scenario = load_case(
        spec,
        own_length_m=own_length_m,
        default_target_length_m=target_length_m,
    )
    engine = SimulationEngine(scenario, None)
    # Reuse the already loaded policy for every case instead of loading the same
    # checkpoint hundreds of times. The network is read-only during evaluation.
    engine.policy = policy
    engine.use_rule_mask = use_rule_mask
    engine.use_safety_monitor = use_monitor
    engine.running = True
    max_risk_score = float(engine.last_snapshot.max_risk_score)
    started = time.perf_counter()
    steps = 0
    for steps in range(1, max_steps + 1):
        engine.step()
        max_risk_score = max(max_risk_score, float(engine.last_snapshot.max_risk_score))
        if not engine.running or engine.terminal:
            break
    wall_time_s = time.perf_counter() - started

    odd_takeover = any(
        isinstance(item, dict) and str(item.get("event", "")).startswith("ODD degraded")
        for item in engine.logs
    )
    collision = engine.metrics.collision_count > 0
    obstacle_collision = engine.metrics.obstacle_collision_count > 0
    completed = bool(engine.metrics.destination_reached)
    rule_violations = sum(bool(row.get('executed_rule_reasons'))
                          for row in engine.logs if 'executed_action' in row)
    sda_failure = engine.metrics.sda_violation_count > 0 or engine.metrics.domain_intrusion_s > 0
    success = (completed and not collision and not obstacle_collision and not odd_takeover
               and not sda_failure and rule_violations == 0)
    if collision:
        status = "发生船舶碰撞"
    elif obstacle_collision:
        status = "发生静态障碍物接触"
    elif odd_takeover:
        status = "ODD要求人工接管"
    elif sda_failure:
        status = "实际最小会遇距离小于SDA，避碰失败"
    elif completed:
        status = "安全到达航线终点" if success else "到达终点但违反规则"
    else:
        status = "达到最大步数，未完成"

    return {
        "案例组": spec.group,
        "场景编号": spec.scene_id,
        "案例文件": spec.path.name,
        "目标船数量": len(scenario.targets),
        "会遇类型": "；".join(x for x in scenario.metadata.get("meeting_situations", []) if x),
        "风险阶段标签": "；".join(x for x in scenario.metadata.get("stage_labels", []) if x),
        "评估成功": "是" if success else "否",
        "结束状态": status,
        "仿真步数": steps,
        "仿真时长_s": _fmt(engine.metrics.elapsed_s, 1),
        "本船初始航速_kn": _fmt(scenario.metadata.get("ownship_sog_kn"), 2),
        "最小船间距离_m": _fmt(engine.metrics.min_separation_m, 2),
        "SDA违规目标数": engine.metrics.sda_violation_count,
        "SDA违规目标": ";".join(engine.metrics.sda_violation_targets),
        "各目标最小会遇距离_m": ";".join(
            f"{key}:{_fmt(value, 2)}" for key, value in sorted(engine.metrics.target_min_separation_m.items())
        ),
        "碰撞计数": engine.metrics.collision_count,
        "静态障碍物接触计数": engine.metrics.obstacle_collision_count,
        "不安全规划次数": engine.metrics.unsafe_prediction_count,
        "舵指令变化次数": engine.metrics.heading_command_changes,
        "最大风险分数": _fmt(max_risk_score, 4),
        "避让后回归航线": "是" if engine.metrics.route_returned else "否",
        "到达航线终点": "是" if completed else "否",
        "ODD接管": "是" if odd_takeover else "否",
        "单案例耗时_s": _fmt(wall_time_s, 2),
        "策略模式": 'ppo' if policy.available else 'deterministic',
        "规则掩码": int(use_rule_mask),
        "末端监控": int(use_monitor),
        "安全域侵入时长_s": engine.metrics.domain_intrusion_s,
        "最小安全域裕度_m": _fmt(engine.metrics.minimum_domain_clearance_m),
        "安全干预次数": engine.metrics.intervention_count,
        "无解决策次数": engine.metrics.infeasible_decision_count,
        "单动作决策比例": _fmt(engine.metrics.singleton_decision_count/max(1,engine.metrics.decision_count)),
        "规则违反步数": rule_violations,
        "规划延迟_p95_s": _fmt(sorted(engine.metrics.planning_times_s)[
            min(len(engine.metrics.planning_times_s)-1,int(.95*len(engine.metrics.planning_times_s)))]
            if engine.metrics.planning_times_s else 0),
        "错误信息": "",
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy on 1T/2T/3T USV encounter cases.")
    parser.add_argument("--checkpoint", help="PPO .pt/.pth checkpoint")
    parser.add_argument("--baseline", action='store_true', help="Use deterministic selector on the same bank")
    parser.add_argument("--split", choices=('train','val','test','all'), default='test')
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=0, help="Optional positive case cap for smoke tests")
    parser.add_argument("--no-rule-mask", action='store_true', help="Offline simulation ablation")
    parser.add_argument("--no-monitor", action='store_true', help="Offline simulation ablation")
    parser.add_argument("--group", choices=("1T", "2T", "3T", "ALL"), default="ALL")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--own-length", type=float, default=200.0)
    parser.add_argument("--target-length", type=float, default=100.0)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    output = Path(args.out).expanduser().resolve()
    if not args.baseline and (checkpoint is None or not checkpoint.is_file()):
        print(f"ERROR checkpoint does not exist: {checkpoint}", flush=True)
        return 2
    if args.max_steps < 1:
        print("ERROR --max-steps must be >= 1", flush=True)
        return 2

    print("EVAL_INIT loading PPO checkpoint", flush=True)
    policy = PPOPolicy(str(checkpoint) if not args.baseline else None)
    if not args.baseline and not policy.available:
        print(f"ERROR {policy.load_error}; evaluation cancelled", flush=True)
        return 3

    group = None if args.group == "ALL" else args.group
    specs = list_cases(args.dataset_root, group)
    if args.split != 'all':
        manifest=policy.metadata.get('data_split') or {}
        if manifest:
            keys=set(manifest['cases'][args.split])
            specs=[s for s in specs if s.key in keys]
            if any(geometry_fingerprint(s)!=manifest['geometry_hashes'][s.key] for s in specs):
                print('ERROR dataset geometry differs from checkpoint',flush=True)
                return 4
        else:
            specs=split_cases(specs,args.seed)[args.split]
    if args.limit > 0:
        specs=specs[:args.limit]
    if not specs:
        print(f"ERROR no cases found for group {args.group}", flush=True)
        return 4

    print(
        f"EVAL_DATASET total={len(specs)} 1T={sum(s.group == '1T' for s in specs)} "
        f"2T={sum(s.group == '2T' for s in specs)} 3T={sum(s.group == '3T' for s in specs)}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "案例组", "场景编号", "案例文件", "目标船数量", "会遇类型", "风险阶段标签",
        "评估成功", "结束状态", "仿真步数", "仿真时长_s", "本船初始航速_kn",
        "最小船间距离_m", "SDA违规目标数", "SDA违规目标", "各目标最小会遇距离_m",
        "碰撞计数", "静态障碍物接触计数", "不安全规划次数", "舵指令变化次数",
        "最大风险分数", "避让后回归航线", "到达航线终点", "ODD接管",
        "单案例耗时_s", "错误信息",
        "策略模式", "规则掩码", "末端监控", "安全域侵入时长_s", "最小安全域裕度_m",
        "安全干预次数", "无解决策次数", "单动作决策比例", "规则违反步数", "规划延迟_p95_s",
    ]
    results = []
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, spec in enumerate(specs, 1):
            try:
                row = evaluate_case(spec, policy, args.max_steps, args.own_length, args.target_length,
                                    not args.no_rule_mask, not args.no_monitor)
            except Exception as exc:
                row = {
                    "案例组": spec.group,
                    "场景编号": spec.scene_id,
                    "案例文件": spec.path.name,
                    "评估成功": "否",
                    "结束状态": "案例运行异常",
                    "错误信息": f"{type(exc).__name__}: {exc}",
                }
            writer.writerow(row)
            file.flush()
            results.append(row)
            print(
                f"EVAL_CASE {index}/{len(specs)} {spec.group}/scene_{spec.scene_id} "
                f"status={row.get('结束状态', '')} success={row.get('评估成功', '否')}",
                flush=True,
            )
            print(f"EVAL_PROGRESS {index} {len(specs)}", flush=True)

    successes = sum(row.get("评估成功") == "是" for row in results)
    failures = len(results) - successes
    print(f"EVAL_COMPLETE total={len(results)} success={successes} failed={failures}", flush=True)
    print(f"EVAL_CSV {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
