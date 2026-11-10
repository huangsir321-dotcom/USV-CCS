"""Run a short simulation without the PySide6 GUI."""
import argparse
from usv_sim.scenarios import make_scenario
from usv_sim.engine import SimulationEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--length", type=float, default=200.0)
    parser.add_argument("--out", default="logs/headless_run.json")
    args = parser.parse_args()
    engine = SimulationEngine(make_scenario(args.targets, args.length))
    for _ in range(args.steps):
        engine.step()
    engine.export_log(args.out)
    print(f"完成 {len(engine.logs)} 步，仿真时间 {engine.metrics.elapsed_s:.1f}s")
    print(f"本船位置=({engine.own.x:.1f}, {engine.own.y:.1f})m，最小船间距离={engine.metrics.min_separation_m:.1f}m")
    print(f"船舶碰撞计数={engine.metrics.collision_count}，静态障碍物接触计数={engine.metrics.obstacle_collision_count}，预测不安全次数={engine.metrics.unsafe_prediction_count}")
    print(f"日志：{args.out}")


if __name__ == "__main__":
    main()
