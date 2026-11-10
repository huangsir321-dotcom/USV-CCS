from __future__ import annotations
import math
from .models import ShipConfig, ShipState, TargetShip, Scenario, StaticObstacle, EnvironmentConfig


def make_scenario(target_count: int = 1, own_length_m: float = 200.0, seed: int = 7) -> Scenario:
    """Generate deterministic mixed encounter scenes with 1-3 targets."""
    target_count = max(1, min(3, int(target_count)))
    cfg = ShipConfig.for_length(own_length_m, "OV")
    own = ShipState(x=0.0, y=0.0, psi=0.0, u=7.0, v=0.0)
    candidates = [
        TargetShip("TV1", ShipState(0.0, 4500.0, math.pi, 7.2), 200.0, 32.0, "bulk"),
        TargetShip("TV2", ShipState(2500.0, 2500.0, 1.5 * math.pi, 7.0), 100.0, 18.0, "container"),
        TargetShip("TV3", ShipState(-4500.0, 4500.0, 0.5 * math.pi, 7.0), 150.0, 24.0, "other"),
    ]
    targets = candidates[:target_count]
    route = [(0.0, 12000.0)]
    return Scenario(
        name=f"1V{target_count} 混合会遇",
        own_config=cfg,
        own_state=own,
        targets=targets,
        route=route,
        description="本船北向航行；目标船包含对遇与交叉会遇，用于验证多船风险评估和局部规划",
        dt_s=2.0,
        obstacles=[],
        environment=EnvironmentConfig(),
    )


def make_uncoordinated_scenario(target_count: int = 1, own_length_m: float = 200.0) -> Scenario:
    sc = make_scenario(target_count, own_length_m)
    if sc.targets:
        sc.targets[0].behavior = "turn_after_delay"
        sc.targets[0].turn_start_s = 150.0
        sc.targets[0].turn_rate_deg_s = -0.5
    sc.name = f"1V{len(sc.targets)} 行动不协调"
    sc.description = "目标船在仿真过程中改变航向，检验运动预测更新和应变规划"
    return sc


def add_demo_obstacle(scenario):
    """Add a polygonal obstruction aligned with the scenario's planned route."""
    ux, uy = math.sin(scenario.own_state.psi), math.cos(scenario.own_state.psi)
    # Perpendicular unit vector, used as the cross-track axis.
    vx, vy = uy, -ux
    cx = scenario.own_state.x + 6800.0 * ux
    cy = scenario.own_state.y + 6800.0 * uy
    half_along, half_cross = 350.0, 300.0
    vertices = [
        (cx - half_along * ux - half_cross * vx, cy - half_along * uy - half_cross * vy),
        (cx + half_along * ux - half_cross * vx, cy + half_along * uy - half_cross * vy),
        (cx + half_along * ux + half_cross * vx, cy + half_along * uy + half_cross * vy),
        (cx - half_along * ux + half_cross * vx, cy - half_along * uy + half_cross * vy),
    ]
    scenario.obstacles.append(StaticObstacle("OBS-1", vertices))
    scenario.name += " + Polygon obstacle"
    return scenario
