import math
from usv_sim.models import ShipConfig, ShipState
from usv_sim.dynamics import Simplified3DOFDynamics
from usv_sim.scenarios import make_scenario
from usv_sim.risk import MultiShipRiskAssessor
from usv_sim.sda import sda_m


def test_dynamics_state_stays_finite():
    cfg = ShipConfig.for_length(200)
    dyn = Simplified3DOFDynamics(cfg)
    s = ShipState(u=7.0)
    for _ in range(100):
        s = dyn.step(s, math.radians(20), 6.0, 1.0)
    assert all(math.isfinite(x) for x in [s.x, s.y, s.psi, s.u, s.v, s.r, s.delta])
    assert abs(s.delta) <= math.radians(cfg.max_rudder_deg) + 1e-9


def test_multi_target_risk_has_all_targets():
    sc = make_scenario(3, 200)
    result = MultiShipRiskAssessor().assess(sc.own_state, sc.own_config, sc.targets)
    assert len(result.assessments) == 3
    assert result.focus_target_id in {"TV1", "TV2", "TV3", None}


def test_sda_is_positive_and_type_adjusted():
    bulk, _ = sda_m(200, 100, "head_on", ship_type="bulk")
    lng, _ = sda_m(200, 100, "head_on", ship_type="lng_chemical")
    assert bulk > 0 and lng > bulk


def test_static_polygon_geometry():
    from usv_sim.obstacles import point_in_polygon, distance_to_polygon
    poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon(5, 5, poly)
    assert not point_in_polygon(15, 5, poly)
    assert abs(distance_to_polygon(15, 5, poly) - 5.0) < 1e-6


def test_engine_one_step():
    from usv_sim.engine import SimulationEngine
    e = SimulationEngine(make_scenario(2, 200))
    e.step()
    assert e.metrics.elapsed_s == e.dt
    assert len(e.own_track) == 2


def test_odd_degraded_on_low_visibility():
    from usv_sim.models import EnvironmentConfig
    from usv_sim.odd import ODDMonitor
    env = EnvironmentConfig(visibility_nm=0.4)
    status = ODDMonitor().evaluate(env)
    assert status.degraded and not status.normal
    assert not status.takeover_required


def test_load_default_json_scenario():
    from pathlib import Path
    from usv_sim.scenario_io import load_scenario_json
    path = Path(__file__).resolve().parents[1] / "configs" / "default_scenario.json"
    scenario = load_scenario_json(path)
    assert len(scenario.targets) == 3
    assert scenario.own_config.length_m == 200


def test_manual_polygon_validation_accepts_simple_polygon():
    from usv_sim.obstacles import validate_polygon
    valid, reason = validate_polygon([(0, 0), (100, 0), (100, 80), (0, 80)])
    assert valid, reason


def test_manual_polygon_validation_rejects_self_intersection():
    from usv_sim.obstacles import validate_polygon
    valid, reason = validate_polygon([(0, 0), (100, 100), (0, 100), (100, 0)])
    assert not valid
    assert "交叉" in reason


def test_engine_obstacles_list_is_live_for_new_user_obstacles():
    from usv_sim.engine import SimulationEngine
    from usv_sim.models import StaticObstacle
    scenario = make_scenario(1, 200)
    engine = SimulationEngine(scenario)
    obstacle = StaticObstacle("USR-OBS-001", [(500, 500), (800, 500), (800, 800), (500, 800)])
    engine.obstacles.append(obstacle)
    assert scenario.obstacles is engine.obstacles
    assert scenario.obstacles[-1].obstacle_id == "USR-OBS-001"


def test_engine_counts_contact_with_static_polygon_obstacle():
    from usv_sim.engine import SimulationEngine
    from usv_sim.models import StaticObstacle
    scenario = make_scenario(1, 200)
    scenario.obstacles.append(StaticObstacle(
        "USR-OBS-TEST", [(-1000, -1000), (1000, -1000), (1000, 1000), (-1000, 1000)]
    ))
    engine = SimulationEngine(scenario)
    engine.step()
    assert engine.metrics.obstacle_collision_count >= 1


def test_circle_polygon_vertices_preserve_requested_radius():
    from usv_sim.obstacles import circle_polygon_vertices, distance_to_polygon
    center = (120.0, -45.0)
    radius = 250.0
    vertices = circle_polygon_vertices(center, radius, segments=64)
    assert len(vertices) == 64
    assert all(math.dist(center, point) >= radius - 1e-8 for point in vertices)
    assert distance_to_polygon(center[0], center[1], vertices) == 0.0
    # A point outside the nominal circle remains outside its filled region.
    assert not __import__("usv_sim.obstacles", fromlist=["point_in_polygon"]).point_in_polygon(
        center[0] + radius * 1.5, center[1], vertices
    )


def test_circle_static_obstacle_metadata_survives_copy():
    from usv_sim.models import StaticObstacle
    from usv_sim.obstacles import circle_polygon_vertices
    obstacle = StaticObstacle(
        "USR-CIRCLE-001", circle_polygon_vertices((10, 20), 75),
        "circle", (10.0, 20.0), 75.0,
    )
    copied = obstacle.copy()
    assert copied is not obstacle
    assert copied.shape_type == "circle"
    assert copied.center == (10.0, 20.0)
    assert copied.radius_m == 75.0
    assert copied.vertices == obstacle.vertices


def test_circle_polygon_vertices_reject_invalid_radius():
    import pytest
    from usv_sim.obstacles import circle_polygon_vertices
    with pytest.raises(ValueError):
        circle_polygon_vertices((0, 0), 0)


def test_engine_counts_contact_with_circle_obstacle():
    from usv_sim.engine import SimulationEngine
    from usv_sim.models import StaticObstacle
    from usv_sim.obstacles import circle_polygon_vertices
    scenario = make_scenario(1, 200)
    center = (scenario.own_state.x, scenario.own_state.y)
    radius = 800.0
    vertices = circle_polygon_vertices(center, radius, segments=64)
    scenario.obstacles.append(StaticObstacle(
        "USR-CIRCLE-TEST", vertices, "circle", center, radius
    ))
    engine = SimulationEngine(scenario)
    engine.step()
    assert engine.metrics.obstacle_collision_count >= 1


def test_route_recovery_threshold_is_exactly_tenth_of_ownship_length():
    from usv_sim.engine import SimulationEngine

    scenario = make_scenario(1, 200)
    engine = SimulationEngine(scenario)
    assert engine.route_recovery_tolerance_m == 20.0
    assert engine.is_within_route_recovery_corridor(20.0)
    assert not engine.is_within_route_recovery_corridor(20.01)


def test_route_recovery_threshold_scales_with_ownship_length():
    from usv_sim.engine import SimulationEngine

    scenario = make_scenario(1, 50)
    engine = SimulationEngine(scenario)
    assert engine.route_recovery_tolerance_m == 5.0
    assert engine.is_within_route_recovery_corridor(4.99)
    assert not engine.is_within_route_recovery_corridor(5.01)


def test_route_cross_track_distance_uses_route_direction():
    from usv_sim.engine import SimulationEngine

    scenario = make_scenario(1, 200)
    engine = SimulationEngine(scenario)
    # Move 12 m perpendicular to the planned route, independent of its heading.
    x = engine.route_start_x - engine.route_uy * 12.0
    y = engine.route_start_y + engine.route_ux * 12.0
    assert abs(engine.route_cross_track_distance_m(x, y) - 12.0) < 1e-8
    assert engine.is_within_route_recovery_corridor(12.0)


def test_segment_circle_distance_uses_exact_segment_geometry():
    from usv_sim.planner import segment_circle_distance

    assert segment_circle_distance((-10, 0), (10, 0), (0, 0), 5) == 0.0
    assert abs(segment_circle_distance((-10, 10), (10, 10), (0, 0), 5) - 5.0) < 1e-9
    assert abs(segment_circle_distance((-10, 0), (-10, 0), (0, 0), 5) - 5.0) < 1e-9
    assert segment_circle_distance((-10, 5), (10, 5), (0, 0), 5) == 0.0


def test_route_recovery_requires_alignment_and_five_stable_control_steps():
    from usv_sim.engine import SimulationEngine

    engine = SimulationEngine(make_scenario(1, 200))
    engine.targets = []
    engine.route_recovery_active = True
    engine.own.x = engine.route_recovery_tolerance_m * 0.5
    route_heading = math.atan2(engine.route_ux, engine.route_uy)

    # Being inside the 0.1L corridor is not enough when the heading is not aligned.
    engine.own.psi = route_heading + math.radians(12.0)
    assert not engine._update_route_recovery()
    assert engine._route_recovery_stable_steps == 0
    assert not engine.metrics.route_returned

    engine.own.psi = route_heading
    for expected in range(1, engine.route_recovery_stable_required):
        assert not engine._update_route_recovery()
        assert engine._route_recovery_stable_steps == expected
    assert engine._update_route_recovery()
    assert engine.metrics.route_returned
    assert not engine.route_recovery_active


def test_route_recovery_stability_counter_resets_when_outside_corridor():
    from usv_sim.engine import SimulationEngine

    engine = SimulationEngine(make_scenario(1, 200))
    engine.targets = []
    engine.route_recovery_active = True
    engine.own.x = engine.route_recovery_tolerance_m * 0.5
    engine.own.psi = math.atan2(engine.route_ux, engine.route_uy)
    for _ in range(engine.route_recovery_stable_required - 1):
        assert not engine._update_route_recovery()
    assert engine._route_recovery_stable_steps == engine.route_recovery_stable_required - 1

    engine.own.x = engine.route_recovery_tolerance_m * 1.1
    assert not engine._update_route_recovery()
    assert engine._route_recovery_stable_steps == 0
    assert not engine.metrics.route_returned


def test_route_recovery_does_not_finish_destination_before_stable_return():
    from usv_sim.engine import SimulationEngine

    engine = SimulationEngine(make_scenario(1, 200))
    engine.targets = []
    engine.own.y = engine.route_y + 100.0
    engine.route_recovery_active = True
    assert not engine._check_destination()
    assert not engine.metrics.destination_reached


def test_recovered_status_is_cleared_by_a_new_avoidance_event():
    from usv_sim.engine import SimulationEngine
    from usv_sim.models import EncounterAssessment, RiskSnapshot

    engine = SimulationEngine(make_scenario(1, 200))
    engine.metrics.route_returned = True
    engine.last_snapshot = RiskSnapshot(
        [EncounterAssessment("TV1", "head_on", True, "COLREG", 0.0, 1000.0,
                             100.0, 300.0, 300.0, "碰撞危险", 0.5)],
        "TV1", 0.5, False,
    )
    engine._update_encounter()
    assert engine.maneuver_active
    assert not engine.metrics.route_returned


def test_polygon_threshold_check_uses_aabb_before_exact_edges(monkeypatch):
    import usv_sim.planner as planner_module
    from usv_sim.models import StaticObstacle

    def unexpected_exact_check(*args, **kwargs):
        raise AssertionError("A distant polygon must be rejected by the AABB test")

    monkeypatch.setattr(planner_module, "segment_polygon_distance", unexpected_exact_check)
    obstacle = StaticObstacle(
        "far-polygon", [(1000, 1000), (1100, 1000), (1100, 1100), (1000, 1100)]
    )
    distance = planner_module.segment_obstacle_distance(
        (0, 0), (10, 0), obstacle, early_exit_distance_m=100.0
    )
    assert distance > 100.0


def test_route_recovery_requires_speed_to_return_to_initial_cruise_setpoint():
    from usv_sim.engine import SimulationEngine
    engine = SimulationEngine(make_scenario(1, 200))
    engine.route_recovery_active = True
    engine.own.x = 0.0
    engine.own.psi = 0.0
    engine.own.u = engine.cruise_speed_mps + 0.5

    assert not engine._update_route_recovery()
    assert engine._route_recovery_stable_steps == 0

    engine.own.u = engine.cruise_speed_mps + engine.route_recovery_speed_tolerance_mps * 0.5
    for expected in range(1, engine.route_recovery_stable_required + 1):
        completed = engine._update_route_recovery()
        assert completed is (expected == engine.route_recovery_stable_required)
        assert engine._route_recovery_stable_steps == expected


def test_post_recovery_planner_holds_initial_speed_instead_of_accelerating():
    from usv_sim.engine import SimulationEngine
    scenario = make_scenario(1, 200)
    scenario.targets = []
    engine = SimulationEngine(scenario)
    initial_target = engine.cruise_speed_mps
    engine.metrics.route_returned = True
    engine.own.u = initial_target + 1.0

    engine.step()

    assert engine.cruise_speed_locked
    assert engine.last_plan is not None
    assert abs(engine.last_plan.speed_cmd_mps - initial_target) < 1e-9
    assert engine.own.u < initial_target + 1.0  # returning smoothly, not accelerating
    assert engine.own.u > initial_target       # speed lag respects vessel dynamics
