"""Regression tests for the v1.4.2 rule-only route-recovery controller."""
import math
import pytest

from usv_sim.dataset import default_dataset_root, list_cases, load_case
from usv_sim.dynamics import Simplified3DOFDynamics
from usv_sim.engine import SimulationEngine
from usv_sim.models import EnvironmentConfig, ShipConfig, ShipState, RiskSnapshot, TargetShip
from usv_sim.planner import DWAMPCPlanner


@pytest.mark.parametrize("group", ["1T", "2T", "3T"])
def test_uploaded_encounter_cases_recover_instead_of_stalling_off_route(group):
    """Reproduce the former ~100 m plateau after the uploaded Scene 001 pass."""
    spec = next(s for s in list_cases(default_dataset_root(), group) if s.scene_id == "001")
    engine = SimulationEngine(load_case(spec, own_length_m=200))

    for _ in range(1600):
        engine.step()
        if engine.metrics.route_returned:
            break

    assert engine.metrics.route_returned, (
        f"{group} did not return: stage={engine.navigation_stage}, "
        f"cross_track={engine.route_cross_track_distance_m():.1f} m, "
        f"threshold={engine.route_recovery_tolerance_m:.1f} m"
    )
    assert engine.route_cross_track_distance_m() <= engine.route_recovery_tolerance_m
    assert engine.route_heading_error_deg <= 8.0
    assert engine._route_recovery_stable_steps == engine.route_recovery_stable_required


def test_recovery_candidate_reduces_cross_track_error_under_wind_and_waves():
    config = ShipConfig.for_length(200)
    environment = EnvironmentConfig()
    route_heading = math.radians(-21.3)
    ux, uy = math.sin(route_heading), math.cos(route_heading)
    nx, ny = -uy, ux
    along_track, signed_cross = 25000.0, -63.0
    own = ShipState(
        x=along_track * ux + signed_cross * nx,
        y=along_track * uy + signed_cross * ny,
        psi=route_heading,
        u=11.956,
    )
    far_target = TargetShip(
        "FAR", ShipState(x=100000.0, y=100000.0, psi=0.0, u=7.0), active=True
    )
    planner = DWAMPCPlanner(config)
    bank = planner.generate_candidates(
        own, [far_target], RiskSnapshot([], None, 0.0, False),
        ux * 100000.0, uy * 100000.0, 2.0, [], environment,
        3400.0, (0.0, 0.0), route_recovery=True,
    )
    selected = planner.select_candidate(bank)
    end_x, end_y = selected.predicted_path[-1]
    final_cross = end_x * nx + end_y * ny
    heading_correction = (selected.course_cmd_rad - route_heading + math.pi) % (2 * math.pi) - math.pi

    assert signed_cross * heading_correction > 0.0  # steer toward the route
    assert abs(final_cross) < abs(signed_cross)


def test_500m_environment_yaw_disturbance_stays_inside_available_yaw_authority():
    config = ShipConfig.for_length(500)
    state = ShipState(psi=math.radians(45.0), x=1234.0, u=6.5)
    disturbance = Simplified3DOFDynamics.yaw_disturbance_acceleration(
        state, EnvironmentConfig(), elapsed_s=123.0
    )
    equivalent_steady_yaw_rate = abs(disturbance * config.yaw_tau_s)
    available_yaw_rate = math.radians(config.max_yaw_rate_deg_s)
    assert equivalent_steady_yaw_rate < available_yaw_rate


def test_course_controller_applies_environment_disturbance_feedforward():
    config = ShipConfig.for_length(200)
    environment = EnvironmentConfig(wind_speed_mps=10.0, wind_dir_rad=0.0, wave_level=0.0)
    state = ShipState(psi=math.pi / 2.0, u=7.0)
    planner = DWAMPCPlanner(config)
    disturbance = planner.dynamics.yaw_disturbance_acceleration(state, environment, 0.0)
    rudder = planner._course_controller(state, state.psi, environment, 0.0)
    assert disturbance < 0.0
    assert rudder > 0.0  # opposite sign rudder rejects the environmental yaw moment
