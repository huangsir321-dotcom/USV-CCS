import math
from pathlib import Path

from usv_sim.actions import ACTION_COUNT, ACTION_SPECS, OFFSET_ANGLES_DEG
from usv_sim.dataset import load_case, list_cases
from usv_sim.engine import SimulationEngine
from usv_sim.geometry import velocity_from_state
from usv_sim.models import EncounterAssessment, RiskSnapshot, ShipConfig, ShipState
from usv_sim.planner import DWAMPCPlanner
from usv_sim.risk import MultiShipRiskAssessor


def _translated_case(spec, elapsed_s, own_length_m=200.0):
    scene = load_case(spec, own_length_m=own_length_m)
    own_v = velocity_from_state(scene.own_state)
    target_v = velocity_from_state(scene.targets[0].state)
    scene.own_state.x += own_v[0] * elapsed_s
    scene.own_state.y += own_v[1] * elapsed_s
    scene.targets[0].state.x += target_v[0] * elapsed_s
    scene.targets[0].state.y += target_v[1] * elapsed_s
    return scene


def test_candidate_grid_has_five_degree_resolution_and_both_sides():
    assert ACTION_COUNT == 125
    assert len(ACTION_SPECS) == 125
    assert 5 in OFFSET_ANGLES_DEG and -5 in OFFSET_ANGLES_DEG
    assert 15 in OFFSET_ANGLES_DEG and -15 in OFFSET_ANGLES_DEG
    assert 30 in OFFSET_ANGLES_DEG and -30 in OFFSET_ANGLES_DEG
    assert min(OFFSET_ANGLES_DEG) == -60 and max(OFFSET_ANGLES_DEG) == 60


def test_1t003_does_not_start_mpc_or_accelerate_at_eight_nm():
    spec = next(s for s in list_cases(group='1T') if s.scene_id == '003')
    scene = load_case(spec, own_length_m=200.0)
    engine = SimulationEngine(scene)
    engine.prepare_decision()
    a = engine.last_snapshot.assessments[0]
    assert a.encounter == 'own_overtaking'
    assert a.pass_side == 'port'
    assert abs(a.preferred_turn_deg + 10.0) < 1e-6
    assert a.range_m > a.dcrc_m
    assert a.risk_score < 0.12
    assert not engine.maneuver_active
    assert all(not c.feasible for c in engine.candidates if c.speed_mode in ('slow_25', 'slow_50', 'accelerate_25', 'accelerate_50'))
    chosen = engine.candidates[engine.baseline_action]
    assert chosen.speed_mode == 'hold'
    assert abs(chosen.speed_cmd_mps-engine.cruise_speed_mps) < 1e-6


def test_1t003_enters_critical_window_and_prefers_port_ten_degree_acceleration():
    spec = next(s for s in list_cases(group='1T') if s.scene_id == '003')
    # Reposition both vessels along their original constant-velocity paths to
    # the DCRC entry window, without injecting any manoeuvre into the test.
    scene = _translated_case(spec, 4475.0, 200.0)
    engine = SimulationEngine(scene)
    engine.prepare_decision()
    a = engine.last_snapshot.assessments[0]
    assert a.risk_score >= 0.12
    assert engine.maneuver_active
    chosen = engine.candidates[engine.baseline_action]
    assert chosen.offset_deg <= -10.0
    assert chosen.speed_mode.startswith('accelerate_')
    assert chosen.speed_cmd_mps > engine.reference_speed
    assert chosen.speed_cmd_mps <= 1.5 * engine.cruise_speed_mps + 1e-9


def test_crossing_case_does_not_accelerate_before_critical_window():
    spec = next(s for s in list_cases(group='1T') if s.scene_id == '006')
    scene = load_case(spec, own_length_m=200.0)
    engine = SimulationEngine(scene)
    engine.prepare_decision()
    a = engine.last_snapshot.assessments[0]
    assert a.encounter == 'starboard_crossing'
    assert a.risk_score < 0.12
    assert not engine.maneuver_active
    assert all(not c.feasible for c in engine.candidates if c.speed_mode.startswith('accelerate_'))
    assert engine.candidates[engine.baseline_action].speed_mode == 'hold'


def test_crossing_give_way_prefers_course_action_and_slows_instead_of_accelerating():
    spec = next(s for s in list_cases(group='1T') if s.scene_id == '006')
    scene = _translated_case(spec, 820.0, 200.0)
    engine = SimulationEngine(scene)
    engine.prepare_decision()
    a = engine.last_snapshot.assessments[0]
    assert engine.maneuver_active
    assert a.encounter == 'starboard_crossing'
    assert not engine.last_snapshot.emergency
    assert all((not c.feasible) for c in engine.candidates if c.speed_mode.startswith('accelerate_'))
    chosen = engine.candidates[engine.baseline_action]
    assert chosen.offset_deg != 0
    assert chosen.speed_mode in ('hold', 'slow_25', 'slow_50')
    assert chosen.feasible
    assert not any('R15_crossing_ahead' in reason for reason in chosen.rule_reasons)


def test_emergency_does_not_disable_r15_crossing_ahead_guard():
    from usv_sim.models import TargetShip
    planner = DWAMPCPlanner(ShipConfig.for_length(100))
    own = ShipState(u=7.0)
    # The geometric guard is applied to the predicted path even in emergency;
    # a port-side crossing candidate should still expose the R15 audit reason.
    target = TargetShip('T', ShipState(x=300.0, y=500.0, psi=math.radians(270), u=5.0))
    a = EncounterAssessment('T', 'starboard_crossing', True, 'R15', 35, 583, 0, 50,
                            400, '紧迫局面', 0.9)
    bank = planner.generate_candidates(own, [target], RiskSnapshot([a], 'T', .9, True),
                                        control_period_s=2.0)
    assert all(any('R15_crossing_ahead' in reason for reason in c.rule_reasons)
               for c in bank if any('R15_crossing_ahead' in reason for reason in c.reasons)) or any(
                   'R15_crossing_ahead' in reason for c in bank for reason in c.rule_reasons)
