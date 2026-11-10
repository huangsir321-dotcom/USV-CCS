import math

from usv_sim.models import ShipConfig, ShipState, TargetShip, RiskSnapshot, EncounterAssessment
from usv_sim.planner import DWAMPCPlanner


def assessment(encounter, target_id='T', giveway=True):
    return EncounterAssessment(target_id, encounter, giveway, 'COLREG', 35, 4000,
                               10, 300, 300, '碰撞危险', 0.7)


def test_candidate_bank_contains_accelerating_overtake_both_sides():
    from usv_sim.actions import ACTION_SPECS
    planner = DWAMPCPlanner(ShipConfig.for_length(200))
    own = ShipState(u=7)
    bank = planner.generate_candidates(own, [], RiskSnapshot([], None, 0, False))
    assert len(bank) == len(ACTION_SPECS) == 125
    assert any(c.offset_deg > 0 and c.speed_cmd_mps > own.u for c in bank)
    assert any(c.offset_deg < 0 and c.speed_cmd_mps > own.u for c in bank)
    assert all(len(c.predicted_path) > 1 for c in bank if c.feasible)


def test_rule14_and_crossing_have_different_constraints():
    from usv_sim.rules import candidate_rule_reasons
    assert candidate_rule_reasons(0, 'slow', [assessment('head_on')])
    assert candidate_rule_reasons(0, 'slow', [assessment('starboard_crossing')])
    assert not candidate_rule_reasons(15, 'hold', [assessment('starboard_crossing')])
    assert candidate_rule_reasons(-20, 'hold', [assessment('head_on')])
    assert not candidate_rule_reasons(20, 'accelerate', [assessment('own_overtaking')])


def test_rules_check_all_targets_and_normal_stand_on():
    from usv_sim.rules import candidate_rule_reasons
    contacts = [assessment('own_overtaking'), assessment('head_on', 'T2')]
    reasons = candidate_rule_reasons(-20, 'hold', contacts)
    assert any('T2' in r for r in reasons)
    stand_on = assessment('port_crossing', giveway=False)
    stand_on.risk_score = 0.45
    assert candidate_rule_reasons(20, 'accelerate', [stand_on])
    assert not candidate_rule_reasons(0, 'hold', [stand_on])


def test_swept_relative_clearance_detects_between_sample_collision():
    from usv_sim.geometry import swept_separation
    assert swept_separation((-10, 0), (10, 0), (0, -10), (0, 10)) == 0


def test_no_feasible_candidate_is_explicit_and_never_marked_safe():
    from usv_sim.models import StaticObstacle
    p = DWAMPCPlanner(ShipConfig.for_length(200))
    obstacle = StaticObstacle('land', [(-10000,-10000),(10000,-10000),
                                      (10000,10000),(-10000,10000)])
    bank = p.generate_candidates(ShipState(), [], RiskSnapshot([], None, 0, False),
                                 obstacles=[obstacle])
    assert not any(c.feasible for c in bank)
    chosen = p.select_candidate(bank)
    assert not chosen.safe
    assert chosen.status == 'minimum_risk'


def test_selected_candidate_first_control_matches_prediction():
    p = DWAMPCPlanner(ShipConfig.for_length(200))
    own = ShipState()
    bank = p.generate_candidates(own, [], RiskSnapshot([], None, 0, False), control_period_s=2)
    selected = p.commit_candidate(next(c for c in bank if c.feasible and c.speed_mode == 'hold'))
    actual = p.dynamics.step(own, selected.rudder_cmd_rad, selected.speed_cmd_mps, 2)
    assert math.dist((actual.x, actual.y), selected.first_endpoint) < 1e-8


def test_rule_audit_remains_available_when_rule_mask_disabled():
    p=DWAMPCPlanner(ShipConfig.for_length(200))
    a=assessment('head_on')
    snapshot=RiskSnapshot([a],'T',a.risk_score,False)
    bank=p.generate_candidates(ShipState(),[],snapshot,use_rule_mask=False)
    assert any('R14' in r for r in bank[0].rule_reasons)
    assert bank[0].feasible


def test_whole_candidate_uses_execution_feedback_period():
    cfg=ShipConfig.for_length(50)
    p=DWAMPCPlanner(cfg)
    own=ShipState()
    bank=p.generate_candidates(own,[],RiskSnapshot([],None,0,False),control_period_s=2)
    chosen=next(c for c in bank if c.feasible and c.speed_mode == 'hold' and c.offset_deg == 0)
    state=own.copy()
    for _ in range(30):
        rudder=p._course_controller(state,chosen.course_cmd_rad)
        state=p.dynamics.step(state,rudder,chosen.speed_cmd_mps,2)
    assert math.dist((state.x,state.y),chosen.predicted_path[-1])<1e-8


def test_hold_speed_does_not_silently_resume_pre_manoeuvre_speed():
    p=DWAMPCPlanner(ShipConfig())
    own=ShipState(u=5)
    a=assessment('own_overtaking')
    bank=p.generate_candidates(own,[],RiskSnapshot([a],'T',a.risk_score,False),
                               reference_heading=0,reference_speed=7)
    assert bank[0].speed_cmd_mps==7


def test_polygon_aabb_broad_phase_skips_exact_edge_distance(monkeypatch):
    import usv_sim.planner as planner_module
    from usv_sim.models import StaticObstacle

    def unexpected_exact_check(*args, **kwargs):
        raise AssertionError("A distant polygon should be rejected by the AABB broad phase")

    monkeypatch.setattr(planner_module, "segment_polygon_distance", unexpected_exact_check)
    planner = DWAMPCPlanner(ShipConfig.for_length(200))
    far_obstacle = StaticObstacle(
        "far-land", [(100000, 100000), (101000, 100000),
                     (101000, 101000), (100000, 101000)]
    )
    bank = planner.generate_candidates(
        ShipState(), [], RiskSnapshot([], None, 0, False), obstacles=[far_obstacle]
    )
    assert len(bank) == 125
    assert all(candidate.feasible for candidate in bank if candidate.speed_mode == 'hold')
    assert all('obstacle:far-land' not in candidate.reasons for candidate in bank)


def test_route_guidance_anticipates_lateral_motion_before_crossing_route():
    planner = DWAMPCPlanner(ShipConfig.for_length(500))
    # The ship is 100 m east of a northbound route, but already moving west.
    # Predictive cross-track guidance should begin counter-steering before the
    # hull crosses the centreline instead of waiting for the sign to change.
    own = ShipState(x=100.0, y=0.0, psi=math.radians(340.0), u=6.5)
    route_heading, guidance, _, _, _ = planner._route_guidance(
        own, 0.0, 12000.0, (0.0, 0.0), 500.0, yaw_tau_s=80.0
    )
    assert abs(route_heading) < 1e-9
    assert math.degrees(guidance) > 0.0


def test_locked_cruise_speed_candidates_restore_mission_setpoint_and_block_speed_actions():
    planner = DWAMPCPlanner(ShipConfig.for_length(200))
    own = ShipState(u=8.2)
    bank = planner.generate_candidates(
        own, [], RiskSnapshot([], None, 0, False), control_period_s=2.0,
        cruise_speed_mps=7.0, lock_speed_to_cruise=True,
    )
    assert len(bank) == 125
    assert all(abs(c.speed_cmd_mps - 7.0) < 1e-9 for c in bank)
    assert all(c.feasible for c in bank if c.speed_mode == 'hold')
    assert all((not c.feasible) and
               'speed:mission_cruise_setpoint_locked' in c.reasons
               for c in bank if c.speed_mode in ('slow_25', 'slow_50', 'accelerate_25', 'accelerate_50'))
