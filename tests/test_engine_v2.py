import math
from usv_sim.engine import SimulationEngine
from usv_sim.models import Scenario, ShipConfig, ShipState, TargetShip, StaticObstacle


def empty_scene(**kw):
    return Scenario('test', ShipConfig.for_length(100), ShipState(), [], **kw)


def test_reset_restores_initial_target_without_mutating_source():
    s=empty_scene()
    s.targets=[TargetShip('T',ShipState(x=8000,u=5))]
    e=SimulationEngine(s)
    e.step()
    assert s.targets[0].state.y == 0
    assert e.targets[0].state.y > 0
    e.reset()
    assert e.targets[0].state.y == 0


def test_prepared_action_is_executed_without_intent_rewrite():
    e=SimulationEngine(empty_scene())
    obs,mask=e.prepare_decision()
    action=e.baseline_action
    expected=e.candidates[action]
    e.step(action_override=action)
    assert e.last_action == action
    assert e.last_executed_action == action
    assert e.last_plan.course_cmd_rad == expected.course_cmd_rad
    assert e.last_plan.speed_cmd_mps == expected.speed_cmd_mps
    assert math.dist((e.own.x,e.own.y),expected.first_endpoint)<1e-8


def test_no_solution_has_nonempty_recovery_mask_and_logged_reason():
    s=empty_scene(obstacles=[StaticObstacle('land',[(-200,-200),(200,-200),(200,200),(-200,200)])])
    e=SimulationEngine(s)
    _,mask=e.prepare_decision()
    assert any(mask)
    assert e.decision_infeasible
    e.step()
    assert not e.last_plan.safe
    assert e.logs[-1]['planning_status']=='minimum_risk'
    assert e.logs[-1]['intervention_reasons']


def test_historical_route_return_does_not_finish_outside_current_corridor():
    e=SimulationEngine(empty_scene())
    e.metrics.route_returned=True
    e.own.x=500
    e.own.y=12000
    e.step()
    assert not e.metrics.destination_reached


def test_repeated_prepare_does_not_recompute_or_mutate_encounter_clock():
    e=SimulationEngine(empty_scene())
    first=e.prepare_decision()
    bank=e.candidates
    second=e.prepare_decision()
    assert bank is e.candidates
    assert first==second


def test_advance_decision_uses_same_two_second_control_clock():
    e=SimulationEngine(empty_scene())
    e.prepare_decision()
    reward,terminal,info=e.advance_decision(e.baseline_action)
    assert e.metrics.elapsed_s==10
    assert len(e.own_track)==6
    assert info['elapsed_s']==10
    assert isinstance(reward,float)


def test_empty_target_scene_supported_and_waypoints_advance():
    e=SimulationEngine(empty_scene(route=[(0,100),(100,100)]))
    e.own.y=100
    e.step()
    assert not e.metrics.destination_reached
    assert e.route_x==100


def test_contact_counter_counts_entry_events_not_contact_steps():
    s=empty_scene(obstacles=[StaticObstacle('land',[(-500,-500),(500,-500),(500,500),(-500,500)])])
    e=SimulationEngine(s)
    e.step()
    e.step()
    assert e.metrics.obstacle_collision_count==1


def test_stand_on_holds_actual_course_when_off_route():
    from usv_sim.models import EncounterAssessment,RiskSnapshot
    from usv_sim.planner import DWAMPCPlanner
    own=ShipState(x=1200,psi=math.radians(30))
    a=EncounterAssessment('T','port_crossing',False,'R17',-40,8000,800,800,1000,'关注',.4)
    bank=DWAMPCPlanner(ShipConfig()).generate_candidates(own,[],RiskSnapshot([a],'T',.4,False))
    assert bank[0].feasible
    assert abs(bank[0].course_cmd_rad-own.psi)<1e-8


def test_rule_constraints_compare_absolute_course_to_encounter_reference():
    from usv_sim.rules import candidate_rule_reasons
    from usv_sim.models import EncounterAssessment
    a=EncounterAssessment('T','port_crossing',False,'R17',-40,8000,800,800,1000,'关注',.4)
    reasons=candidate_rule_reasons(0,'hold',[a],course_cmd_rad=.5,speed_cmd_mps=7,
                                   references={'T':(0.,7.)})
    assert any('R17' in r for r in reasons)
