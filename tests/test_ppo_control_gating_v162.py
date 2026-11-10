import math

from usv_sim.actions import ACTION_COUNT, ACTION_SPECS
from usv_sim.engine import SimulationEngine
from usv_sim.models import (Scenario, ShipConfig, ShipState, EncounterAssessment, RiskSnapshot)


class AggressiveFakePPO:
    """A deliberately bad policy that always asks for a 60-degree turn."""
    available = True
    checkpoint = 'fake-policy.pt'

    def __init__(self):
        self.calls = 0
        self.action = next(i for i, spec in enumerate(ACTION_SPECS)
                           if spec.offset_deg == 60 and spec.speed_mode == 'hold')

    def predict(self, observation, risk_snapshot=None, action_mask=None):
        self.calls += 1
        if action_mask is not None and not action_mask[self.action]:
            return next(i for i, allowed in enumerate(action_mask) if allowed)
        return self.action


def empty_scene():
    return Scenario('ppo-gate-test', ShipConfig.for_length(200), ShipState(), [],
                    route=[(0.0, 12000.0)])


def test_ppo_cannot_start_turn_during_monitor_phase():
    engine = SimulationEngine(empty_scene())
    fake = AggressiveFakePPO()
    engine.policy = fake

    _, mask = engine.prepare_decision()
    assert not engine.maneuver_active
    assert sum(mask) == 1
    assert mask[engine.baseline_action]

    engine.step()
    assert fake.calls == 0
    assert engine.last_executed_action == engine.baseline_action
    assert engine.last_policy_source == 'rule_mpc_monitor'


def test_ppo_cannot_override_dedicated_route_recovery_controller():
    engine = SimulationEngine(empty_scene())
    # Put the own ship well outside the route corridor and activate recovery.
    engine.own.x = 350.0
    engine.own.y = 1500.0
    engine.route_recovery_active = True
    engine.maneuver_active = False
    engine._prepared_key = None
    fake = AggressiveFakePPO()
    engine.policy = fake

    _, mask = engine.prepare_decision()
    assert sum(mask) == 1
    assert mask[engine.baseline_action]
    assert not engine.ppo_action_authorized
    assert engine.policy_gate_reason == 'route_recovery_uses_rule_mpc'

    engine.step()
    assert fake.calls == 0
    assert engine.last_executed_action == engine.baseline_action
    assert engine.last_policy_source == 'rule_mpc_route_recovery'
    assert engine.route_recovery_active


def test_ppo_can_choose_during_active_collision_manoeuvre():
    engine = SimulationEngine(empty_scene())
    engine.maneuver_active = True
    # Empty target set keeps this isolated policy-routing fixture active.
    engine.maneuver_target_ids = set()
    fake = AggressiveFakePPO()
    engine.policy = fake
    engine._prepared_key = None

    _, mask = engine.prepare_decision()
    assert engine.ppo_action_authorized
    assert sum(mask) > 1

    engine.step()
    assert fake.calls == 1
    assert engine.last_policy_source == 'ppo_active_encounter'


def test_policy_gate_does_not_change_action_or_observation_contract():
    from usv_sim.ppo_agent import ACTION_COUNT as PPO_ACTION_COUNT, OBS_SIZE
    engine = SimulationEngine(empty_scene())
    observation, mask = engine.prepare_decision()
    assert PPO_ACTION_COUNT == ACTION_COUNT
    assert len(observation) == OBS_SIZE
    assert len(mask) == ACTION_COUNT



def test_1t_003_ppo_and_rule_policy_match_before_encounter_window():
    from usv_sim.dataset import default_dataset_root, list_cases, load_case
    import copy

    spec = next(s for s in list_cases(default_dataset_root(), '1T') if s.scene_id == '003')
    scene = load_case(spec, own_length_m=200, default_target_length_m=100)
    rule_engine = SimulationEngine(copy.deepcopy(scene))
    ppo_engine = SimulationEngine(copy.deepcopy(scene))
    fake = AggressiveFakePPO()
    ppo_engine.policy = fake

    # This supplied overtaking case begins well outside the active manoeuvre
    # window; the trained policy must not initiate an early turn or acceleration.
    assert not rule_engine.maneuver_active
    assert not ppo_engine.maneuver_active
    assert ppo_engine.last_snapshot.assessments[0].risk_score < 0.12

    for _ in range(5):
        rule_engine.step()
        ppo_engine.step()

    assert fake.calls == 0
    assert not ppo_engine.maneuver_active
    assert ppo_engine.last_policy_source == 'rule_mpc_monitor'
    assert ppo_engine.last_executed_action == ppo_engine.baseline_action
    assert math.isclose(ppo_engine.own.x, rule_engine.own.x, abs_tol=1e-9)
    assert math.isclose(ppo_engine.own.y, rule_engine.own.y, abs_tol=1e-9)
    assert math.isclose(ppo_engine.own.psi, rule_engine.own.psi, abs_tol=1e-9)


def test_entering_route_recovery_forces_immediate_policy_replan():
    engine = SimulationEngine(empty_scene())
    engine.metrics.elapsed_s = 10.0
    engine._next_policy_update_s = 20.0
    engine._held_course = 0.5
    engine._held_speed = 8.0
    engine.maneuver_active = True
    engine.maneuver_target_ids = {'T1'}
    engine.last_snapshot = RiskSnapshot([EncounterAssessment(
        target_id='T1', encounter='none', own_give_way=False, rule='none',
        relative_bearing_deg=0.0, range_m=10000.0, dcpa_m=5000.0,
        tcpa_s=-100.0, sda_m=100.0, risk_level='无危险', risk_score=0.0)],
        None, 0.0, False)

    engine._update_encounter()

    assert not engine.maneuver_active
    assert engine.route_recovery_active
    assert engine._held_course is None and engine._held_speed is None
    assert engine._next_policy_update_s == engine.metrics.elapsed_s


def test_new_collision_manoeuvre_forces_immediate_policy_replan():
    engine = SimulationEngine(empty_scene())
    engine.metrics.elapsed_s = 10.0
    engine._next_policy_update_s = 20.0
    engine._held_course = 0.5
    engine._held_speed = 8.0
    engine.last_snapshot = RiskSnapshot([EncounterAssessment(
        target_id='T1', encounter='head_on', own_give_way=True, rule='R14',
        relative_bearing_deg=0.0, range_m=5000.0, dcpa_m=50.0,
        tcpa_s=500.0, sda_m=500.0, risk_level='碰撞危险', risk_score=0.5)],
        'T1', 0.5, False)

    engine._update_encounter()

    assert engine.maneuver_active
    assert not engine.route_recovery_active
    assert engine._held_course is None and engine._held_speed is None
    assert engine._next_policy_update_s == engine.metrics.elapsed_s

