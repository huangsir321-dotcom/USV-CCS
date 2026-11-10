import math

from usv_sim.colregs import classify_encounter
from usv_sim.models import EncounterAssessment, ShipState
from usv_sim.rules import candidate_rule_reasons
from usv_sim.risk import _preferred_manoeuvre
from usv_sim.models import TargetShip


def _assessment(encounter, give_way=True, score=0.7, risk_level="碰撞危险"):
    return EncounterAssessment("T", encounter, give_way, "COLREG", 45.0,
        2000.0, 100.0, 300.0, 500.0, risk_level, score,
        preferred_turn_deg=20.0 if encounter == "head_on" else 30.0)


def test_rule14_head_on_requires_clear_starboard_alteration():
    a = _assessment("head_on")
    assert candidate_rule_reasons(0.0, "hold", [a])
    assert candidate_rule_reasons(-20.0, "hold", [a])
    assert candidate_rule_reasons(5.0, "hold", [a])
    assert not candidate_rule_reasons(10.0, "hold", [a])
    assert not candidate_rule_reasons(20.0, "hold", [a])


def test_appendix_c_right_and_left_beam_contacts_are_separated():
    own = ShipState(x=0.0, y=0.0, psi=0.0, u=6.0)
    starboard = TargetShip("R", ShipState(x=1000.0, y=0.0, psi=math.radians(270.0), u=5.0))
    port = TargetShip("L", ShipState(x=-1000.0, y=0.0, psi=math.radians(90.0), u=5.0))
    assert classify_encounter(own, starboard.state)[0] == "starboard_beam"
    assert classify_encounter(own, port.state)[0] == "port_beam"


def test_right_beam_give_way_requires_substantial_action_not_mandatory_starboard_turn():
    a = _assessment("starboard_beam", True)
    assert any("R16_substantial_action" in reason
               for reason in candidate_rule_reasons(0.0, "hold", [a]))
    assert any("R16_substantial_action" in reason
               for reason in candidate_rule_reasons(5.0, "hold", [a]))
    assert not any("R16_substantial_action" in reason
                   for reason in candidate_rule_reasons(15.0, "hold", [a]))
    # Rule 15 does not mandate a starboard turn: a meaningful deceleration
    # remains a valid candidate, provided its predicted path passes astern.
    assert not any("R16_substantial_action" in reason
                   for reason in candidate_rule_reasons(0.0, "slow_25", [a]))
    assert not any("R16_substantial_action" in reason
                   for reason in candidate_rule_reasons(-15.0, "hold", [a]))


def test_rule17_stand_on_keeps_course_and_speed_before_intervention():
    a = _assessment("port_crossing", False, score=0.40)
    assert candidate_rule_reasons(15.0, "accelerate_25", [a], speed_cmd_mps=7.5)
    assert not candidate_rule_reasons(0.0, "hold", [a], speed_cmd_mps=7.0)


def test_appendix_a_reference_angles_are_advisory_manoeuvre_preferences():
    own = ShipState(psi=0.0, u=7.0)
    head_on_target = TargetShip("H", ShipState(x=0.0, y=2000.0, psi=math.pi, u=7.0))
    crossing_target = TargetShip("C", ShipState(x=1400.0, y=1400.0, psi=math.radians(45), u=5.0))
    beam_target = TargetShip("B", ShipState(x=2000.0, y=0.0, psi=math.radians(270), u=5.0))
    assert _preferred_manoeuvre(own, head_on_target, "head_on") == ("starboard", 20.0)
    assert _preferred_manoeuvre(own, crossing_target, "starboard_crossing") == ("starboard", 30.0)
    assert _preferred_manoeuvre(own, beam_target, "starboard_beam") == ("starboard", 60.0)


def test_minimum_risk_fallback_preserves_rule14_starboard_when_available():
    from usv_sim.engine import SimulationEngine
    from usv_sim.scenarios import make_scenario

    engine = SimulationEngine(make_scenario(target_count=2))
    engine.prepare_decision()
    assert not any(candidate.feasible for candidate in engine.candidates)
    chosen = engine.planner.select_candidate(engine.candidates)
    # The synthetic mixed scene has no fully SDA-feasible path at t=0. The
    # fallback must not silently choose a port turn while a core-rule-aligned
    # starboard candidate remains available.
    assert chosen.offset_deg >= 10.0
    assert not any("R14_starboard_required" in reason for reason in chosen.rule_reasons)
    assert chosen.status == "minimum_risk"
    assert "preserving Rule 14 direction" in chosen.message
