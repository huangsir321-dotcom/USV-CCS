import math

from usv_sim.sda import sda_m, NM_TO_M
from usv_sim.models import EncounterAssessment, RiskSnapshot
from usv_sim.scenarios import make_scenario
from usv_sim.engine import SimulationEngine


def test_appendix_b_table_b1_head_on_exact_value():
    value, estimated = sda_m(100, 50, "head_on")
    assert math.isclose(value, 0.34 * NM_TO_M, rel_tol=0, abs_tol=1e-9)
    assert not estimated


def test_appendix_b_table_b2_starboard_crossing_stern_exact_value():
    value, estimated = sda_m(200, 300, "starboard_crossing", "stern")
    assert math.isclose(value, 0.95 * NM_TO_M, rel_tol=0, abs_tol=1e-9)
    assert not estimated


def test_appendix_b_has_distinct_beam_rows():
    right_beam, _ = sda_m(200, 200, "starboard_beam", "bow")
    right_cross, _ = sda_m(200, 200, "starboard_crossing", "bow")
    left_beam, _ = sda_m(200, 300, "port_beam", "stern")
    left_cross, _ = sda_m(200, 300, "port_crossing", "stern")
    assert math.isclose(right_beam, 0.84 * NM_TO_M, abs_tol=1e-9)
    assert not math.isclose(right_beam, right_cross, abs_tol=1e-9)
    assert math.isclose(left_beam, 0.88 * NM_TO_M, abs_tol=1e-9)
    assert not math.isclose(left_beam, left_cross, abs_tol=1e-9)


def test_appendix_b_type_factor_is_applied():
    bulk, _ = sda_m(200, 100, "head_on", ship_type="bulk")
    container, _ = sda_m(200, 100, "head_on", ship_type="container")
    assert math.isclose(container, bulk * 1.33, rel_tol=1e-12)


def test_interpolated_ship_lengths_are_marked_as_estimated():
    value, estimated = sda_m(150, 150, "head_on")
    assert value > 0
    assert estimated


def test_unclassified_encounter_uses_conservative_estimated_boundary():
    value, estimated = sda_m(200, 100, "none")
    assert value > 0
    assert estimated


def test_crossing_sda_boundary_uses_conservative_bow_stern_envelope():
    bow, _ = sda_m(200, 300, "starboard_crossing", "bow")
    stern, _ = sda_m(200, 300, "starboard_crossing", "stern")
    envelope, estimated = sda_m(200, 300, "starboard_crossing", "conservative_max")
    assert math.isclose(envelope, max(bow, stern), abs_tol=1e-9)
    assert not estimated


def test_conservative_max_alias_selects_stern_when_stern_row_is_larger():
    bow, _ = sda_m(200, 50, "starboard_beam", "bow")
    stern, _ = sda_m(200, 50, "starboard_beam", "stern")
    envelope, estimated = sda_m(200, 50, "starboard_beam", "conservative_max")
    assert math.isclose(bow, 0.63 * NM_TO_M, abs_tol=1e-9)
    assert math.isclose(stern, 0.64 * NM_TO_M, abs_tol=1e-9)
    assert math.isclose(envelope, stern, abs_tol=1e-9)
    assert not estimated


def test_unknown_sda_uses_conservative_estimated_envelope():
    value, estimated = sda_m(200, 100, "following")
    assert value > 0
    assert estimated


def _assessment(sda=1000.0):
    return EncounterAssessment(
        target_id="TV1", encounter="head_on", own_give_way=True,
        rule="COLREG Rule 14", relative_bearing_deg=0.0,
        range_m=1000.0, dcpa_m=0.0, tcpa_s=10.0,
        sda_m=sda, risk_level="碰撞危险", risk_score=0.5,
    )


def test_actual_swept_distance_below_sda_marks_failure_even_without_hull_collision():
    engine = SimulationEngine(make_scenario(1, 200))
    target = engine.targets[0]
    target.state.x, target.state.y = 0.0, 999.0
    engine.last_snapshot = RiskSnapshot([_assessment(1000.0)], "TV1", 0.5, False)

    # Same own-ship position; target moves through the SDA boundary from
    # 1001 m to 999 m. The closest actual centre-to-centre distance is 999 m.
    engine._record_contacts(engine.own.copy(), {"TV1": (0.0, 1001.0)})

    assert engine.metrics.sda_violation_count == 1
    assert engine.metrics.sda_violation_targets == ["TV1"]
    assert engine.metrics.domain_intrusion_s > 0
    assert engine.metrics.minimum_domain_clearance_m < 0
    assert engine.metrics.collision_count == 0


def test_sda_failure_is_latched_once_per_target_and_not_cleared_after_exit():
    engine = SimulationEngine(make_scenario(1, 200))
    target = engine.targets[0]
    engine.last_snapshot = RiskSnapshot([_assessment(1000.0)], "TV1", 0.5, False)

    target.state.x, target.state.y = 999.0, 0.0
    engine._record_contacts(engine.own.copy(), {"TV1": (999.0, 0.0)})
    target.state.x, target.state.y = 1200.0, 0.0
    engine._record_contacts(engine.own.copy(), {"TV1": (1200.0, 0.0)})

    assert engine.metrics.sda_violation_count == 1
    assert engine.metrics.sda_violation_targets == ["TV1"]


def test_actual_distance_equal_to_or_above_sda_does_not_fail():
    engine = SimulationEngine(make_scenario(1, 200))
    target = engine.targets[0]
    target.state.x, target.state.y = 1000.5, 0.0
    engine.last_snapshot = RiskSnapshot([_assessment(1000.0)], "TV1", 0.5, False)
    engine._record_contacts(engine.own.copy(), {"TV1": (1000.5, 0.0)})
    assert engine.metrics.sda_violation_count == 0
    assert engine.metrics.domain_intrusion_s == 0
    assert engine.metrics.minimum_domain_clearance_m >= 0
