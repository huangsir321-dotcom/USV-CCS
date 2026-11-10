import copy
import math
import pytest
from usv_sim.models import ShipState, ShipConfig, TargetShip, EnvironmentConfig
from usv_sim.dynamics import Simplified3DOFDynamics
from usv_sim.colregs import classify_encounter
from usv_sim.risk import MultiShipRiskAssessor
from usv_sim.sda import sda_m
from usv_sim.geometry import swept_separation
from usv_sim.geometry import cpa


def test_swept_separation_detects_between_endpoint_collision():
    assert swept_separation((-10, 0), (10, 0), (0, -10), (0, 10)) == pytest.approx(0)


def test_near_field_risk_survives_nonpositive_tcpa():
    own = ShipState(u=7)
    for target in [ShipState(y=200, u=7), ShipState(y=200, u=9), ShipState(u=7)]:
        result = MultiShipRiskAssessor().assess(own, ShipConfig(), [TargetShip('t', target)])
        assert result.max_risk_score > 0
        assert result.emergency
    far = MultiShipRiskAssessor().assess(own, ShipConfig(), [TargetShip('t', ShipState(y=10000, u=9))])
    assert far.max_risk_score == 0


def test_target_oracle_and_update_share_substeps_and_thresholds():
    target = TargetShip('t', ShipState(u=7), behavior='turn_after_delay', turn_start_s=2.5,
                        turn_rate_deg_s=5, speed_change_start_s=2.5, speed_after_mps=9)
    env = EnvironmentConfig(current_x_mps=0.4)
    predicted = target.predict_position(8, env, oracle=True)
    updated = copy.deepcopy(target)
    updated.update(8, env)
    assert predicted == pytest.approx((updated.state.x, updated.state.y))
    assert updated.state.u == pytest.approx(7 + 0.04 * 5.5)
    assert updated.state.psi == pytest.approx(math.radians(5 * 5.5))
    repeated = copy.deepcopy(target)
    for _ in range(8):
        repeated.update(1, env)
    assert vars(updated.state) == pytest.approx(vars(repeated.state))


def test_observed_prediction_does_not_read_future_scripts():
    target = TargetShip('t', ShipState(u=7), behavior='turn_after_delay', turn_start_s=0,
                        turn_rate_deg_s=20, speed_change_start_s=0, speed_after_mps=9)
    env = EnvironmentConfig(current_x_mps=1)
    assert target.predict_position(10, env) == pytest.approx((10, 70))
    for actual, expected in zip(target.prediction_path([0, 2, 10], env), [(0, 0), (2, 14), (10, 70)]):
        assert actual == pytest.approx(expected)


def test_dynamics_large_step_matches_one_second_steps():
    dyn = Simplified3DOFDynamics(ShipConfig())
    env = EnvironmentConfig(wave_level=2, wind_speed_mps=4)
    single = dyn.step(ShipState(), 0.3, 5, 8, env, 10)
    repeated = ShipState()
    for i in range(8):
        repeated = dyn.step(repeated, 0.3, 5, 1, env, 10+i)
    assert vars(single) == pytest.approx(vars(repeated))


def test_parallel_following_and_directly_astern_overtaking():
    own = ShipState(u=7)
    assert classify_encounter(own, ShipState(x=100, u=7))[0] == 'none'
    assert classify_encounter(own, ShipState(y=500, u=9))[0] == 'following'
    assert classify_encounter(own, ShipState(y=500, u=5))[0] == 'own_overtaking'
    assert classify_encounter(own, ShipState(y=-500, u=9))[0] == 'being_overtaken'


def test_encounter_release_uses_elapsed_time_and_current_bearing():
    own, cfg = ShipState(u=7), ShipConfig()
    target = TargetShip('t', ShipState(y=1000, psi=math.pi, u=7))
    assessor = MultiShipRiskAssessor()
    assessor.assess(own, cfg, [target], elapsed_s=0)
    target.state = ShipState(x=100, y=-5000, u=9, psi=math.pi)
    first = assessor.assess(own, cfg, [target], elapsed_s=10).assessments[0]
    assert first.encounter == 'head_on'
    assert abs(first.relative_bearing_deg) > 90
    for _ in range(10):
        assessor.assess(own, cfg, [target], elapsed_s=10)
    assert 't' in assessor._encounter_memory
    target.state.y -= 100
    assessor.assess(own, cfg, [target], elapsed_s=15)
    assert 't' in assessor._encounter_memory
    target.state.y -= 100
    assessor.assess(own, cfg, [target], elapsed_s=16)
    assert 't' not in assessor._encounter_memory


def test_unknown_sda_mapping_is_explicitly_estimated():
    for encounter in ['none', 'following', 'unknown']:
        distance, estimated = sda_m(200, 100, encounter)
        assert distance > 0 and estimated


def test_release_does_not_depend_on_distance_per_assessment():
    own, cfg = ShipState(u=7), ShipConfig()
    target = TargetShip('t', ShipState(y=1000, psi=math.pi, u=7))
    assessor = MultiShipRiskAssessor()
    assessor.assess(own, cfg, [target], elapsed_s=0)
    target.state = ShipState(y=-5000, psi=math.pi, u=7)
    for i in range(62):
        target.state.y -= 0.1
        assessor.assess(own, cfg, [target], elapsed_s=10 + i * 0.1)
    assert 't' not in assessor._encounter_memory


def test_oracle_samples_do_not_change_integrated_trajectory():
    target = TargetShip('t', ShipState(u=7), behavior='turn_after_delay',
                        turn_start_s=0, turn_rate_deg_s=30)
    path = target.prediction_path([0.5, 1, 1.5, 2], oracle=True)
    for t, position in zip([0.5, 1, 1.5, 2], path):
        assert position == pytest.approx(target.predict_position(t, oracle=True))
    assert target.prediction_path([2, 0.5, 1], oracle=True) == [path[3], path[0], path[1]]


def test_cpa_includes_sway_velocity():
    dcpa, tcpa, _ = cpa(ShipState(u=0), ShipState(x=100, u=0, v=-10))
    assert dcpa == pytest.approx(0)
    assert tcpa == pytest.approx(10)


def test_following_memory_can_upgrade_to_overtaking():
    own, cfg = ShipState(u=7), ShipConfig()
    target = TargetShip('t', ShipState(y=1000, u=9))
    assessor = MultiShipRiskAssessor()
    assert assessor.assess(own, cfg, [target], elapsed_s=0).assessments[0].encounter == 'following'
    target.state.u = 5
    assessment = assessor.assess(own, cfg, [target], elapsed_s=1).assessments[0]
    assert assessment.encounter == 'own_overtaking'
    assert assessment.own_give_way


@pytest.mark.parametrize('initial,command', [(10.3, 100), (0.5, -100)])
def test_out_of_range_initial_speed_converges_without_instant_jump(initial, command):
    cfg = ShipConfig(max_speed_mps=9, min_speed_mps=2)
    dyn = Simplified3DOFDynamics(cfg)
    result = dyn.step(ShipState(u=initial), 0, command, 1)
    assert -cfg.max_decel_mps2 <= result.u - initial <= cfg.max_accel_mps2
    assert min(initial, max(cfg.min_speed_mps, min(cfg.max_speed_mps, command))) <= result.u
    assert result.u <= max(initial, max(cfg.min_speed_mps, min(cfg.max_speed_mps, command)))


@pytest.mark.parametrize('initial,command', [(2.0, -100), (9.0, 100), (5.0, -100), (5.0, 100)])
def test_in_range_speed_remains_in_configured_bounds(initial, command):
    cfg = ShipConfig(max_speed_mps=9, min_speed_mps=2)
    result = Simplified3DOFDynamics(cfg).step(ShipState(u=initial), 0, command, 20)
    assert cfg.min_speed_mps <= result.u <= cfg.max_speed_mps
