import pytest

from usv_sim.actions import speed_envelope, speed_setpoint
from usv_sim.dynamics import Simplified3DOFDynamics
from usv_sim.models import ShipConfig, ShipState


@pytest.mark.parametrize("length_m", [50.0, 200.0, 500.0])
@pytest.mark.parametrize("initial_speed", [3.5, 6.0, 7.8])
def test_tactical_speed_targets_stay_inside_half_to_one_and_half_initial_speed(length_m, initial_speed):
    config = ShipConfig.for_length(length_m)
    state = ShipState(u=initial_speed)
    low, high = speed_envelope(config, initial_speed)
    assert low >= 0.5 * initial_speed - 1e-9
    assert high <= 1.5 * initial_speed + 1e-9
    for mode in ("hold", "slow_25", "slow_50", "accelerate_25", "accelerate_50"):
        target = speed_setpoint(state, config, mode, reference_speed=initial_speed,
                                mission_speed_mps=initial_speed)
        assert low - 1e-9 <= target <= high + 1e-9
        assert config.min_speed_mps - 1e-9 <= target <= config.max_speed_mps + 1e-9


def test_nominal_speed_steps_are_quarter_and_half_of_initial_speed_when_physical_limits_allow():
    config = ShipConfig.for_length(200)
    v0 = 5.0
    state = ShipState(u=v0)
    assert speed_setpoint(state, config, "slow_25", reference_speed=v0, mission_speed_mps=v0) == pytest.approx(0.75*v0)
    assert speed_setpoint(state, config, "slow_50", reference_speed=v0, mission_speed_mps=v0) == pytest.approx(0.50*v0)
    assert speed_setpoint(state, config, "accelerate_25", reference_speed=v0, mission_speed_mps=v0) == pytest.approx(1.25*v0)
    assert speed_setpoint(state, config, "accelerate_50", reference_speed=v0, mission_speed_mps=v0) == pytest.approx(1.50*v0)


def test_vessel_length_limits_actual_acceleration_and_response_rate():
    small, medium, large = [ShipConfig.for_length(length) for length in (50, 200, 500)]
    assert small.max_accel_mps2 > medium.max_accel_mps2 > large.max_accel_mps2
    assert small.max_decel_mps2 > medium.max_decel_mps2 > large.max_decel_mps2
    assert small.speed_tau_s < medium.speed_tau_s < large.speed_tau_s

    for config in (small, medium, large):
        dynamics = Simplified3DOFDynamics(config)
        initial = ShipState(u=5.0)
        accelerated = dynamics.step(initial, 0.0, 7.0, 2.0)
        decelerated = dynamics.step(initial, 0.0, 3.0, 2.0)
        assert 0.0 <= accelerated.u-initial.u <= config.max_accel_mps2*2.0+1e-9
        assert 0.0 <= initial.u-decelerated.u <= config.max_decel_mps2*2.0+1e-9


def test_speed_setpoint_is_non_ratcheting_and_anchored_to_encounter_start():
    config = ShipConfig.for_length(200)
    mission_v0 = 6.0
    state_after_slowing = ShipState(u=4.8)
    first = speed_setpoint(state_after_slowing, config, "accelerate_25",
                           reference_speed=mission_v0, mission_speed_mps=mission_v0)
    later = speed_setpoint(ShipState(u=5.3), config, "accelerate_25",
                           reference_speed=mission_v0, mission_speed_mps=mission_v0)
    assert first == pytest.approx(1.25*mission_v0)
    assert later == pytest.approx(first)
