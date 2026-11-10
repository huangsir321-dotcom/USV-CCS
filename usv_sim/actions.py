"""Versioned, fine-grained manoeuvres shared by planning, learning and display.

Tactical speed changes are anchored to the mission's initial cruise speed. The
hard command envelope is +/-50%; MPC can select 25% or 50% levels without
ratcheting the target on every re-plan. Actual surge acceleration remains
limited by the length-dependent vessel dynamics in ``models.py``/``dynamics.py``.
"""
from dataclasses import dataclass

CONTRACT_VERSION = 'candidate-v4'
CONTROL_DT_S = 2.0
DECISION_PERIOD_S = 10.0
MAX_TACTICAL_SPEED_DELTA_RATIO = 0.50


@dataclass(frozen=True)
class ActionSpec:
    offset_deg: float
    speed_mode: str


# 5-degree resolution across a practical local manoeuvre envelope.
OFFSET_ANGLES_DEG = (0,) + tuple(
    value for step in range(5, 65, 5) for value in (step, -step)
)
# The largest tactical speed increase/decrease is 50% of the mission's initial
# speed. 25% levels give MPC a less abrupt intermediate option.
SPEED_MODES = ('hold', 'slow_25', 'slow_50', 'accelerate_25', 'accelerate_50')
SPEED_MODE_RATIO = {
    'hold': 0.0,
    'slow_25': -0.25,
    'slow_50': -0.50,
    'accelerate_25': 0.25,
    'accelerate_50': 0.50,
}
SPEED_MODE_LABEL = {
    'hold': '保持速度',
    'slow_25': '减速25%',
    'slow_50': '减速50%',
    'accelerate_25': '加速25%',
    'accelerate_50': '加速50%',
}


def is_slow_mode(mode: str) -> bool:
    return str(mode).startswith('slow_')


def is_accelerate_mode(mode: str) -> bool:
    return str(mode).startswith('accelerate_')


def is_speed_change_mode(mode: str) -> bool:
    return mode != 'hold'


ACTION_SPECS = tuple(ActionSpec(angle, mode)
                     for angle in OFFSET_ANGLES_DEG
                     for mode in SPEED_MODES)
ACTION_COUNT = len(ACTION_SPECS)
ACTION_NAMES = [f"{'保持航向' if a.offset_deg == 0 else ('右转' if a.offset_deg > 0 else '左转') + str(abs(a.offset_deg)) + '°'} / "
                + SPEED_MODE_LABEL[a.speed_mode]
                for a in ACTION_SPECS]
ACTION_INTENTS = [f'{a.offset_deg:+g}:{a.speed_mode}' for a in ACTION_SPECS]


def speed_envelope(config, mission_speed_mps):
    """Return configured tactical setpoint bounds [0.5V0, 1.5V0].

    Vessel-specific min/max speed limits may narrow the envelope. This is a
    command bound; it is not an acceleration-rate bound in m/s^2.
    """
    v0 = max(0.0, float(mission_speed_mps))
    lower = max(float(config.min_speed_mps), (1.0-MAX_TACTICAL_SPEED_DELTA_RATIO)*v0)
    upper = min(float(config.max_speed_mps), (1.0+MAX_TACTICAL_SPEED_DELTA_RATIO)*v0)
    if lower > upper:
        # Preserve a valid one-point interval for unusually constrained configs.
        lower = upper = max(float(config.min_speed_mps), min(float(config.max_speed_mps), v0))
    return lower, upper


def speed_setpoint(state, config, mode, reference_speed=None, mission_speed_mps=None):
    """Return a non-ratcheting speed target within the +/-50% mission envelope.

    ``reference_speed`` is the fixed speed at the start of the current encounter.
    ``mission_speed_mps`` is the scenario's initial ownship speed (V0), which
    defines the absolute tactical envelope and 25/50% increments.
    """
    anchor = float(state.u if reference_speed is None else reference_speed)
    v0 = float(anchor if mission_speed_mps is None else mission_speed_mps)
    if mode not in SPEED_MODE_RATIO:
        raise ValueError(f'unknown speed mode: {mode}')
    requested = anchor + SPEED_MODE_RATIO[mode]*max(0.0, v0)
    lower, upper = speed_envelope(config, v0)
    return max(lower, min(upper, requested))
