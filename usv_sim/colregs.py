"""Rule-aware encounter classification for open-water power-driven vessels.

The classifier is deliberately conservative and remains an approximation: a
bearing/heading snapshot is not a complete legal lookout assessment. The
Appendix-C beam sectors are exposed separately so tests, UI and rule masks can
reason about right/left beam contacts without collapsing them into a generic
crossing label.
"""
from __future__ import annotations
import math
from .models import EncounterAssessment, wrap_angle
from .geometry import range_bearing, cpa, relative_bearing_deg


HEAD_ON_BEARING_LIMIT_DEG = 10.0
HEAD_ON_HEADING_TOLERANCE_DEG = 15.0
BEAM_SECTOR_HALF_WIDTH_DEG = 15.0


def is_starboard_give_way_encounter(encounter: str) -> bool:
    return encounter in ('starboard_crossing', 'starboard_beam')


def is_port_stand_on_encounter(encounter: str) -> bool:
    return encounter in ('port_crossing', 'port_beam', 'being_overtaken')


def classify_encounter(own, target):
    """Return (encounter, own_give_way, rule, bearing, note).

    Signs follow the simulator's nautical convention: positive relative
    bearing is starboard; positive course alteration is starboard.
    """
    rel_bearing = relative_bearing_deg(own, target)
    rel_abs = abs(rel_bearing)
    heading_diff = abs(math.degrees(wrap_angle(target.psi - own.psi)))
    dcpa, tcpa, rng = cpa(own, target)

    # COLREG Rule 13: ownship approaches from abaft the target's beam by more
    # than 22.5 degrees. If in doubt, the caller should retain overtaking status
    # through encounter memory until finally past and clear.
    _, own_from_target = range_bearing(target, own)
    own_bearing_rel_target = math.degrees(wrap_angle(own_from_target - target.psi))
    own_is_overtaking = abs(own_bearing_rel_target) > 112.5 and tcpa > 0 and rng > 0
    target_is_overtaking = 112.5 < rel_abs <= 180.0 and heading_diff < 70.0 and tcpa > 0

    # Rule 14: reciprocal or nearly reciprocal courses, target nearly ahead.
    # Rule 14(3) says to assume this situation if in doubt.
    if (rel_abs <= HEAD_ON_BEARING_LIMIT_DEG and
            abs(180.0-heading_diff) <= HEAD_ON_HEADING_TOLERANCE_DEG):
        return "head_on", True, "COLREG Rule 14", rel_bearing, "对遇：本船应向右转，与他船左舷通过"
    if own_is_overtaking:
        return "own_overtaking", True, "COLREG Rule 13", rel_bearing, "本船作为追越船，应持续让清被追越船直至驶过让清"
    if target_is_overtaking:
        return "being_overtaken", False, "COLREG Rule 13/17", rel_bearing, "本船为被追越船，通常保持航向航速并监测"

    if heading_diff < 15.0:
        if rel_abs < 90.0 and own.u <= target.u:
            return "following", False, "monitor", rel_bearing, "同向尾随且未追近，持续监测"
        return "none", False, "monitor", rel_bearing, "同向平行或远离，持续监测"

    # GB/T 46851-2025 Appendix C separates crossing and beam-on sectors.
    # +/-15 degrees around the beam is a simulator bin, not a COLREG angle.
    if rel_bearing > 0.0:
        if 90.0-BEAM_SECTOR_HALF_WIDTH_DEG <= rel_abs <= 90.0+BEAM_SECTOR_HALF_WIDTH_DEG:
            return "starboard_beam", True, "COLREG Rule 15/16", rel_bearing, "右正横目标：本船承担让路责任，应避免横越其船首"
        return "starboard_crossing", True, "COLREG Rule 15/16", rel_bearing, "目标在本船右舷：本船让路，环境许可时避免从其船首通过"
    if 90.0-BEAM_SECTOR_HALF_WIDTH_DEG <= rel_abs <= 90.0+BEAM_SECTOR_HALF_WIDTH_DEG:
        return "port_beam", False, "COLREG Rule 15/17", rel_bearing, "左正横目标：本船通常为直航船，先保持航向航速"
    return "port_crossing", False, "COLREG Rule 15/17", rel_bearing, "目标在本船左舷：本船通常为直航船，先保持航向航速"


def assess_rule(own, target, risk_level: str):
    encounter, give_way, rule, bearing, note = classify_encounter(own, target)
    # Positive turn is starboard. Rule 14's direction is mandatory when the
    # head-on risk condition is met; crossing-course preference is soft.
    preferred_turn = 0
    if encounter == "head_on":
        preferred_turn = 20
    elif is_starboard_give_way_encounter(encounter):
        heading_difference = abs(math.degrees(wrap_angle(target.psi-own.psi)))
        preferred_turn = 60 if encounter == 'starboard_beam' or heading_difference >= 60.0 else 30
    elif encounter == "own_overtaking":
        preferred_turn = 0  # Rule 13 does not prescribe a passing side.
    elif encounter in ("port_crossing", "port_beam", "being_overtaken"):
        preferred_turn = 0

    if risk_level == "紧迫局面" and not give_way:
        note += "；已进入紧迫阶段，直航船必要时应采取最有助于避碰的行动"
        preferred_turn = 1 if bearing >= 0 else -1
    return encounter, give_way, rule, preferred_turn, bearing, note
