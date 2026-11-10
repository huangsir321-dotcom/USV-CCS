"""Executable COLREG rule constraints for open-water power-driven encounters.

Hard constraints implement explicit directional/trajectory duties where the
current state model supports them. Appendix-A initial alteration angles and
stand-on intervention thresholds are engineering preferences, not legal
thresholds. Unmodelled visibility/traffic/status cases are documented as gaps.
"""
from dataclasses import dataclass
import math
from .models import wrap_angle
from .colregs import is_port_stand_on_encounter


@dataclass(frozen=True)
class RuleConfig:
    activation_score: float = 0.12
    stand_on_intervention_score: float = 0.65
    obvious_turn_deg: float = 10.0
    head_on_open_water_reference_deg: float = 20.0
    crossing_reference_weight: float = 0.0020
    head_on_reference_weight: float = 0.0040


DEFAULT_RULES = RuleConfig()


def candidate_rule_reasons(offset_deg, speed_mode, assessments, config=DEFAULT_RULES,
                           course_cmd_rad=None, speed_cmd_mps=None, references=None):
    reasons = []
    for a in assessments:
        if a.risk_score < config.activation_score:
            continue
        prefix = a.target_id + ':'
        reference = (references or {}).get(a.target_id)
        effective_offset = (
            math.degrees(wrap_angle(course_cmd_rad-reference[0]))
            if reference is not None and course_cmd_rad is not None else offset_deg
        )
        if a.own_give_way and a.encounter == 'head_on':
            # Rule 14: ownship must make a starboard alteration. Ten degrees is
            # the configured minimum noticeable alteration; 20 degrees is the
            # open-water initial-angle reference in GB/T 46851-2025 Table A.1.
            if effective_offset < config.obvious_turn_deg-1e-6:
                reasons.append(prefix + 'R14_starboard_required')
            continue

        if (not a.own_give_way and a.encounter in ('port_crossing', 'port_beam')
                and a.risk_score >= config.stand_on_intervention_score
                and effective_offset < -1e-6):
            # Rule 17(c): when a stand-on power-driven vessel acts under 17(a)(ii),
            # avoid altering to port for a vessel on ownship's port side if the
            # circumstances admit. Kept as a mask preference; safety fallback logs override.
            reasons.append(prefix + 'R17c_avoid_port_for_port_contact')

        if a.own_give_way and a.encounter in ('starboard_crossing', 'starboard_beam'):
            # Rules 15/16 do NOT prescribe a mandatory starboard turn for a
            # crossing give-way vessel. Require a noticeable course alteration
            # OR an intentional speed reduction; the predicted path check below
            # separately enforces avoiding crossing ahead when circumstances admit.
            substantial_course = abs(effective_offset) >= config.obvious_turn_deg-1e-6
            substantial_speed = str(speed_mode).startswith('slow_')
            if not (substantial_course or substantial_speed):
                reasons.append(prefix + 'R16_substantial_action')

        if a.risk_level == '紧迫局面':
            # Rule 17(b): once close quarters are reached and give-way action alone
            # cannot avoid collision, stand-on may/must take the best action.
            continue

        if (not a.own_give_way and is_port_stand_on_encounter(a.encounter)
                and a.risk_score < config.stand_on_intervention_score):
            speed_error = (abs(speed_cmd_mps-reference[1]) if reference is not None and
                           speed_cmd_mps is not None else float(speed_mode != 'hold'))
            if abs(effective_offset) > 1.0 or speed_error > 0.1:
                reasons.append(prefix + 'R17_hold_course_speed')
    return tuple(reasons)


def phase_for(snapshot, active=False):
    if snapshot.emergency:
        return 'emergency'
    if any(not a.own_give_way and a.risk_score >= DEFAULT_RULES.stand_on_intervention_score
           for a in snapshot.assessments):
        return 'stand_on_intervene'
    if any(a.own_give_way and a.risk_score >= DEFAULT_RULES.activation_score
           for a in snapshot.assessments):
        return 'give_way'
    if any(not a.own_give_way and is_port_stand_on_encounter(a.encounter)
           and a.risk_score >= DEFAULT_RULES.activation_score for a in snapshot.assessments):
        return 'stand_on'
    return 'passing' if active else 'monitor'
