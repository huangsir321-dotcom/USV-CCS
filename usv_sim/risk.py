from __future__ import annotations
import math

from .geometry import cpa, velocity_from_state, range_bearing
from .models import EncounterAssessment, RiskSnapshot, wrap_angle
from .colregs import assess_rule
from .sda import sda_m


def _preferred_manoeuvre(own_state, target, encounter):
    """Return (pass-side label, signed initial course alteration in degrees).

    Signs follow the simulator's nautical convention: positive is starboard,
    negative is port. Table A.1 values are used as indicative initial angles,
    not as mandatory or fixed turn commands.
    """
    if encounter == 'own_overtaking':
        _, own_bearing_from_target = range_bearing(target.state, own_state)
        relative = math.degrees(wrap_angle(own_bearing_from_target-target.state.psi))
        # The vessel already being overtaken from the target's port/starboard
        # quarter is the side to preserve as the preferred passing corridor.
        sign = -1.0 if relative < 0.0 else 1.0
        return ('port' if sign < 0 else 'starboard', sign * 10.0)
    if encounter == 'head_on':
        return 'starboard', 20.0
    if encounter in ('starboard_crossing', 'starboard_beam'):
        heading_difference = abs(math.degrees(wrap_angle(target.state.psi-own_state.psi)))
        # Appendix A Table A.1: open-water right-side crossing references are
        # 30 deg (small-angle) / 60 deg (large-angle); beam-on gets the larger
        # initial reference. These are planner preferences, not legal mandates.
        initial_angle = 60.0 if encounter == 'starboard_beam' or heading_difference >= 60.0 else 30.0
        return 'starboard', initial_angle
    return 'unspecified', 0.0


def _critical_distance_threshold(own_state, target_state, own_config,
                                 signed_turn_deg, clearance_m):
    """Appendix-A-inspired critical-range estimate for one candidate turn.

    We solve when the *pre-manoeuvre* relative trajectory enters the clearance
    band around the post-turn relative-motion line. A response-delay allowance
    moves the trigger earlier to account for rudder/yaw lag. This is a practical
    simulator approximation, not a verbatim implementation of every Annex-A
    equation nor a legal/approval determination.

    Returns (range at trigger, time until trigger). An infinite threshold means
    the geometry does not produce a usable approach-side root.
    """
    rx = target_state.x-own_state.x
    ry = target_state.y-own_state.y
    rng = math.hypot(rx, ry)
    if rng <= 1e-9 or clearance_m <= 0.0:
        return rng, 0.0

    own_vx, own_vy = velocity_from_state(own_state)
    target_vx, target_vy = velocity_from_state(target_state)
    v0x, v0y = target_vx-own_vx, target_vy-own_vy
    vv = v0x*v0x+v0y*v0y
    if vv <= 1e-10:
        return math.inf, math.inf
    tcpa = -(rx*v0x+ry*v0y)/vv
    if tcpa <= 0.0:
        return math.inf, math.inf

    new_psi = own_state.psi+math.radians(signed_turn_deg)
    new_own_vx = own_state.u*math.sin(new_psi)+own_state.v*math.cos(new_psi)
    new_own_vy = own_state.u*math.cos(new_psi)-own_state.v*math.sin(new_psi)
    v1x, v1y = target_vx-new_own_vx, target_vy-new_own_vy
    speed1 = math.hypot(v1x, v1y)
    if speed1 <= 1e-8:
        return math.inf, math.inf

    cross_now = rx*v1y-ry*v1x
    cross_rate = v0x*v1y-v0y*v1x
    if abs(cross_rate) <= 1e-10:
        return math.inf, math.inf
    band = max(0.0, float(clearance_m))*speed1
    roots = sorted(((band-cross_now)/cross_rate,
                    (-band-cross_now)/cross_rate))
    approach_roots = [value for value in roots if 0.0 <= value <= tcpa]
    if not approach_roots:
        return math.inf, math.inf
    entry_time = approach_roots[0]

    max_yaw_rate = max(0.05, float(own_config.max_yaw_rate_deg_s))
    turn_time = abs(float(signed_turn_deg))/max_yaw_rate
    response_delay = max(20.0, 1.25*float(own_config.yaw_tau_s)
                         + 1.5*float(own_config.rudder_tau_s) + turn_time)
    trigger_time = max(0.0, entry_time-response_delay)
    trigger_range = math.hypot(rx+v0x*trigger_time, ry+v0y*trigger_time)
    return trigger_range, trigger_time


class MultiShipRiskAssessor:
    """Pairwise CPA/TCPA and a rule-aware, bounded intervention window.

    The DCRC/DCQS-style window is an engineering approximation inspired by
    Appendix A of GB/T 46851-2025. The actual standard's Appendix A is informative;
    the result is not a substitute for full COLREGs interpretation or validation.
    """
    def __init__(self, close_time_s: float = 90.0):
        self.close_time_s = close_time_s
        self._encounter_memory = {}

    def assess(self, own_state, own_config, targets, elapsed_s=None, environment=None):
        now = float(elapsed_s) if elapsed_s is not None else None
        assessments = []
        for target in targets:
            if not target.active:
                continue

            dcpa, tcpa, rng = cpa(own_state, target.state)
            observed = assess_rule(own_state, target.state, "无危险")
            observed_encounter, observed_give_way, observed_rule, _, observed_bearing, observed_note = observed

            memory = self._encounter_memory.get(target.target_id)
            if memory is not None:
                encounter = memory["encounter"]
                give_way = memory["give_way"]
                rule = memory["rule"]
                bearing = observed_bearing
                note = memory["note"]
            else:
                encounter = observed_encounter
                give_way = observed_give_way
                rule = observed_rule
                bearing = observed_bearing
                note = observed_note

            # For crossing/beam situations the map ring and measured-distance
            # failure threshold use a conservative Appendix-B envelope: max of
            # the table's bow/stern SDA values until a route-specific threshold
            # is explicitly configured. This avoids under-sizing the risk circle.
            sda_pass_position = (
                "conservative_max"
                if encounter in ("starboard_crossing", "port_crossing", "starboard_beam", "port_beam")
                else "generic"
            )
            sda, estimated = sda_m(
                own_config.length_m, target.length_m, encounter,
                sda_pass_position, target.ship_type
            )
            if estimated and sda > 0.0:
                note += "；SDA 按附录B表值插值/外推估算"
            elif sda <= 0.0:
                note += "；当前会遇分类无附录B对应SDA表项"
            visibility_factor = (2.0 if environment is not None and
                                 environment.visibility_nm < environment.min_normal_visibility_nm else 1.0)
            sda_pref = (visibility_factor + 0.274) * sda
            pass_side, preferred_turn = _preferred_manoeuvre(own_state, target, encounter)

            dcrc, time_to_dcrc = math.inf, math.inf
            dcqs, time_to_dcqs = math.inf, math.inf
            if preferred_turn:
                dcrc, time_to_dcrc = _critical_distance_threshold(
                    own_state, target.state, own_config, preferred_turn, sda_pref
                )
                dcqs, time_to_dcqs = _critical_distance_threshold(
                    own_state, target.state, own_config,
                    math.copysign(90.0, preferred_turn), sda
                )

            overlap = rng <= 0.5 * (own_config.length_m + target.length_m)
            if overlap or rng <= sda * 1.10:
                risk = "紧迫局面"
                score = min(1.0, 0.82 + 0.18 * max(0.0, 1.0-rng/max(sda, 1.0)))
            elif tcpa <= 0.0:
                risk = "无危险"
                score = 0.0
            elif dcpa > sda:
                # A safe nominal CPA is monitored but does not start a manoeuvre.
                if dcpa <= sda_pref and rng <= max(3.0*sda_pref, dcrc if math.isfinite(dcrc) else 0):
                    risk, score = "关注", 0.05
                else:
                    risk, score = "无危险", 0.0
            elif math.isfinite(dcrc):
                # Far from DCRC: keep the encounter under observation, not MPC.
                # Entering the DCRC band raises the score above the engine's
                # activation threshold; DCQS marks the close-quarters zone.
                if rng > dcrc:
                    risk, score = "关注", 0.06
                elif (math.isfinite(dcqs) and (rng <= dcqs or time_to_dcqs <= 1e-6)):
                    risk = "紧迫局面"
                    score = min(1.0, 0.78 + 0.22*max(0.0, 1.0-rng/max(dcqs, 1.0)))
                else:
                    span = max(1.0, dcrc-(dcqs if math.isfinite(dcqs) else 0.0))
                    penetration = max(0.0, min(1.0, (dcrc-rng)/span))
                    risk = "碰撞危险"
                    score = 0.12 + 0.56*penetration
            else:
                # Degenerate/near-parallel cases do not yield a critical-distance
                # root. Retain a conservative short-TCPA fallback rather than
                # silently ignoring a close collision geometry.
                if rng <= 3.0*sda or tcpa <= max(600.0, self.close_time_s):
                    risk = "碰撞危险"
                    time_factor = max(0.0, 1.0-min(tcpa, 1800.0)/1800.0)
                    score = min(0.70, 0.20+0.35*max(0.0, 1.0-dcpa/max(sda,1.0))+0.15*time_factor)
                else:
                    risk, score = "关注", 0.06

            if memory is None and score >= 0.05 and observed_encounter not in ("none", "following"):
                memory = {
                    "encounter": observed_encounter,
                    "give_way": observed_give_way,
                    "rule": observed_rule,
                    "bearing": observed_bearing,
                    "note": observed_note,
                    "last_range": rng,
                    "receding_since": None,
                    "last_elapsed": now,
                }
                self._encounter_memory[target.target_id] = memory
                encounter = memory["encounter"]
                give_way = memory["give_way"]
                rule = memory["rule"]
                bearing = observed_bearing

            memory = self._encounter_memory.get(target.target_id)
            if memory is not None:
                last_time = memory.get("last_elapsed")
                if now is not None and (last_time is None or now > last_time):
                    prev_range = float(memory.get("last_range", rng))
                    safely_receding = (
                        tcpa < -60.0
                        and rng >= max(2.0*sda, 2.0*max(own_config.length_m, target.length_m))
                        and rng > prev_range
                    )
                    if safely_receding:
                        if memory.get("receding_since") is None:
                            memory["receding_since"] = now
                    else:
                        memory["receding_since"] = None
                    memory["last_range"] = rng
                    memory["last_elapsed"] = now
                    since = memory.get("receding_since")
                    if since is not None and now-since >= 6.0:
                        self._encounter_memory.pop(target.target_id, None)
                        memory = None

            if memory is not None:
                encounter = memory["encounter"]
                give_way = memory["give_way"]
                rule = memory["rule"]
                bearing = observed_bearing
                note2 = memory["note"]
                if encounter != observed_encounter:
                    note2 += "；保持本次会遇的初始责任分类"
            else:
                encounter, give_way, rule, _, bearing, note2 = assess_rule(own_state, target.state, risk)

            if math.isfinite(dcrc):
                note2 += "；按附录A思路估算DCRC/DCQS干预窗口（工程近似）"
            if pass_side != 'unspecified' and encounter == 'own_overtaking':
                note2 += f"；追越优先评估{pass_side}舷通过（安全约束优先）"

            assessments.append(EncounterAssessment(
                target_id=target.target_id,
                encounter=encounter,
                own_give_way=give_way,
                rule=rule,
                relative_bearing_deg=bearing,
                range_m=rng,
                dcpa_m=dcpa,
                tcpa_s=tcpa,
                sda_m=sda,
                risk_level=risk,
                risk_score=score,
                notes=note2 + ("；SDA 按附录B表值插值/外推估算" if estimated and sda > 0.0 else
                               "；当前会遇分类无附录B对应SDA表项" if sda <= 0.0 else ""),
                pass_side=pass_side,
                preferred_turn_deg=preferred_turn,
                dcrc_m=dcrc,
                dcqs_m=dcqs,
                time_to_dcrc_s=time_to_dcrc,
                time_to_dcqs_s=time_to_dcqs,
                sda_pass_position=sda_pass_position,
                sda_estimated=estimated,
            ))

        assessments.sort(key=lambda a: (a.risk_score, -a.tcpa_s), reverse=True)
        focus = assessments[0].target_id if assessments and assessments[0].risk_score > 0 else None
        max_score = max((a.risk_score for a in assessments), default=0.0)
        emergency = any(a.risk_level == "紧迫局面" for a in assessments)
        return RiskSnapshot(assessments, focus, max_score, emergency)
