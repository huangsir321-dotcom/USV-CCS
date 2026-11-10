"""Finite-candidate receding-horizon controller; policy selects actual trajectories."""
from dataclasses import dataclass, field
import math
from .actions import (ACTION_SPECS, ACTION_INTENTS, SPEED_MODES, speed_setpoint,
                      is_slow_mode, is_accelerate_mode)
from .colregs import is_port_stand_on_encounter, is_starboard_give_way_encounter
from .dynamics import Simplified3DOFDynamics
from .geometry import swept_separation, point_segment_distance
from .models import PlannerResult, wrap_angle
from .obstacles import point_in_polygon, _segments_intersect
from .rules import candidate_rule_reasons, DEFAULT_RULES


@dataclass
class CandidatePlan(PlannerResult):
    candidate_id: int = 0
    offset_deg: float = 0.0
    speed_mode: str = 'hold'
    course_cmd_rad: float = 0.0
    first_endpoint: tuple = (0.0, 0.0)
    violation: float = 0.0
    feasible: bool = False
    reasons: tuple = ()
    status: str = 'feasible'
    times: list = field(default_factory=list)
    preference_clearance_m: float = math.inf
    rule_reasons: tuple = ()


def segment_polygon_distance(start, end, vertices):
    """Exact distance between a segment and a polygon boundary/interior."""
    if len(vertices) < 3:
        return math.inf
    if point_in_polygon(*start, vertices) or point_in_polygon(*end, vertices):
        return 0.0
    best = math.inf
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        if _segments_intersect(start, end, a, b):
            return 0.0
        d = min(point_segment_distance(*start, *a, *b),
                point_segment_distance(*end, *a, *b),
                point_segment_distance(*a, *start, *end),
                point_segment_distance(*b, *start, *end))
        if d < best:
            best = d
    return best


def segment_circle_distance(start, end, center, radius):
    """Exact segment-to-circle clearance (zero when the segment touches it)."""
    return max(0.0, point_segment_distance(center[0], center[1],
                                           start[0], start[1], end[0], end[1])
               - max(0.0, float(radius)))


def _segment_aabb_distance(start, end, bounds):
    """Lower bound between a segment and an obstacle AABB; O(1) broad phase."""
    min_x, min_y, max_x, max_y = bounds
    seg_min_x, seg_max_x = min(start[0], end[0]), max(start[0], end[0])
    seg_min_y, seg_max_y = min(start[1], end[1]), max(start[1], end[1])
    dx = max(0.0, min_x - seg_max_x, seg_min_x - max_x)
    dy = max(0.0, min_y - seg_max_y, seg_min_y - max_y)
    return math.hypot(dx, dy)


def segment_obstacle_distance(start, end, obstacle, early_exit_distance_m=None):
    """Distance to a static obstacle with analytic/broad-phase fast paths.

    When ``early_exit_distance_m`` is supplied, a segment/AABB lower bound
    already beyond that threshold is enough to prove that the exact polygon
    cannot be contacted; callers that only need a threshold decision can skip
    the expensive edge scan safely.
    """
    if getattr(obstacle, 'shape_type', 'polygon') == 'circle' and obstacle.center is not None:
        return segment_circle_distance(start, end, obstacle.center,
                                       obstacle.radius_m or 0.0)
    vertices = obstacle.vertices
    if len(vertices) < 3:
        return math.inf
    if early_exit_distance_m is not None:
        xs = [point[0] for point in vertices]
        ys = [point[1] for point in vertices]
        bounds = (min(xs), min(ys), max(xs), max(ys))
        lower_bound = _segment_aabb_distance(start, end, bounds)
        if lower_bound > max(0.0, float(early_exit_distance_m)):
            return lower_bound
    return segment_polygon_distance(start, end, vertices)


def _prepare_obstacles(obstacles):
    """Prepare immutable geometry once per decision rather than per path sample."""
    prepared = []
    for obstacle in obstacles:
        if getattr(obstacle, 'shape_type', 'polygon') == 'circle' and obstacle.center is not None:
            cx, cy = obstacle.center
            radius = max(0.0, float(obstacle.radius_m or 0.0))
            bounds = (cx-radius, cy-radius, cx+radius, cy+radius)
            prepared.append((obstacle, bounds, True))
            continue
        vertices = obstacle.vertices
        if len(vertices) < 3:
            continue
        xs = [p[0] for p in vertices]
        ys = [p[1] for p in vertices]
        bounds = (min(xs), min(ys), max(xs), max(ys))
        prepared.append((obstacle, bounds, False))
    return prepared


class DWAMPCPlanner:
    def __init__(self, own_config, horizon_s=1000.0, prediction_dt_s=10.0, rudder_samples=7):
        self.config = own_config
        self.dynamics = Simplified3DOFDynamics(own_config)
        # A local MPC should not integrate 1,000 s at the finest control step
        # for every action. Risk assessment remains global; candidate prediction
        # uses a bounded local horizon and is refreshed every decision period.
        self.horizon_s = min(float(horizon_s), 360.0)
        self.prediction_dt_s = prediction_dt_s
        self.max_horizon_s = 600.0
        self.adaptive_horizon = True
        self.last_rudder_cmd = 0.0
        self.last_speed_cmd = None
        self.last_course_cmd_rad = None
        self.last_score = None
        self.last_candidates = []
        self.oracle_prediction = False
        self.rule_config = DEFAULT_RULES
        self.preferred_margin_ratio = .274
        self.position_uncertainty_m = 0.0
        self.velocity_uncertainty_mps = 0.0

    @staticmethod
    def _route_guidance(own, rx, ry, start, length, environment=None, yaw_tau_s=32.0,
                        recovery_horizon_s=None):
        """Track-line guidance with a bounded, cross-track-aware look-ahead.

        In recovery mode the look-ahead is tied to the distance the vessel can
        travel during the local prediction horizon. This makes the nominal
        guidance angle compatible with the planner's terminal cross-track
        objective instead of asking the discrete action offsets to cancel an
        over-aggressive intercept angle.
        """
        dx, dy = rx - start[0], ry - start[1]
        distance = max(1.0, math.hypot(dx, dy))
        ux, uy = dx / distance, dy / distance
        progress = (own.x - start[0])*ux + (own.y - start[1])*uy
        signed_cross = (own.x-start[0])*(-uy) + (own.y-start[1])*ux
        if recovery_horizon_s is None:
            lookahead = max(3.0*length, min(8.0*length, 0.75*abs(signed_cross)))
        else:
            # A 0.8-horizon pure-pursuit distance makes the selected heading
            # close the lateral error over the same horizon that MPC scores.
            # Retain a hull-length floor for large, sluggish vessels and cap
            # the intercept distance to avoid vanishingly small angles.
            lookahead = max(1.5*length,
                            min(8.0*length, 0.8*max(2.0, own.u)*recovery_horizon_s))
        route_heading = math.atan2(ux, uy)

        # Predict cross-track travel during the vessel's yaw response. A pure
        # geometric intercept keeps steering toward the line even when a large
        # ship is already moving laterally toward it; the long yaw lag then
        # carries the ship across the route and makes the command alternate.
        # Ground velocity is used because the route is fixed in earth coordinates.
        current_x = getattr(environment, "current_x_mps", 0.0)
        current_y = getattr(environment, "current_y_mps", 0.0)
        vx = own.u*math.sin(own.psi) + own.v*math.cos(own.psi) + current_x
        vy = own.u*math.cos(own.psi) - own.v*math.sin(own.psi) + current_y
        cross_rate = vx*(-uy) + vy*ux
        anticipation_s = max(25.0, min(150.0, 1.5*float(yaw_tau_s)))
        predicted_cross = signed_cross + cross_rate*anticipation_s
        correction = math.atan2(predicted_cross, lookahead)

        # Bound the commanded intercept angle for large-displacement recovery.
        # The vessel model still applies yaw inertia, rate limits and rudder lag;
        # the setpoint is gradual and cannot cause an instantaneous heading jump.
        max_correction_deg = max(22.0, min(36.0, 36.0*math.sqrt(200.0/max(50.0, length))))
        correction = max(-math.radians(max_correction_deg),
                         min(math.radians(max_correction_deg), correction))
        intercept_heading = wrap_angle(route_heading+correction)
        return (route_heading, intercept_heading, (ux, uy), distance, progress)

    def _course_controller(self, state, course, environment=None, elapsed_s=0.0):
        max_r = math.radians(self.config.max_yaw_rate_deg_s)
        # Feed forward the known environmental yaw disturbance. Without this
        # term the model's wind/wave yaw acceleration creates a steady heading
        # bias that a proportional heading command cannot remove; the vessel
        # can then travel parallel to the route with a fixed cross-track error.
        disturbance = self.dynamics.yaw_disturbance_acceleration(
            state, environment, elapsed_s
        )
        disturbance_compensation = disturbance * self.config.yaw_tau_s
        desired = (wrap_angle(course-state.psi)/max(45.0, 1.4*self.config.yaw_tau_s)
                   -0.85*state.r - disturbance_compensation)
        desired = max(-max_r, min(max_r, desired))
        gain = max(1e-4, self.config.yaw_gain*max(0.25, state.u/7))
        limit = math.radians(self.config.max_rudder_deg)
        return max(-limit, min(limit, desired/gain))

    def _speed_candidates(self, state, mission_speed_mps=None, reference_speed=None):
        anchor = state.u if reference_speed is None else float(reference_speed)
        speed_state = state.copy()
        speed_state.u = anchor
        return [speed_setpoint(speed_state, self.config, mode,
                               reference_speed=anchor,
                               mission_speed_mps=mission_speed_mps)
                for mode in SPEED_MODES]

    def generate_candidates(self, own_state, targets, risk_snapshot, route_x=0., route_y=12000.,
                            control_period_s=2., obstacles=None, environment=None, elapsed_s=0.,
                            route_start=(0.,0.), reference_heading=None, reference_speed=None,
                            use_rule_mask=True, encounter_references=None,
                            route_recovery=False, cruise_speed_mps=None,
                            lock_speed_to_cruise=False):
        obstacles = obstacles or []
        prepared_obstacles = _prepare_obstacles(obstacles)
        active = [t for t in targets if t.active]
        assessments = {a.target_id: a for a in risk_snapshot.assessments}
        horizon = min(self.max_horizon_s, self.horizon_s) if active else 60.
        if self.adaptive_horizon and active:
            # The earliest relevant CPA governs the local window. Using the
            # *largest* TCPA previously expanded every plan toward 1,500 s.
            tcpas = [a.tcpa_s+120.0 for a in assessments.values()
                     if a.tcpa_s > 0 and a.risk_score >= .08]
            if tcpas:
                horizon = min(self.max_horizon_s, max(180.0, min(horizon, min(tcpas))))
        route_heading, guidance, unit, _, _ = self._route_guidance(
            own_state, route_x, route_y, route_start, self.config.length_m,
            environment, self.config.yaw_tau_s,
            recovery_horizon_s=horizon if route_recovery else None)
        base = guidance if reference_heading is None else reference_heading
        references=dict(encounter_references or {})
        for a in risk_snapshot.assessments:
            references.setdefault(a.target_id,(own_state.psi,own_state.u))
        normal_stand_on=[a for a in risk_snapshot.assessments if not a.own_give_way
                         and is_port_stand_on_encounter(a.encounter)
                         and self.rule_config.activation_score<=a.risk_score<self.rule_config.stand_on_intervention_score
                         and a.risk_level!='紧迫局面']
        active_give_way=[a for a in risk_snapshot.assessments
                         if a.own_give_way and a.encounter not in ('none','following')
                         and a.risk_score >= self.rule_config.activation_score]
        tactical_active=bool(risk_snapshot.emergency or active_give_way)
        active_overtakes=[a for a in active_give_way if a.encounter=='own_overtaking']
        active_crossing_or_headon=[a for a in active_give_way
                                   if a.encounter in ('starboard_crossing','starboard_beam','head_on')]
        hold_speed=own_state.u
        if normal_stand_on:
            base,hold_speed=references[normal_stand_on[0].target_id]
        if not math.isfinite(control_period_s) or control_period_s <= 0:
            raise ValueError('control_period_s must be finite and positive')
        # The feedback law is refreshed on the actual control clock throughout
        # prediction; coarse controller holds can mispredict small agile vessels.
        count=max(1,math.ceil(horizon/control_period_s))
        times=[min(k*control_period_s,horizon) for k in range(count+1)]
        predictions = {t.target_id: [(t.state.x,t.state.y)]+t.prediction_path(
            times[1:], environment, oracle=self.oracle_prediction) for t in active}
        bank = []
        for index, spec in enumerate(ACTION_SPECS):
            course = base+math.radians(spec.offset_deg)
            speed_state = own_state.copy()
            if lock_speed_to_cruise:
                # Route recovery/post-recovery cruise uses one fixed mission
                # target. It is not relative to the current speed, so repeated
                # planning cannot ratchet the vessel faster after each decision.
                cruise_target = (own_state.u if cruise_speed_mps is None
                                 else float(cruise_speed_mps))
                speed = max(self.config.min_speed_mps,
                            min(self.config.max_speed_mps, cruise_target))
            else:
                if spec.speed_mode == 'hold':
                    speed_state.u = (float(reference_speed) if tactical_active and reference_speed is not None
                                     else hold_speed)
                speed_anchor = (float(reference_speed) if tactical_active and reference_speed is not None
                                else hold_speed)
                speed = speed_setpoint(speed_state, self.config, spec.speed_mode,
                                       reference_speed=speed_anchor,
                                       mission_speed_mps=cruise_speed_mps)
            rule_reasons = list(candidate_rule_reasons(spec.offset_deg, spec.speed_mode,
                risk_snapshot.assessments,config=self.rule_config,course_cmd_rad=course,
                speed_cmd_mps=speed,references=references))
            reasons = list(rule_reasons) if use_rule_mask else []
            if lock_speed_to_cruise and spec.speed_mode != 'hold':
                reasons.append('speed:mission_cruise_setpoint_locked')
            if not tactical_active and not lock_speed_to_cruise and is_accelerate_mode(spec.speed_mode):
                reasons.append('speed:no_active_collision_manoeuvre')
            if (not tactical_active and not lock_speed_to_cruise and not prepared_obstacles
                and is_slow_mode(spec.speed_mode)):
                reasons.append('speed:no_slow_without_obstacle_or_encounter')
            if active_crossing_or_headon and is_accelerate_mode(spec.speed_mode) and not risk_snapshot.emergency:
                # This is a simulator preference, not a standalone COLREGs clause:
                # crossing/head-on avoidance should first use course change or
                # slowing, not gain closure speed without a demonstrated need.
                reasons.append('speed:no_unjustified_acceleration_crossing_headon')
            if environment is not None and (environment.visibility_nm < environment.min_normal_visibility_nm
                or environment.wind_speed_mps > environment.max_normal_wind_mps
                or environment.wave_level > environment.max_normal_wave_level):
                if is_accelerate_mode(spec.speed_mode):
                    reasons.append('ODD:no_acceleration')
            if ((is_slow_mode(spec.speed_mode) and speed >= own_state.u-0.05)
                or (is_accelerate_mode(spec.speed_mode) and speed <= own_state.u+0.05)):
                reasons.append('actuator:duplicate_speed_limit')
            # Do not integrate complete trajectories for actions already excluded
            # by a global speed-mode lock. Keep their fixed action slots and audit
            # reasons for PPO, but skip the costly path sweep. Geometry/rule-masked
            # candidates still run through full prediction for minimum-risk fallback.
            skip_reasons = {'speed:mission_cruise_setpoint_locked',
                            'speed:no_active_collision_manoeuvre',
                            'speed:no_slow_without_obstacle_or_encounter',
                            'speed:no_unjustified_acceleration_crossing_headon',
                            'ODD:no_acceleration'}
            if any(reason in skip_reasons for reason in reasons):
                first_rudder = self._course_controller(own_state, course, environment, elapsed_s)
                bank.append(CandidatePlan(first_rudder,speed,[(own_state.x,own_state.y)],1e6,False,
                    ACTION_INTENTS[index],math.inf,';'.join(reasons),candidate_id=index,
                    offset_deg=spec.offset_deg,speed_mode=spec.speed_mode,course_cmd_rad=course,
                    first_endpoint=(own_state.x,own_state.y),violation=1e6,feasible=False,
                    reasons=tuple(reasons),status='masked',times=[0.0],
                    preference_clearance_m=math.inf,rule_reasons=tuple(rule_reasons)))
                continue
            state = own_state.copy()
            path = [(state.x,state.y)]
            minimum, pref_min, violation, cross_cost, rudder_cost = math.inf, math.inf, 0., 0., 0.
            first_rudder = self._course_controller(state, course, environment, elapsed_s)
            for k in range(1,len(times)):
                dt = times[k]-times[k-1]
                rudder = self._course_controller(state, course, environment,
                                                 elapsed_s+times[k-1])
                state = self.dynamics.step(state, rudder, speed, dt, environment, elapsed_s+times[k-1])
                previous, current = path[-1], (state.x,state.y)
                path.append(current)
                curvature_pad = self.config.max_speed_mps*math.radians(self.config.max_yaw_rate_deg_s)*dt*dt/8
                for target in active:
                    tp, tn = predictions[target.target_id][k-1:k+1]
                    a = assessments.get(target.target_id)
                    sda = a.sda_m if a else max(200.,2*self.config.length_m)
                    uncertainty=max(0.,self.position_uncertainty_m)+max(0.,self.velocity_uncertainty_mps)*times[k]
                    distance = swept_separation(previous,current,tp,tn)-curvature_pad-uncertainty
                    clearance = distance-sda
                    minimum = min(minimum,clearance)
                    visibility_factor=(2. if environment is not None and
                                       environment.visibility_nm<environment.min_normal_visibility_nm else 1.)
                    # Engineering preference on adjusted SDA, explicitly separate
                    # from the hard distance. Not a literal Annex-A reproduction.
                    pref_min = min(pref_min,distance-(visibility_factor+self.preferred_margin_ratio)*sda)
                    if clearance < 0:
                        violation += (-clearance/max(sda,1.))**2*dt/horizon
                        if 'SDA:'+target.target_id not in reasons:
                            reasons.append('SDA:'+target.target_id)
                    if (a and a.own_give_way and is_starboard_give_way_encounter(a.encounter)
                        and a.risk_score >= self.rule_config.activation_score):
                        psi = target.state.psi
                        r0=(previous[0]-tp[0],previous[1]-tp[1])
                        r1=(current[0]-tn[0],current[1]-tn[1])
                        lat0=r0[0]*math.cos(psi)-r0[1]*math.sin(psi)
                        lat1=r1[0]*math.cos(psi)-r1[1]*math.sin(psi)
                        if lat0*lat1 < 0:
                            alpha=abs(lat0)/(abs(lat0)+abs(lat1))
                            forward=(r0[0]*(1-alpha)+r1[0]*alpha)*math.sin(psi)+(r0[1]*(1-alpha)+r1[1]*alpha)*math.cos(psi)
                            if forward > 0:
                                reason=target.target_id+':R15_crossing_ahead'
                                if reason not in rule_reasons:
                                    rule_reasons.append(reason)
                                if use_rule_mask and reason not in reasons:
                                    reasons.append(reason)
                obstacle_margin = .5*self.config.length_m + 100.0 + curvature_pad
                for obstacle, bounds, is_circle in prepared_obstacles:
                    # Segment/AABB separation is a conservative lower bound.
                    # When it already exceeds the required clearance, the exact
                    # obstacle geometry cannot be relevant and is skipped in O(1).
                    # Otherwise compute exact circle/segment or segment/polygon
                    # clearance to avoid false negatives near the obstacle.
                    broad_clearance = _segment_aabb_distance(previous, current, bounds) - obstacle_margin
                    if broad_clearance > 0.0:
                        clearance = broad_clearance
                    elif is_circle:
                        clearance = segment_circle_distance(
                            previous, current, obstacle.center, obstacle.radius_m or 0.0
                        ) - obstacle_margin
                    else:
                        clearance = segment_polygon_distance(
                            previous, current, obstacle.vertices
                        ) - obstacle_margin
                    minimum = min(minimum, clearance)
                    if clearance < 0:
                        violation += (-clearance/100.0)**2*dt/horizon
                        reason='obstacle:'+obstacle.obstacle_id
                        if reason not in reasons:
                            reasons.append(reason)
                cross=abs((state.x-route_start[0])*(-unit[1])+(state.y-route_start[1])*unit[0])
                cross_cost += cross*dt/horizon/1000
                rudder_cost += math.degrees(rudder)**2*dt/horizon
            feasible = not reasons
            terminal_cross=abs((state.x-route_start[0])*(-unit[1])+(state.y-route_start[1])*unit[0])
            forward=(state.x-own_state.x)*unit[0]+(state.y-own_state.y)*unit[1]
            continuity=abs(math.degrees(wrap_angle(course-(self.last_course_cmd_rad if self.last_course_cmd_rad is not None else own_state.psi))))
            if route_recovery:
                # Recovery is a distinct control objective: terminal and
                # integrated cross-track convergence take priority over making
                # distance along the route.
                initial_cross = abs((own_state.x-route_start[0])*(-unit[1])
                                    +(own_state.y-route_start[1])*unit[0])
                recovery_gain = (initial_cross-terminal_cross) / max(initial_cross, self.config.length_m)
                route_cost = 1.50*cross_cost + .020*terminal_cross - .50*recovery_gain
                offset_weight = .015
                continuity_weight = .015
                progress_reward = 0.0
            elif tactical_active:
                # During collision avoidance, route progress must not reward
                # straight-line acceleration at the expense of the encounter.
                route_cost = .12*cross_cost + .0005*terminal_cross
                offset_weight = .018
                continuity_weight = .018
                progress_reward = 0.0
            else:
                route_cost = .4*cross_cost + .002*terminal_cross
                offset_weight = .035
                continuity_weight = .025
                progress_reward = -.0004*max(0,forward)

            overtake_preference = 0.0
            for encounter_assessment in active_overtakes:
                sign = -1.0 if encounter_assessment.pass_side == 'port' else 1.0
                preferred = abs(encounter_assessment.preferred_turn_deg) or 10.0
                if spec.offset_deg * sign >= 10.0:
                    # Soft preference for a visible pass-side turn (10 degrees
                    # is the open-water overtaking reference, not a fixed command).
                    # A safe alternative remains available if this side is blocked.
                    overtake_preference += -0.55 + 0.0012*(abs(spec.offset_deg)-preferred)**2
                    if is_slow_mode(spec.speed_mode):
                        overtake_preference += 0.45
                    elif spec.speed_mode == 'accelerate_25':
                        # The moderate speed level is the preferred overtake
                        # aid when ownship already has positive closure speed.
                        # The 50% level is left for geometries where it wins on
                        # predicted clearance/time, rather than being rewarded
                        # merely for being faster.
                        target_state = next((t.state for t in active
                                             if t.target_id == encounter_assessment.target_id), None)
                        if target_state is not None and own_state.u > target_state.u + 0.1:
                            overtake_preference -= 0.28
                        else:
                            overtake_preference -= 0.08
                    elif spec.speed_mode == 'accelerate_50':
                        # A stronger speed increase is not automatically better;
                        # the score prefers the moderate level unless geometry
                        # demonstrates that a larger speed margin is useful.
                        overtake_preference -= 0.10
                elif abs(spec.offset_deg) < 10.0:
                    overtake_preference += 0.75
                else:
                    overtake_preference += 1.40

            # GB/T 46851-2025 Appendix A Table A.1 provides indicative initial
            # course-alteration angles by encounter and water area. They are
            # implemented as soft costs (never as invented COLREG obligations).
            rule_angle_cost = 0.0
            for a in risk_snapshot.assessments:
                if (not a.own_give_way or
                        a.risk_score < self.rule_config.activation_score):
                    continue
                ref = references.get(a.target_id)
                if ref is None:
                    effective_turn = spec.offset_deg
                else:
                    effective_turn = math.degrees(wrap_angle(course-ref[0]))
                if a.encounter == 'head_on':
                    desired = float(a.preferred_turn_deg or 20.0)
                    rule_angle_cost += self.rule_config.head_on_reference_weight * (effective_turn-desired)**2
                elif is_starboard_give_way_encounter(a.encounter):
                    desired = float(a.preferred_turn_deg or 30.0)
                    rule_angle_cost += self.rule_config.crossing_reference_weight * (effective_turn-desired)**2

            score=(route_cost+offset_weight*abs(spec.offset_deg)
                   +.02*math.sqrt(rudder_cost)+continuity_weight*continuity
                   +.08*abs(speed-own_state.u)+progress_reward
                   +2*max(0,-pref_min/1000)**2+overtake_preference+rule_angle_cost)
            bank.append(CandidatePlan(first_rudder,speed,path,score,feasible,ACTION_INTENTS[index],
                minimum,'Feasible candidate' if feasible else ';'.join(reasons),
                candidate_id=index,offset_deg=spec.offset_deg,speed_mode=spec.speed_mode,
                course_cmd_rad=course,first_endpoint=path[1],violation=violation,feasible=feasible,
                reasons=tuple(reasons),status='feasible' if feasible else 'infeasible',times=times,
                preference_clearance_m=pref_min,rule_reasons=tuple(rule_reasons)))
        self.last_candidates=bank
        return bank

    def select_candidate(self, bank):
        allowed=[c for c in bank if c.feasible]
        if allowed:
            return min(allowed,key=lambda c:(c.score,c.candidate_id))

        # When the conservative SDA/obstacle sweep rejects every trajectory,
        # do not let the geometric-violation fallback silently reverse an
        # explicit COLREG direction (most importantly Rule 14 starboard in a
        # head-on encounter). First search for the least-risk candidate that
        # still honours the core directional/action requirements. Conditional
        # preferences such as Rule 15's "if circumstances admit" remain a
        # secondary factor, so they may be relaxed only when needed.
        core_codes = ('R14_starboard_required', 'R16_substantial_action')
        core_compliant = [c for c in bank if not any(
            code in reason for reason in c.rule_reasons for code in core_codes
        )]
        if core_compliant:
            chosen = min(core_compliant, key=lambda c:(
                c.violation,
                sum('R15_crossing_ahead' in r or 'R17c_avoid_port' in r
                    for r in c.rule_reasons),
                c.score, c.candidate_id))
            chosen.status='minimum_risk'
            chosen.message=('No fully feasible candidate; selected minimum-risk action '
                            'while preserving Rule 14 direction and Rule 16 substantial-action requirements: '
                            + ';'.join(chosen.reasons))
            return chosen

        # Last resort: no candidate can retain the core rule direction. Keep
        # the minimum physical-violation fallback explicit instead of labeling
        # it safe or rule-compliant; the audit reasons are returned to the GUI/log.
        chosen=min(bank,key=lambda c:(c.violation,len(c.reasons),c.score,c.candidate_id))
        chosen.status='minimum_risk'
        chosen.message='No feasible or core-rule-compliant candidate; minimum-risk fallback requiring review: '+ ';'.join(chosen.reasons)
        return chosen

    def commit_candidate(self, chosen):
        self.last_course_cmd_rad=chosen.course_cmd_rad
        self.last_rudder_cmd=chosen.rudder_cmd_rad
        self.last_speed_cmd=chosen.speed_cmd_mps
        self.last_score=chosen.score
        return chosen

    def plan(self, own_state, targets, risk_snapshot, route_x=0., route_y=12000., intent='maintain',
             control_period_s=2., obstacles=None, environment=None, elapsed_s=0., route_start=(0.,0.),
             route_recovery=False, cruise_speed_mps=None, lock_speed_to_cruise=False):
        return self.commit_candidate(self.select_candidate(self.generate_candidates(
            own_state,targets,risk_snapshot,route_x,route_y,control_period_s,obstacles,
            environment,elapsed_s,route_start,route_recovery=route_recovery,
            cruise_speed_mps=cruise_speed_mps,lock_speed_to_cruise=lock_speed_to_cruise)))
