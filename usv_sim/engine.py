"""Shared training/deployment execution chain for candidate-based collision avoidance."""
import copy
import json
import math
import time
from pathlib import Path
from .actions import ACTION_NAMES, ACTION_COUNT, DECISION_PERIOD_S, CONTROL_DT_S
from .models import SimulationMetrics
from .dynamics import Simplified3DOFDynamics
from .risk import MultiShipRiskAssessor
from .planner import DWAMPCPlanner, segment_obstacle_distance
from .ppo_agent import PPOPolicy, build_observation
from .odd import ODDMonitor
from .geometry import swept_separation
from .rules import phase_for, DEFAULT_RULES
from .safety import check_control


class SimulationEngine:
    def __init__(self, scenario, ppo_checkpoint=None):
        self.scenario=scenario
        self.config=copy.deepcopy(scenario.own_config)
        self.own=scenario.own_state.copy()
        # The scenario's initial own-ship speed is the mission cruise setpoint.
        # Tactical speed changes during collision avoidance must not silently
        # redefine this value; recovery and post-recovery cruise return to it.
        self.cruise_speed_mps=max(self.config.min_speed_mps,
                                  min(self.config.max_speed_mps, float(self.own.u)))
        self.route_recovery_speed_tolerance_mps=max(0.10, 0.02*self.cruise_speed_mps)
        self.targets=copy.deepcopy(scenario.targets)
        # GUI obstacle editing intentionally shares this list; initial targets do not.
        self.obstacles=scenario.obstacles
        self.environment=scenario.environment
        self.dt=CONTROL_DT_S
        self.policy_update_period_s=DECISION_PERIOD_S
        self.route_index=0
        self.route_start_x,self.route_start_y=self.own.x,self.own.y
        self._set_route(scenario.route[0])
        self.dynamics=Simplified3DOFDynamics(self.config)
        self.risk_engine=MultiShipRiskAssessor()
        self.planner=DWAMPCPlanner(self.config)
        self.policy=PPOPolicy(ppo_checkpoint)
        self.odd_monitor=ODDMonitor()
        self.odd_status=self.odd_monitor.evaluate(self.environment)
        self.metrics=SimulationMetrics()
        self.own_track=[(self.own.x,self.own.y)]
        self.target_tracks={t.target_id:[(t.state.x,t.state.y)] for t in self.targets}
        self.logs=[]
        self.running=False
        self.last_snapshot=self.risk_engine.assess(self.own,self.config,self.targets,elapsed_s=0,environment=self.environment)
        self.last_plan=None
        self.last_action=0
        self.last_executed_action=0
        self.last_intervention=False
        self.intervention_reasons=()
        self.avoidance_started=False
        self.maneuver_active=False
        self.route_recovery_active=False
        self._route_recovery_stable_steps=0
        self.route_recovery_heading_tolerance_rad=math.radians(8.0)
        self.route_recovery_stable_required=5  # 5 x 2 s control steps
        self.maneuver_target_ids=set()
        self.maneuver_intent='maintain'
        self.reference_heading=None
        self.reference_speed=None
        self.encounter_references={}
        self.phase='monitor'
        self.last_policy_source='rule_baseline_monitor'
        self.last_decision_policy_authorized=False
        self.last_decision_gate_reason='monitor_phase_waits_for_rule_trigger'
        self.last_maneuver_start_time_s=None
        self.candidates=[]
        self.baseline_action=0
        self.decision_infeasible=False
        self.use_rule_mask=True
        self.use_safety_monitor=True
        self._prepared_key=None
        self._prepared=None
        self._next_policy_update_s=0.
        self._held_course=None
        self._held_speed=None
        self._contact_ids=set()
        self._obstacle_contact_ids=set()
        self._sda_violation_ids=set()

    def _set_route(self, endpoint):
        self.route_x,self.route_y=endpoint
        self.route_dx=self.route_x-self.route_start_x
        self.route_dy=self.route_y-self.route_start_y
        self.route_length=max(1.,math.hypot(self.route_dx,self.route_dy))
        self.route_ux=self.route_dx/self.route_length
        self.route_uy=self.route_dy/self.route_length

    @property
    def route_recovery_tolerance_m(self):
        return .1*self.config.length_m

    def route_cross_track_distance_m(self,x=None,y=None):
        x=self.own.x if x is None else x
        y=self.own.y if y is None else y
        return abs((x-self.route_start_x)*(-self.route_uy)+(y-self.route_start_y)*self.route_ux)

    def is_within_route_recovery_corridor(self,distance_m=None):
        distance=self.route_cross_track_distance_m() if distance_m is None else distance_m
        return distance <= self.route_recovery_tolerance_m+1e-9

    @property
    def cruise_speed_locked(self):
        """Keep the mission cruise speed during recovery and after it completes.

        A new avoidance manoeuvre clears route_returned and temporarily unlocks
        tactical speed actions again.
        """
        return self.route_recovery_active or (self.metrics.route_returned and not self.maneuver_active)

    @property
    def cruise_speed_error_mps(self):
        return abs(float(self.own.u)-self.cruise_speed_mps)

    @property
    def route_heading_error_deg(self):
        """Absolute heading error relative to the active route segment."""
        route_heading=math.atan2(self.route_ux,self.route_uy)
        error=abs((self.own.psi-route_heading+math.pi)%(2*math.pi)-math.pi)
        return math.degrees(error)

    def _force_policy_replan(self):
        """Do not hold an old policy command across a control-authority change."""
        self._next_policy_update_s=float(self.metrics.elapsed_s)
        self._held_course=None
        self._held_speed=None
        self._prepared_key=None

    def _update_encounter(self):
        snapshot=self.last_snapshot
        for a in snapshot.assessments:
            if a.risk_score >= .12 and a.encounter not in ('none','following'):
                self.encounter_references.setdefault(a.target_id,(self.own.psi,self.own.u))
            elif (a.target_id not in self.risk_engine._encounter_memory and
                  a.risk_score<.08 and a.range_m>2*a.sda_m):
                self.encounter_references.pop(a.target_id,None)
        involved={a.target_id for a in snapshot.assessments if a.risk_score>=.12}
        stand_on_intervention = any(
            (not a.own_give_way) and
            a.risk_score >= DEFAULT_RULES.stand_on_intervention_score
            for a in snapshot.assessments
        )
        should_start=(snapshot.emergency or stand_on_intervention or
                      any(a.own_give_way and a.risk_score>=DEFAULT_RULES.activation_score
                          for a in snapshot.assessments))
        if should_start and not self.maneuver_active:
            # A new encounter interrupts an unfinished return-to-route phase.
            self.route_recovery_active=False
            self._route_recovery_stable_steps=0
            # A new avoidance event invalidates the previous return-to-route
            # event; the next recovery must satisfy the full corridor/alignment
            # stability test again.
            self.metrics.route_returned=False
            self.maneuver_active=True
            self.last_maneuver_start_time_s=float(self.metrics.elapsed_s)
            self.avoidance_started=True
            self.reference_heading=self.own.psi
            self.reference_speed=self.own.u
            self._force_policy_replan()
        if self.maneuver_active:
            new_targets=involved-self.maneuver_target_ids
            if new_targets:
                self.maneuver_target_ids.update(new_targets)
                self._force_policy_replan()
            by_id={a.target_id:a for a in snapshot.assessments}
            clear=all(tid not in by_id or (by_id[tid].tcpa_s < -60 and
                by_id[tid].range_m>1.5*1.274*by_id[tid].sda_m and by_id[tid].risk_score<.08)
                for tid in self.maneuver_target_ids)
            if self.maneuver_target_ids and clear:
                self.maneuver_active=False
                self.maneuver_target_ids.clear()
                self.reference_heading=None
                self.reference_speed=None
                self.route_recovery_active=True
                self._route_recovery_stable_steps=0
                self._force_policy_replan()
        self.phase=phase_for(snapshot,self.maneuver_active)

    @property
    def ppo_action_authorized(self):
        """PPO may choose tactical actions only during an active encounter.

        Normal monitoring and route recovery remain deterministic MPC/rule
        control phases.  This prevents a learned policy from initiating an
        untriggered manoeuvre or overriding the route-return objective.
        """
        return bool(self.maneuver_active and not self.route_recovery_active
                    and not self.odd_status.degraded)

    @property
    def policy_action_authorized(self):
        return self.ppo_action_authorized

    @property
    def policy_gate_reason(self):
        if self.route_recovery_active:
            return 'route_recovery_uses_rule_mpc'
        if self.metrics.route_returned and not self.maneuver_active:
            return 'post_recovery_cruise_uses_rule_mpc'
        if self.maneuver_active:
            return 'active_encounter_window'
        return 'monitor_phase_waits_for_rule_trigger'

    def _state_key(self):
        return (self.metrics.elapsed_s,tuple(vars(self.own).values()),
                tuple((t.target_id,t.active,tuple(vars(t.state).values())) for t in self.targets),
                repr(self.obstacles),repr(self.environment),self.use_rule_mask,self.route_index)

    def prepare_decision(self):
        key=self._state_key()
        if key==self._prepared_key:
            return self._prepared
        self.last_snapshot=self.risk_engine.assess(self.own,self.config,self.targets,
                                                  elapsed_s=self.metrics.elapsed_s,
                                                  environment=self.environment)
        self.odd_status=self.odd_monitor.evaluate(self.environment)
        self._update_encounter()
        started=time.perf_counter()
        self.candidates=self.planner.generate_candidates(
            self.own,self.targets,self.last_snapshot,self.route_x,self.route_y,
            self.dt,self.obstacles,self.environment,self.metrics.elapsed_s,
            (self.route_start_x,self.route_start_y),self.reference_heading,self.reference_speed,
            self.use_rule_mask,self.encounter_references,
            route_recovery=self.route_recovery_active,
            cruise_speed_mps=self.cruise_speed_mps,
            lock_speed_to_cruise=self.cruise_speed_locked)
        self.metrics.planning_times_s.append(time.perf_counter()-started)
        baseline=self.planner.select_candidate(self.candidates)
        self.baseline_action=baseline.candidate_id
        mask=[c.feasible for c in self.candidates]
        self.decision_infeasible=not any(mask)
        if self.decision_infeasible:
            # Recovery slot is available to keep a valid categorical distribution;
            # its trajectory remains infeasible, explicitly logged and penalized.
            mask[self.baseline_action]=True
        if not self.policy_action_authorized:
            # PPO must not change the course during passive monitoring, must not
            # start a manoeuvre before the common rule/risk trigger, must
            # never take over the dedicated route-recovery controller, and must
            # defer to the deterministic controller in a degraded ODD. The MPC
            # baseline remains free to avoid static hazards.
            mask=[i==self.baseline_action for i in range(len(self.candidates))]
        context=dict(maneuver_active=self.maneuver_active,reference_heading=self.candidates[0].course_cmd_rad,
                     reference_speed=self.candidates[0].speed_cmd_mps,elapsed_s=self.metrics.elapsed_s,
                     phase=self.phase,last_action=self.last_action,
                     intervention=self.last_intervention,odd_degraded=self.odd_status.degraded)
        obs=build_observation(self.own,self.config,self.targets,self.last_snapshot,
            self.route_x,self.route_y,self.route_start_x,self.route_start_y,
            context=context,candidates=self.candidates)
        self._prepared=(obs,mask)
        self._prepared_key=self._state_key()
        return self._prepared

    def action_mask(self,snapshot=None):
        return list(self.prepare_decision()[1])

    def observation(self):
        return list(self.prepare_decision()[0])

    def _update_route_recovery(self):
        """Finish recovery only after the ship is back on and aligned with the route."""
        if not self.route_recovery_active or self.maneuver_active:
            return False
        aligned = self.route_heading_error_deg <= math.degrees(
            self.route_recovery_heading_tolerance_rad)
        within = self.is_within_route_recovery_corridor()
        speed_aligned = self.cruise_speed_error_mps <= self.route_recovery_speed_tolerance_mps
        if within and aligned and speed_aligned:
            self._route_recovery_stable_steps += 1
        else:
            self._route_recovery_stable_steps = 0
        if self._route_recovery_stable_steps >= self.route_recovery_stable_required:
            self.route_recovery_active = False
            self.metrics.route_returned = True
            return True
        return False

    @property
    def navigation_stage(self):
        if self.maneuver_active:
            return '避碰机动'
        if self.route_recovery_active:
            return '航线回归'
        if self.metrics.route_returned:
            return '已回归航线'
        return '正常航行'

    def _check_destination(self):
        progress=(self.own.x-self.route_start_x)*self.route_ux+(self.own.y-self.route_start_y)*self.route_uy
        if progress < self.route_length-min(150.,self.route_length*.05):
            return False
        # Do not end the episode or advance to the next route leg while an
        # avoidance/recovery state is still active. Recovery completion already
        # enforces the 0.1L corridor, heading alignment and dwell-time checks.
        if self.maneuver_active or self.route_recovery_active:
            return False
        if not self.is_within_route_recovery_corridor():
            return False
        if self.route_index+1 < len(self.scenario.route):
            self.route_start_x,self.route_start_y=self.route_x,self.route_y
            self.route_index+=1
            self._set_route(self.scenario.route[self.route_index])
            self._next_policy_update_s=self.metrics.elapsed_s
            self._prepared_key=None
            return False
        self.metrics.destination_reached=True
        self.running=False
        return True

    def step(self,action_override=None):
        self.odd_status=self.odd_monitor.evaluate(self.environment)
        if self.odd_status.takeover_required:
            self.running=False
            self.logs.append(dict(time_s=self.metrics.elapsed_s,event='ODD degraded: operator takeover required',
                                  reasons=self.odd_status.reasons))
            return
        if self._check_destination():
            return
        self.last_snapshot=self.risk_engine.assess(self.own,self.config,self.targets,elapsed_s=self.metrics.elapsed_s,environment=self.environment)
        self._update_encounter()
        due=self.metrics.elapsed_s>=self._next_policy_update_s or self._held_course is None
        self.last_intervention=False
        self.intervention_reasons=()
        if due or action_override is not None:
            obs,mask=self.prepare_decision()
            self.last_decision_policy_authorized=self.ppo_action_authorized
            self.last_decision_gate_reason=self.policy_gate_reason
            if action_override is not None:
                action=int(action_override)
                if not 0<=action<ACTION_COUNT or not mask[action]:
                    raise ValueError(f'PPO action {action} is disallowed by action mask {mask}')
                self.last_policy_source='training_override'
            elif self.odd_status.degraded:
                action=self.baseline_action
                self.last_policy_source='rule_baseline_odd_degraded'
            elif not self.policy_action_authorized:
                action=self.baseline_action
                self.last_policy_source=(
                    'rule_mpc_route_recovery' if self.route_recovery_active else
                    ('rule_mpc_post_recovery' if self.metrics.route_returned else 'rule_mpc_monitor')
                )
            elif not getattr(self.policy,'available',False):
                action=self.baseline_action
                self.last_policy_source='rule_mpc_active_encounter'
            else:
                action=self.policy.predict(obs,self.last_snapshot,mask)
                self.last_policy_source='ppo_active_encounter'
            self.last_action=action
            self.last_executed_action=action
            if not 0<=action<ACTION_COUNT or not mask[action]:
                self.last_executed_action=self.baseline_action
                self.last_intervention=True
                self.intervention_reasons=('policy:invalid_candidate',)
            plan=self.candidates[self.last_executed_action]
            self._held_course=plan.course_cmd_rad
            self._held_speed=plan.speed_cmd_mps
            self.metrics.decision_count+=1
            self.metrics.infeasible_decision_count+=int(self.decision_infeasible)
            self.metrics.singleton_decision_count+=int(sum(mask)==1)
            if self.decision_infeasible:
                self.last_intervention=True
                self.intervention_reasons+=('planner:no_feasible_candidate',)+plan.reasons
            self._next_policy_update_s=self.metrics.elapsed_s+self.policy_update_period_s
        else:
            plan=copy.copy(self.last_plan)
            plan.rudder_cmd_rad=self.planner._course_controller(
                self.own, self._held_course, self.environment, self.metrics.elapsed_s
            )
            plan.speed_cmd_mps=self._held_speed
        reasons=check_control(self.own,self.config,self.dynamics,plan.rudder_cmd_rad,
            plan.speed_cmd_mps,self.dt,self.targets,self.last_snapshot,self.obstacles,
            self.environment,self.metrics.elapsed_s) if self.use_safety_monitor else ()
        if reasons:
            self.prepare_decision()
            plan=self.planner.select_candidate(self.candidates)
            self.last_executed_action=plan.candidate_id
            self._held_course,self._held_speed=plan.course_cmd_rad,plan.speed_cmd_mps
            self.last_intervention=True
            self.intervention_reasons+=reasons
        self.planner.commit_candidate(plan)
        self.metrics.intervention_count+=int(self.last_intervention)
        previous=self.own.copy()
        previous_targets={t.target_id:(t.state.x,t.state.y) for t in self.targets if t.active}
        self.own=self.dynamics.step(self.own,plan.rudder_cmd_rad,plan.speed_cmd_mps,self.dt,
                                     self.environment,self.metrics.elapsed_s)
        for target in self.targets:
            target.update(self.dt,self.environment)
            self.target_tracks[target.target_id].append((target.state.x,target.state.y))
        self.metrics.elapsed_s+=self.dt
        self.own_track.append((self.own.x,self.own.y))
        self._record_contacts(previous,previous_targets)
        if not plan.safe:
            self.metrics.unsafe_prediction_count+=1
        self.metrics.heading_command_changes+=int(self.last_plan is not None and
            abs(plan.rudder_cmd_rad-self.last_plan.rudder_cmd_rad)>math.radians(3))
        self.last_plan=plan
        self.maneuver_intent=plan.chosen_intent
        self.last_snapshot=self.risk_engine.assess(self.own,self.config,self.targets,elapsed_s=self.metrics.elapsed_s,environment=self.environment)
        self._update_encounter()
        self._update_route_recovery()
        self._check_destination()
        self._prepared_key=None
        critical=max(self.last_snapshot.assessments,key=lambda a:a.risk_score,default=None)
        self.logs.append(dict(time_s=self.metrics.elapsed_s,x_m=self.own.x,y_m=self.own.y,
            heading_deg=math.degrees(self.own.psi)%360,speed_mps=self.own.u,rudder_deg=math.degrees(self.own.delta),
            intent=plan.chosen_intent,ppo_action=ACTION_NAMES[self.last_action],
            policy_source=self.last_policy_source,
            policy_action_authorized=self.last_decision_policy_authorized,
            policy_gate_reason=self.last_decision_gate_reason,
            maneuver_start_time_s=self.last_maneuver_start_time_s,
            proposed_action=self.last_action,executed_action=self.last_executed_action,
            plan_safe=plan.safe,planning_status=plan.status,phase=self.phase,
            intervention=self.last_intervention,intervention_reasons=self.intervention_reasons,
            feasible_action_count=sum(c.feasible for c in self.candidates),
            candidate_reasons=[list(c.reasons) for c in self.candidates],
            executed_rule_reasons=list(plan.rule_reasons),
            maneuver_active=self.maneuver_active,route_recovery_active=self.route_recovery_active,
            navigation_stage=self.navigation_stage,
            route_cross_track_m=self.route_cross_track_distance_m(),
            route_heading_error_deg=self.route_heading_error_deg,
            route_recovery_stable_steps=self._route_recovery_stable_steps,
            cruise_speed_target_mps=self.cruise_speed_mps,
            cruise_speed_error_mps=self.cruise_speed_error_mps,
            cruise_speed_locked=self.cruise_speed_locked,
            target_course_deg=math.degrees(plan.course_cmd_rad)%360,
            minimum_predicted_clearance_m=plan.minimum_clearance_m,focus_target=self.last_snapshot.focus_target_id,
            risk=critical.risk_level if critical else 'No risk',odd_degraded=self.odd_status.degraded,
            dcpa_m=critical.dcpa_m if critical else None,tcpa_s=critical.tcpa_s if critical else None,
            dcrc_m=critical.dcrc_m if critical else None,dcqs_m=critical.dcqs_m if critical else None,
            pass_side=critical.pass_side if critical else None,preferred_turn_deg=critical.preferred_turn_deg if critical else None,
            sda_m=critical.sda_m if critical else None,
            sda_pass_position=critical.sda_pass_position if critical else None,
            sda_violation_count=self.metrics.sda_violation_count,
            sda_violation_targets=list(self.metrics.sda_violation_targets),
            minimum_actual_separation_m=self.metrics.min_separation_m,
            minimum_sda_margin_m=self.metrics.minimum_domain_clearance_m))

    def _record_contacts(self,previous,previous_targets):
        a0,a1=(previous.x,previous.y),(self.own.x,self.own.y)
        contacts=set()
        assessments={a.target_id:a for a in self.last_snapshot.assessments}
        intrusion=False
        for target in self.targets:
            if not target.active:
                continue
            distance=swept_separation(a0,a1,previous_targets[target.target_id],(target.state.x,target.state.y))
            self.metrics.min_separation_m=min(self.metrics.min_separation_m,distance)
            hull_radius=.5*math.hypot(self.config.length_m,self.config.beam_m)+.5*math.hypot(target.length_m,target.beam_m)
            if distance<hull_radius:
                contacts.add(target.target_id)
            self.metrics.target_min_separation_m[target.target_id] = min(
                self.metrics.target_min_separation_m.get(target.target_id, float("inf")), distance
            )
            assessment = assessments.get(target.target_id)
            if assessment is not None and assessment.sda_m > 0.0:
                # The measured swept centre-to-centre separation is compared
                # against the exact SDA radius used by the displayed target ring.
                clearance = distance - assessment.sda_m
                self.metrics.minimum_domain_clearance_m = min(
                    self.metrics.minimum_domain_clearance_m, clearance
                )
                self.metrics.target_min_sda_margin_m[target.target_id] = min(
                    self.metrics.target_min_sda_margin_m.get(target.target_id, float("inf")),
                    clearance,
                )
                if clearance < -1e-6:
                    intrusion = True
                    if target.target_id not in self._sda_violation_ids:
                        self._sda_violation_ids.add(target.target_id)
                        self.metrics.sda_violation_count += 1
                        self.metrics.sda_violation_targets.append(target.target_id)
        self.metrics.collision_count+=len(contacts-self._contact_ids)
        self.metrics.collision_duration_s+=self.dt*bool(contacts)
        self._contact_ids=contacts
        self.metrics.domain_intrusion_s+=self.dt*intrusion
        obstacle_contact_threshold=.5*math.hypot(self.config.length_m,self.config.beam_m)
        obstacles={o.obstacle_id for o in self.obstacles if
            segment_obstacle_distance(a0,a1,o,early_exit_distance_m=obstacle_contact_threshold)
            < obstacle_contact_threshold}
        self.metrics.obstacle_collision_count+=len(obstacles-self._obstacle_contact_ids)
        self.metrics.obstacle_contact_duration_s+=self.dt*bool(obstacles)
        self._obstacle_contact_ids=obstacles

    @property
    def terminal(self):
        return bool(self.metrics.destination_reached or self.metrics.collision_count or
                    self.metrics.obstacle_collision_count or self.odd_status.takeover_required)

    def advance_decision(self,action):
        from .rewards import compute_reward
        start=self.metrics.elapsed_s
        reward=0.
        components={}
        intervention=False
        while self.metrics.elapsed_s-start < self.policy_update_period_s-1e-9:
            previous=self.own.copy()
            collision=self.metrics.collision_count
            obstacle=self.metrics.obstacle_collision_count
            returned=self.metrics.route_returned
            reached=self.metrics.destination_reached
            before=self.metrics.elapsed_s
            self.step(action_override=action if before==start else None)
            intervention |= self.last_intervention
            dt=self.metrics.elapsed_s-before
            value,parts=compute_reward(self,previous,previous.delta,collision,obstacle,
                                        returned,reached,dt_s=dt)
            discount=.99**((before-start)/DECISION_PERIOD_S)
            reward+=discount*value
            for key,value in parts.items():
                components[key]=components.get(key,0.)+discount*value
            if self.terminal or dt<=0:
                break
        return float(reward),self.terminal,dict(elapsed_s=self.metrics.elapsed_s-start,
            reward_components=components,intervention=intervention)

    def clear_tracks(self):
        self.own_track[:]=[(self.own.x,self.own.y)]
        for target in self.targets:
            self.target_tracks[target.target_id]=[(target.state.x,target.state.y)]

    def reset(self,scenario=None):
        checkpoint=getattr(self.policy,'checkpoint',None)
        self.__init__(self.scenario if scenario is None else scenario,checkpoint)

    def export_log(self,path):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(self.logs,ensure_ascii=False,indent=2),encoding='utf-8')
        return path
