"""Physical-scale reward shared by interactive simulation and training; no torch."""
import math


def compute_reward(engine, previous_state, previous_delta, collision_before,
                   obstacle_collision_before, route_returned_before=False,
                   destination_before=False, dt_s=None):
    """Return total and named additive terms, without clipping collision events.

    Default weights per 10 s: progress +.08 * metres/(nominal speed*10s),
    cross-track -.10*min(3,|metres|/1500), rudder -.06*|delta|/max_delta,
    rudder rate -.10*|d_delta|/(max_rate*dt), yaw -.04*|r|/max_r,
    domain -.90*penetration/domain; rule violation -1.5; shield -.25; time -.02.
    Running costs scale dt/10; progress is displacement based. Newly observed
    collision costs -100, route return +3 and destination +10 once per event.
    Near-field domain penetration is included even when TCPA <= 0.
    """
    dt=max(0.,float(engine.dt if dt_s is None else dt_s)); scale=dt/10
    own=engine.own; cfg=engine.config
    progress=(own.x-previous_state.x)*engine.route_ux+(own.y-previous_state.y)*engine.route_uy
    lateral=abs((own.x-engine.route_start_x)*-engine.route_uy+(own.y-engine.route_start_y)*engine.route_ux)
    risks=[]
    for a in engine.last_snapshot.assessments:
        separation=min(a.range_m,a.dcpa_m) if a.tcpa_s>0 else a.range_m
        risks.append(max(0.,1-separation/max(a.sda_m,1)))
    plan=getattr(engine,'last_plan',None)
    violation=any(any(rule in str(reason) for rule in ('R14','R15','R17'))
                  for reason in getattr(plan,'rule_reasons',getattr(plan,'reasons',())))
    parts={
      'time':-.02*scale,
      'progress':.08*progress/max(cfg.nominal_speed_mps*10,1),
      'cross_track':-.10*min(3.,lateral/1500)*scale,
      'rudder':-.06*abs(own.delta)/max(math.radians(cfg.max_rudder_deg),.001)*scale,
      'rudder_rate':-.10*abs(own.delta-previous_delta)/max(math.radians(cfg.rudder_rate_deg_s)*dt,.001)*scale,
      'yaw':-.04*abs(own.r)/max(math.radians(cfg.max_yaw_rate_deg_s),.001)*scale,
      'domain':-.90*max(risks,default=0)*scale,
      'rule':-1.5*float(violation)*scale,
      'intervention':-.25*float(bool(getattr(engine,'last_intervention',False)))*scale,
      'collision':-100.*float(engine.metrics.collision_count>collision_before or engine.metrics.obstacle_collision_count>obstacle_collision_before),
      'route_return':3.*float(engine.metrics.route_returned and not route_returned_before),
      'destination':10.*float(engine.metrics.destination_reached and not destination_before)}
    return float(sum(parts.values())),parts
