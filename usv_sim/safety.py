"""Short-horizon execution monitor, independent of the candidate scoring function.

Uses latest observed target velocity and one-second swept checks. Shares the
simulation model, so this is not a formal independent redundancy guarantee.
"""
from .geometry import swept_separation
from .planner import segment_obstacle_distance


def check_control(own, config, dynamics, rudder, speed, dt, targets, snapshot,
                  obstacles, environment, elapsed_s):
    state = own.copy()
    assessments = {a.target_id:a for a in snapshot.assessments}
    active = [t for t in targets if t.active]
    positions = {t.target_id:t.predict_position(0,environment) for t in active}
    reasons = []
    elapsed = 0.
    while elapsed < dt-1e-9:
        h = min(1.,dt-elapsed)
        before = (state.x,state.y)
        state = dynamics.step(state,rudder,speed,h,environment,elapsed_s+elapsed)
        after = (state.x,state.y)
        for t in active:
            next_pos=t.predict_position(elapsed+h,environment)
            a=assessments.get(t.target_id)
            required=a.sda_m if a else max(200.,2*config.length_m)
            if swept_separation(before,after,positions[t.target_id],next_pos) < required:
                reasons.append('monitor:SDA:'+t.target_id)
            positions[t.target_id]=next_pos
        obstacle_clearance = .5*config.length_m+100.0
        for obstacle in obstacles:
            # The AABB lower bound rejects distant polygons in constant time;
            # exact polygon edges or analytic circle geometry are evaluated only
            # when the segment may enter the safety-clearance band.
            distance=segment_obstacle_distance(
                before, after, obstacle, early_exit_distance_m=obstacle_clearance
            )
            if distance < obstacle_clearance:
                reasons.append('monitor:obstacle:'+obstacle.obstacle_id)
        elapsed += h
    return tuple(dict.fromkeys(reasons))
