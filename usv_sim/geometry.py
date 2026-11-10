import math
from .models import wrap_angle


def velocity_from_state(state):
    return (state.u * math.sin(state.psi) + state.v * math.cos(state.psi),
            state.u * math.cos(state.psi) - state.v * math.sin(state.psi))


def range_bearing(own, target):
    dx = target.x - own.x
    dy = target.y - own.y
    return math.hypot(dx, dy), math.atan2(dx, dy) % (2 * math.pi)


def relative_bearing_deg(own, target):
    _, bearing = range_bearing(own, target)
    return math.degrees(wrap_angle(bearing - own.psi))


def cpa(own, target):
    rx, ry = target.x - own.x, target.y - own.y
    ovx, ovy = velocity_from_state(own)
    tvx, tvy = velocity_from_state(target)
    vx, vy = tvx - ovx, tvy - ovy
    vv = vx * vx + vy * vy
    if vv < 1e-8:
        tcpa = 0.0
    else:
        tcpa = -(rx * vx + ry * vy) / vv
    tcpa = max(-1e6, min(1e6, tcpa))
    dcpa = math.hypot(rx + vx * tcpa, ry + vy * tcpa)
    return dcpa, tcpa, math.hypot(rx, ry)


def point_segment_distance(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    if den <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / den))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def cross_track_error(x, y, route_x=0.0):
    return x - route_x


def swept_separation(a0, a1, b0, b1):
    """Minimum separation for linearly moving points during one shared interval."""
    return point_segment_distance(0, 0, a0[0]-b0[0], a0[1]-b0[1], a1[0]-b1[0], a1[1]-b1[1])
