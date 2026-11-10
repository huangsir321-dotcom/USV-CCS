from dataclasses import dataclass


@dataclass
class ODDStatus:
    normal: bool
    degraded: bool
    takeover_required: bool
    reasons: list


class ODDMonitor:
    """Runtime-design-domain monitor for the simulation's declared operating envelope."""
    def evaluate(self, environment):
        reasons = []
        if not environment.sensor_valid:
            reasons.append("Perception system invalid")
        if not environment.control_valid:
            reasons.append("Control system invalid")
        if environment.visibility_nm < environment.min_normal_visibility_nm:
            reasons.append("Visibility below normal operating limit")
        if environment.wind_speed_mps > environment.max_normal_wind_mps:
            reasons.append("Wind speed above normal operating limit")
        if environment.wave_level > environment.max_normal_wave_level:
            reasons.append("Sea state above normal operating limit")
        return ODDStatus(
            normal=len(reasons) == 0,
            degraded=len(reasons) > 0,
            takeover_required=(not environment.control_valid or not environment.sensor_valid),
            reasons=reasons,
        )
