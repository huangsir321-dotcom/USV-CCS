from __future__ import annotations
import math
from .models import ShipConfig, ShipState, wrap_angle


class Simplified3DOFDynamics:
    """3-DOF manoeuvring surrogate with rudder, yaw and surge lags.

    This is intentionally a tunable research model for software-in-the-loop tests.
    It is not a calibrated MMG model and must not be used for real-vessel control.
    The interface is kept compatible with a later MMG implementation.
    """
    def __init__(self, config: ShipConfig):
        self.config = config

    @staticmethod
    def yaw_disturbance_acceleration(state: ShipState, environment=None, elapsed_s: float = 0.0) -> float:
        """Small environmental yaw-acceleration disturbance used by the model.

        This is an illustrative disturbance, not a calibrated wind-load model.
        Coefficients are deliberately bounded so even the slowest large-vessel
        configuration retains enough rudder authority for heading keeping.
        """
        if environment is None:
            return 0.0
        wind_rel = environment.wind_dir_rad - state.psi
        wind = 0.0000025 * environment.wind_speed_mps * math.sin(wind_rel)
        wave = 0.00001 * environment.wave_level * math.sin(
            0.17 * elapsed_s + 0.001 * state.x
        )
        return wind + wave

    def step(self, state: ShipState, rudder_cmd_rad: float, speed_cmd_mps: float, dt: float, environment=None, elapsed_s: float = 0.0) -> ShipState:
        remaining = max(0.0, float(dt))
        current = state.copy()
        while remaining > 1e-9:
            h = min(1.0, remaining)
            current = self._step_once(current, rudder_cmd_rad, speed_cmd_mps, h, environment, elapsed_s)
            remaining -= h
            elapsed_s += h
        return current

    def _step_once(self, state, rudder_cmd_rad, speed_cmd_mps, dt, environment, elapsed_s):
        c = self.config
        s = state.copy()
        max_delta = math.radians(c.max_rudder_deg)
        max_rate = math.radians(c.rudder_rate_deg_s)
        rudder_cmd_rad = max(-max_delta, min(max_delta, rudder_cmd_rad))

        desired_delta_rate = (rudder_cmd_rad - s.delta) / max(c.rudder_tau_s, 1e-3)
        desired_delta_rate = max(-max_rate, min(max_rate, desired_delta_rate))
        s.delta += desired_delta_rate * dt
        s.delta = max(-max_delta, min(max_delta, s.delta))

        # First-order Nomoto-like yaw response, with a speed-dependent cap.
        yaw_target = c.yaw_gain * s.delta * max(0.25, s.u / 7.0)
        max_r = math.radians(c.max_yaw_rate_deg_s)
        yaw_target = max(-max_r, min(max_r, yaw_target))
        r_dot = (yaw_target - s.r) / max(c.yaw_tau_s, 1e-3)
        r_dot += self.yaw_disturbance_acceleration(s, environment, elapsed_s)
        s.r += r_dot * dt
        s.r = max(-max_r, min(max_r, s.r))

        # Surge response is delayed and acceleration-limited.
        speed_cmd_mps = max(c.min_speed_mps, min(c.max_speed_mps, speed_cmd_mps))
        u_dot = (speed_cmd_mps - s.u) / max(c.speed_tau_s, 1e-3)
        u_dot = max(-c.max_decel_mps2, min(c.max_accel_mps2, u_dot))
        # Clamp commands, not an externally supplied state: out-of-range initial
        # speeds must return gradually without violating acceleration limits.
        next_u = s.u + u_dot * dt
        s.u = max(min(s.u, speed_cmd_mps), min(max(s.u, speed_cmd_mps), next_u))

        # Approximate sway response from yaw rate; this provides a small sideslip term.
        sway_target = -0.12 * s.u * s.r
        s.v += (sway_target - s.v) * min(1.0, dt / max(3.0, 0.06 * c.length_m))

        s.psi = (s.psi + s.r * dt) % (2.0 * math.pi)
        x_dot = s.u * math.sin(s.psi) + s.v * math.cos(s.psi)
        y_dot = s.u * math.cos(s.psi) - s.v * math.sin(s.psi)
        if environment is not None:
            x_dot += environment.current_x_mps
            y_dot += environment.current_y_mps
        s.x += x_dot * dt
        s.y += y_dot * dt
        s.n = speed_cmd_mps / max(c.max_speed_mps, 1e-3)
        return s

    def simulate(self, state: ShipState, rudder_cmd_rad: float, speed_cmd_mps: float, dt: float, steps: int, environment=None):
        s = state.copy()
        path = [(s.x, s.y)]
        for _ in range(steps):
            s = self.step(s, rudder_cmd_rad, speed_cmd_mps, dt, environment, _ * dt)
            path.append((s.x, s.y))
        return s, path
