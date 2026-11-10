from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import math


def wrap_angle(angle: float) -> float:
    """Wrap radians to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class ShipConfig:
    name: str = "Own ship"
    length_m: float = 200.0
    beam_m: float = 32.0
    max_speed_mps: float = 9.0
    min_speed_mps: float = 2.0
    nominal_speed_mps: float = 7.0
    max_rudder_deg: float = 35.0
    rudder_rate_deg_s: float = 2.0
    rudder_tau_s: float = 5.0
    yaw_tau_s: float = 32.0
    yaw_gain: float = 0.075
    speed_tau_s: float = 35.0
    max_accel_mps2: float = 0.08
    max_decel_mps2: float = 0.12
    max_yaw_rate_deg_s: float = 1.8

    @classmethod
    def for_length(cls, length_m: float, name: str = "Own ship") -> "ShipConfig":
        """Create a conservative illustrative configuration, not a vessel-specific calibration."""
        L = max(50.0, min(500.0, float(length_m)))
        scale = L / 200.0
        return cls(
            name=name,
            length_m=L,
            beam_m=max(8.0, 0.16 * L),
            max_speed_mps=9.0 if L < 250 else 8.0,
            min_speed_mps=2.0,
            nominal_speed_mps=7.0 if L <= 300 else 6.5,
            max_rudder_deg=35.0,
            rudder_rate_deg_s=max(0.8, min(3.0, 1.5 / math.sqrt(scale))),
            rudder_tau_s=5.0 + 2.0 * scale,
            yaw_tau_s=32.0 * scale,
            yaw_gain=0.075 / scale,
            speed_tau_s=35.0 * scale,
            max_accel_mps2=0.08 / math.sqrt(scale),
            max_decel_mps2=0.12 / math.sqrt(scale),
            max_yaw_rate_deg_s=max(0.18, min(1.8, 0.6 / scale)),
        )


@dataclass
class ShipState:
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0            # radians; 0=north, positive clockwise
    u: float = 7.0              # surge speed, m/s
    v: float = 0.0              # sway speed, m/s
    r: float = 0.0              # yaw rate, rad/s
    delta: float = 0.0          # actual rudder angle, radians
    n: float = 0.0              # normalized propulsion state

    def copy(self) -> "ShipState":
        return ShipState(self.x, self.y, self.psi, self.u, self.v, self.r, self.delta, self.n)

    @property
    def speed(self) -> float:
        return math.hypot(self.u, self.v)


@dataclass
class TargetShip:
    target_id: str
    state: ShipState
    length_m: float = 100.0
    beam_m: float = 16.0
    ship_type: str = "other"
    behavior: str = "constant_velocity"
    turn_start_s: Optional[float] = None
    turn_rate_deg_s: float = 0.0
    speed_change_start_s: Optional[float] = None
    speed_after_mps: Optional[float] = None
    elapsed_s: float = 0.0
    active: bool = True

    def velocity(self):
        return (self.state.u * math.sin(self.state.psi), self.state.u * math.cos(self.state.psi))

    def _advance(self, state, elapsed, dt, environment=None, oracle=True):
        """Shared <=1 s integrator; delayed actions apply only after their threshold."""
        remaining = max(0.0, float(dt))
        while remaining > 1e-9:
            h = min(1.0, remaining)
            # Keep one-second grid across requested prediction samples.
            h = min(h, 1.0 - elapsed % 1.0)
            end = elapsed + h
            if oracle:
                if self.behavior == "turn_after_delay" and self.turn_start_s is not None:
                    active = max(0.0, end - max(elapsed, self.turn_start_s))
                    state.psi = (state.psi + math.radians(self.turn_rate_deg_s) * active) % (2 * math.pi)
                if self.speed_change_start_s is not None and self.speed_after_mps is not None:
                    active = max(0.0, end - max(elapsed, self.speed_change_start_s))
                    change = 0.04 * active
                    state.u += max(-change, min(change, self.speed_after_mps - state.u))
            vx = state.u * math.sin(state.psi) + state.v * math.cos(state.psi)
            vy = state.u * math.cos(state.psi) - state.v * math.sin(state.psi)
            if environment is not None:
                vx += environment.current_x_mps
                vy += environment.current_y_mps
            state.x += vx * h
            state.y += vy * h
            elapsed = end
            remaining -= h
        return elapsed

    def update(self, dt: float, environment=None):
        if self.active:
            self.elapsed_s = self._advance(self.state, self.elapsed_s, dt, environment)

    def prediction_path(self, times, environment=None, oracle=False):
        """Positions at requested future times, in input order, without changing state.

        Default prediction uses observed velocity only. Oracle mode explicitly uses
        the scenario script and shares update's numerical integration.
        """
        times = [max(0.0, float(t)) for t in times]
        result = [None] * len(times)
        state = self.state.copy()
        elapsed = self.elapsed_s
        integrated_future = 0.0
        for index in sorted(range(len(times)), key=times.__getitem__):
            future = times[index]
            # Commit only fixed grid boundaries. A fractional query uses a copy,
            # so adding samples cannot perturb later integration states.
            if self.active:
                while True:
                    h = 1.0 - elapsed % 1.0
                    if integrated_future + h > future + 1e-9:
                        break
                    elapsed = self._advance(state, elapsed, h, environment, oracle)
                    integrated_future += h
                sampled = state.copy()
                self._advance(sampled, elapsed, future - integrated_future, environment, oracle)
            else:
                sampled = state
            result[index] = (sampled.x, sampled.y)
        return result

    def predict_position(self, future_s: float, environment=None, oracle=False):
        return self.prediction_path([future_s], environment, oracle)[0]


@dataclass
class StaticObstacle:
    obstacle_id: str
    vertices: List[tuple]
    # Circle obstacles retain their semantic geometry for rendering and editing.
    # Their dense boundary vertices are also consumed by the existing polygon MPC.
    shape_type: str = "polygon"
    center: Optional[tuple] = None
    radius_m: Optional[float] = None

    def copy(self) -> "StaticObstacle":
        return StaticObstacle(
            self.obstacle_id, list(self.vertices), self.shape_type,
            tuple(self.center) if self.center is not None else None,
            float(self.radius_m) if self.radius_m is not None else None,
        )


@dataclass
class EnvironmentConfig:
    current_x_mps: float = 0.0
    current_y_mps: float = 0.0
    wind_speed_mps: float = 3.0
    wind_dir_rad: float = 0.0
    wave_level: float = 0.5
    visibility_nm: float = 5.0
    sensor_valid: bool = True
    control_valid: bool = True
    max_normal_wind_mps: float = 10.0
    max_normal_wave_level: float = 3.0
    min_normal_visibility_nm: float = 1.0


@dataclass
class Scenario:
    name: str
    own_config: ShipConfig
    own_state: ShipState
    targets: List[TargetShip]
    route: List[tuple] = field(default_factory=lambda: [(0.0, 12000.0)])
    description: str = ""
    dt_s: float = 2.0
    obstacles: List[StaticObstacle] = field(default_factory=list)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    metadata: dict = field(default_factory=dict)


@dataclass
class EncounterAssessment:
    target_id: str
    encounter: str
    own_give_way: bool
    rule: str
    relative_bearing_deg: float
    range_m: float
    dcpa_m: float
    tcpa_s: float
    sda_m: float
    risk_level: str
    risk_score: float
    notes: str = ""
    # Tactical guidance derived from relative geometry and an Appendix-A-inspired
    # critical-distance estimate. These remain engineering approximations.
    pass_side: str = "unspecified"
    preferred_turn_deg: float = 0.0
    dcrc_m: float = math.inf
    dcqs_m: float = math.inf
    time_to_dcrc_s: float = math.inf
    time_to_dcqs_s: float = math.inf
    sda_pass_position: str = "generic"
    sda_estimated: bool = False


@dataclass
class RiskSnapshot:
    assessments: List[EncounterAssessment]
    focus_target_id: Optional[str]
    max_risk_score: float
    emergency: bool


@dataclass
class PlannerResult:
    rudder_cmd_rad: float
    speed_cmd_mps: float
    predicted_path: List[tuple]
    score: float
    safe: bool
    chosen_intent: str
    minimum_clearance_m: float
    message: str


@dataclass
class SimulationMetrics:
    elapsed_s: float = 0.0
    min_separation_m: float = float("inf")
    collision_count: int = 0
    obstacle_collision_count: int = 0
    unsafe_prediction_count: int = 0
    heading_command_changes: int = 0
    last_heading_cmd: Optional[float] = None
    route_returned: bool = False
    destination_reached: bool = False
    decision_count: int = 0
    intervention_count: int = 0
    infeasible_decision_count: int = 0
    singleton_decision_count: int = 0
    domain_intrusion_s: float = 0.0
    collision_duration_s: float = 0.0
    obstacle_contact_duration_s: float = 0.0
    minimum_domain_clearance_m: float = float('inf')
    sda_violation_count: int = 0
    sda_violation_targets: List[str] = field(default_factory=list)
    target_min_separation_m: dict = field(default_factory=dict)
    target_min_sda_margin_m: dict = field(default_factory=dict)
    planning_times_s: List[float] = field(default_factory=list)
