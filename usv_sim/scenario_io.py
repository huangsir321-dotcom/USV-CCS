from __future__ import annotations
import json
import math
from pathlib import Path
from .models import ShipConfig, ShipState, TargetShip, Scenario, StaticObstacle, EnvironmentConfig


def load_scenario_json(path):
    """Load a user-editable scenario JSON file; angles in JSON are degrees."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    own_data = data.get("own_ship", {})
    length = float(own_data.get("length_m", 200.0))
    cfg = ShipConfig.for_length(length, "OV")
    key_map = {
        "max_rudder_deg": "max_rudder_deg",
        "rudder_rate_deg_s": "rudder_rate_deg_s",
        "rudder_time_constant_s": "rudder_tau_s",
        "yaw_time_constant_s": "yaw_tau_s",
        "max_yaw_rate_deg_s": "max_yaw_rate_deg_s",
        "max_speed_mps": "max_speed_mps",
        "min_speed_mps": "min_speed_mps",
        "nominal_speed_mps": "nominal_speed_mps",
        "speed_time_constant_s": "speed_tau_s",
    }
    for source, dest in key_map.items():
        if source in own_data:
            setattr(cfg, dest, float(own_data[source]))
    own = ShipState(
        x=float(own_data.get("x_m", 0.0)),
        y=float(own_data.get("y_m", 0.0)),
        psi=math.radians(float(own_data.get("initial_heading_deg", 0.0))),
        u=float(own_data.get("initial_speed_mps", 7.0)),
    )
    targets = []
    for td in data.get("targets", []):
        targets.append(TargetShip(
            target_id=str(td.get("id", f"TV{len(targets)+1}")),
            state=ShipState(
                x=float(td.get("x_m", 0.0)), y=float(td.get("y_m", 0.0)),
                psi=math.radians(float(td.get("heading_deg", 0.0))),
                u=float(td.get("speed_mps", 0.0)),
            ),
            length_m=float(td.get("length_m", 100.0)),
            beam_m=float(td.get("beam_m", max(8.0, 0.16 * float(td.get("length_m", 100.0))))),
            ship_type=str(td.get("type", "other")),
            behavior=str(td.get("behavior", "constant_velocity")),
            turn_start_s=td.get("turn_start_s"),
            turn_rate_deg_s=float(td.get("turn_rate_deg_s", 0.0)),
            speed_change_start_s=td.get("speed_change_start_s"),
            speed_after_mps=td.get("speed_after_mps"),
        ))
    route = [tuple(map(float, p)) for p in data.get("route", [[0.0, 12000.0]])]
    envd = data.get("environment", {})
    environment = EnvironmentConfig(
        current_x_mps=float(envd.get("current_x_mps", 0.0)),
        current_y_mps=float(envd.get("current_y_mps", 0.0)),
        wind_speed_mps=float(envd.get("wind_speed_mps", 3.0)),
        wind_dir_rad=math.radians(float(envd.get("wind_dir_deg", 0.0))),
        wave_level=float(envd.get("wave_level", 0.5)),
        visibility_nm=float(envd.get("visibility_nm", 5.0)),
        sensor_valid=bool(envd.get("sensor_valid", True)),
        control_valid=bool(envd.get("control_valid", True)),
    )
    obstacles = [StaticObstacle(str(o.get("id", f"OBS{i+1}")), [tuple(map(float, p)) for p in o["vertices"]])
                 for i, o in enumerate(data.get("obstacles", []))]
    return Scenario(
        name=str(data.get("scenario_name", path.stem)), own_config=cfg, own_state=own,
        targets=targets, route=route, description=str(data.get("description", "JSON 自定义场景")),
        dt_s=float(data.get("dt_s", 2.0)), obstacles=obstacles, environment=environment,
        metadata={"source_group": "JSON", "scene_id": path.stem},
    )
