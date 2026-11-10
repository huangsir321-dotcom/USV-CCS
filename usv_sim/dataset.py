"""Loader for the user's 1T/2T/3T CSV encounter-case collections.

The source CSVs contain initial encounter states (not time-series tracks). Each row
represents one target vessel in the same encounter; therefore 1T/2T/3T files have
1/2/3 data rows respectively. Angles use nautical convention: 0 deg = north,
clockwise positive. Speeds are knots.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import ShipConfig, ShipState, TargetShip, Scenario, EnvironmentConfig

KNOT_TO_MPS = 0.514444
NM_TO_M = 1852.0
METERS_PER_DEG_LAT = 111_132.0
GROUP_DIRS = {"1T": "1t", "2T": "2t_0720", "3T": "3t"}
GROUP_COUNTS = {"1T": 138, "2T": 99, "3T": 31}


@dataclass(frozen=True)
class CaseSpec:
    group: str
    scene_id: str
    path: Path

    @property
    def key(self) -> str:
        return f"{self.group}/scene_{self.scene_id}.csv"


def default_dataset_root() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets" / "cases"


def list_cases(dataset_root: str | Path | None = None, group: str | None = None) -> list[CaseSpec]:
    root = Path(dataset_root) if dataset_root else default_dataset_root()
    groups = [group.upper()] if group else list(GROUP_DIRS)
    specs: list[CaseSpec] = []
    for g in groups:
        if g not in GROUP_DIRS:
            continue
        folder = root / GROUP_DIRS[g]
        if not folder.exists():
            continue
        for p in sorted(folder.glob("scene_*.csv")):
            scene_id = p.stem.replace("scene_", "")
            specs.append(CaseSpec(g, scene_id, p))
    return specs


def _read_rows(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    text = None
    errors = []
    for encoding in ("utf-8-sig", "gb18030", "gbk"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    if text is None:
        raise ValueError(f"Cannot decode case CSV {path}: {'; '.join(errors)}")
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError(f"Case CSV has no target rows: {path}")
    return rows


def _number(row: dict[str, str], key: str, default: float | None = None) -> float:
    value = (row.get(key) or "").strip()
    if not value:
        if default is None:
            raise ValueError(f"Missing required field '{key}'")
        return float(default)
    return float(value)


def _local_xy(lat: float, lon: float, origin_lat: float, origin_lon: float) -> tuple[float, float]:
    """Equirectangular local tangent approximation, suitable for these short encounters."""
    y = (lat - origin_lat) * METERS_PER_DEG_LAT
    x = (lon - origin_lon) * METERS_PER_DEG_LAT * math.cos(math.radians(origin_lat))
    return x, y


def _clean_token(value: str | None) -> str:
    return (value or "").strip()


def load_case(
    spec: CaseSpec | str | Path,
    own_length_m: float = 200.0,
    target_lengths_m: Iterable[float] | None = None,
    default_target_length_m: float = 100.0,
    dt_s: float = 2.0,
) -> Scenario:
    """Convert one uploaded source case to a simulation Scenario.

    Source case lat/lon, SOG/COG, target count and target initial conditions are
    preserved. The uploaded CSVs do not contain vessel dimensions or time histories;
    own/target lengths are supplied by UI settings or defaulted here.
    """
    if isinstance(spec, CaseSpec):
        path, group, scene_id = spec.path, spec.group, spec.scene_id
    else:
        path = Path(spec)
        group = path.parent.name.upper()
        scene_id = path.stem.replace("scene_", "")
    rows = _read_rows(path)
    r0 = rows[0]
    own_lat = _number(r0, "ownship_lat")
    own_lon = _number(r0, "ownship_lon")
    own_sog_kn = _number(r0, "ownship_sog(KN)")
    own_cog_deg = _number(r0, "ownship_cog(deg)") % 360.0
    own_speed = own_sog_kn * KNOT_TO_MPS

    own_cfg = ShipConfig.for_length(float(own_length_m), "Own ship")
    # The source cases include up to 20 kn. Keep the model limit above the initial speed.
    own_cfg.max_speed_mps = max(own_cfg.max_speed_mps, own_speed * 1.30, 12.0)
    own_cfg.nominal_speed_mps = min(own_cfg.max_speed_mps, own_speed)
    own_cfg.min_speed_mps = min(own_cfg.min_speed_mps, max(0.8, own_speed * 0.35))
    own = ShipState(x=0.0, y=0.0, psi=math.radians(own_cog_deg), u=own_speed)

    supplied_lengths = list(target_lengths_m or [])
    targets: list[TargetShip] = []
    meeting_labels: list[str] = []
    stage_labels: list[str] = []
    source_tcpa_min: list[float] = []
    source_dcpa_nm: list[float] = []
    for index, row in enumerate(rows):
        tid = f"TV{index + 1}"
        lat = _number(row, "target_lat")
        lon = _number(row, "target_lon")
        x, y = _local_xy(lat, lon, own_lat, own_lon)
        speed_mps = _number(row, "target_sog(KN)") * KNOT_TO_MPS
        course_deg = _number(row, "target_cog(deg)") % 360.0
        length = supplied_lengths[index] if index < len(supplied_lengths) else default_target_length_m
        length = max(50.0, min(500.0, float(length)))
        targets.append(TargetShip(
            target_id=tid,
            state=ShipState(x=x, y=y, psi=math.radians(course_deg), u=speed_mps),
            length_m=length,
            beam_m=max(8.0, 0.16 * length),
            ship_type="other",
            behavior="constant_velocity",
        ))
        meeting_labels.append(_clean_token(row.get("meeting_situation")))
        stage_labels.append(_clean_token(row.get("stage")))
        try:
            source_tcpa_min.append(float((row.get("TCPA(min)") or "").strip()))
        except (TypeError, ValueError):
            source_tcpa_min.append(0.0)
        try:
            source_dcpa_nm.append(float((row.get("DCPA(NM)") or "").strip()))
        except (TypeError, ValueError):
            source_dcpa_nm.append(0.0)

    # The source CSV does not provide a global route. Build a route long enough
    # that the generated episode does not terminate before the longest positive
    # source TCPA. Add 10 min after CPA for a pass-and-clear observation window.
    max_source_tcpa_s = max([v * 60.0 for v in source_tcpa_min if v > 0.0] + [0.0])
    route_distance = max(15.0 * NM_TO_M, own_speed * (max_source_tcpa_s + 600.0))
    route = [(route_distance * math.sin(own.psi), route_distance * math.cos(own.psi))]
    name = f"{group} - Scene {scene_id}"
    labels = "; ".join(f"TV{i+1}: {m or 'not specified'}" for i, m in enumerate(meeting_labels))
    description = f"Imported user case: {len(targets)} target(s). Encounter labels: {labels}"
    return Scenario(
        name=name,
        own_config=own_cfg,
        own_state=own,
        targets=targets,
        route=route,
        description=description,
        dt_s=float(dt_s),
        obstacles=[],
        environment=EnvironmentConfig(),
        metadata={
            "source_group": group,
            "scene_id": scene_id,
            "source_file": path.name,
            "ownship_stage": _clean_token(r0.get("stage")),
            "meeting_situations": meeting_labels,
            "stage_labels": stage_labels,
            "ownship_sog_kn": own_sog_kn,
            "ownship_cog_deg": own_cog_deg,
            "target_count": len(targets),
            "source_tcpa_min": source_tcpa_min,
            "source_dcpa_nm": source_dcpa_nm,
            "max_source_tcpa_s": max_source_tcpa_s,
        },
    )


def dataset_summary(dataset_root: str | Path | None = None) -> dict:
    return {g: len(list_cases(dataset_root, g)) for g in GROUP_DIRS}
