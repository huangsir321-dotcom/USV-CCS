"""Safe Distance of Approach (SDA) examples from GB/T 46851-2025 Appendix B.

Appendix B is an informative example database. Its bulk-carrier tables give
SDA in nautical miles for own-ship lengths 100/200/300 m and target lengths
50/100/200/300 m. The function interpolates/extrapolates when the configured
lengths fall outside those tabulated points and reports that result as estimated.

The returned distance is in metres. It is the radius used by the simulator's
SDA boundary and the threshold for measured centre-to-centre encounter distance.
"""
from __future__ import annotations

OWN_LENGTHS = (100.0, 200.0, 300.0)
TARGET_LENGTHS = (50.0, 100.0, 200.0, 300.0)
NM_TO_M = 1852.0

# Table B.1: 100 m bulk carrier, Table B.2: 200 m, Table B.3: 300 m.
# Every array is ordered by target length [50, 100, 200, 300] m.
# "bow" / "stern" applies to crossing, beam and rear-quarter rows.
# Rows whose source table distinguishes port/starboard passing have equal
# values on both sides, so they are represented by "generic".
_TABLE = {
    100.0: {
        "head_on": {"generic": [0.34, 0.40, 0.50, 0.60]},
        "starboard_crossing": {"bow": [0.45, 0.52, 0.61, 0.73], "stern": [0.43, 0.47, 0.56, 0.70]},
        "port_crossing": {"bow": [0.43, 0.49, 0.58, 0.69], "stern": [0.42, 0.44, 0.52, 0.62]},
        "starboard_beam": {"bow": [0.42, 0.50, 0.60, 0.70], "stern": [0.38, 0.41, 0.50, 0.60]},
        "port_beam": {"bow": [0.44, 0.50, 0.59, 0.75], "stern": [0.38, 0.42, 0.52, 0.63]},
        "being_overtaken": {"generic": [0.33, 0.40, 0.50, 0.60]},
        "overtaking": {"generic": [0.36, 0.40, 0.50, 0.60]},
        "starboard_rear_overtaking": {"bow": [0.41, 0.50, 0.60, 0.70], "stern": [0.38, 0.41, 0.50, 0.60]},
        "port_rear_overtaking": {"bow": [0.41, 0.50, 0.60, 0.70], "stern": [0.38, 0.41, 0.50, 0.60]},
        "port_rear_being_overtaken": {"bow": [0.39, 0.42, 0.50, 0.62], "stern": [0.33, 0.37, 0.45, 0.57]},
        "starboard_rear_being_overtaken": {"bow": [0.39, 0.42, 0.50, 0.62], "stern": [0.33, 0.37, 0.45, 0.57]},
    },
    200.0: {
        "head_on": {"generic": [0.46, 0.55, 0.65, 0.74]},
        "starboard_crossing": {"bow": [0.66, 0.70, 0.85, 1.00], "stern": [0.61, 0.65, 0.80, 0.95]},
        "port_crossing": {"bow": [0.64, 0.68, 0.82, 0.97], "stern": [0.57, 0.63, 0.76, 0.92]},
        "starboard_beam": {"bow": [0.63, 0.69, 0.84, 1.00], "stern": [0.64, 0.65, 0.79, 0.95]},
        "port_beam": {"bow": [0.63, 0.66, 0.79, 0.96], "stern": [0.57, 0.60, 0.72, 0.88]},
        "being_overtaken": {"generic": [0.53, 0.62, 0.72, 0.82]},
        "overtaking": {"generic": [0.57, 0.64, 0.72, 0.81]},
        "starboard_rear_overtaking": {"bow": [0.62, 0.68, 0.78, 0.94], "stern": [0.54, 0.57, 0.67, 0.77]},
        "port_rear_overtaking": {"bow": [0.62, 0.68, 0.78, 0.94], "stern": [0.54, 0.57, 0.67, 0.77]},
        "port_rear_being_overtaken": {"bow": [0.57, 0.63, 0.72, 0.90], "stern": [0.55, 0.59, 0.67, 0.78]},
        "starboard_rear_being_overtaken": {"bow": [0.57, 0.63, 0.72, 0.90], "stern": [0.55, 0.59, 0.67, 1.10]},
    },
    300.0: {
        "head_on": {"generic": [0.61, 0.64, 0.73, 0.91]},
        "starboard_crossing": {"bow": [0.82, 0.93, 1.07, 1.26], "stern": [0.73, 0.79, 0.97, 1.20]},
        "port_crossing": {"bow": [0.70, 0.81, 1.01, 1.21], "stern": [0.76, 0.75, 0.95, 1.13]},
        "starboard_beam": {"bow": [0.75, 0.83, 1.01, 1.21], "stern": [0.72, 0.75, 0.91, 1.12]},
        "port_beam": {"bow": [0.72, 0.78, 0.96, 1.16], "stern": [0.62, 0.79, 0.82, 1.07]},
        "being_overtaken": {"generic": [0.62, 0.68, 0.77, 0.95]},
        "overtaking": {"generic": [0.63, 0.68, 0.77, 0.95]},
        "starboard_rear_overtaking": {"bow": [0.75, 0.82, 0.97, 1.14], "stern": [0.58, 0.75, 0.84, 0.99]},
        "port_rear_overtaking": {"bow": [0.75, 0.82, 0.97, 1.14], "stern": [0.58, 0.75, 0.84, 0.99]},
        "port_rear_being_overtaken": {"bow": [0.74, 0.80, 0.93, 1.10], "stern": [0.68, 0.73, 0.82, 0.97]},
        "starboard_rear_being_overtaken": {"bow": [0.74, 0.80, 0.93, 1.10], "stern": [0.68, 0.73, 0.82, 0.97]},
    },
}

# Table B.4: type coefficient relative to the bulk-carrier reference database.
TYPE_FACTOR = {
    "lng_chemical": 1.50,
    "lng": 1.50,
    "chemical": 1.50,
    "container": 1.33,
    "passenger": 1.20,
    "tanker": 1.36,
    "other": 1.00,
    "bulk": 1.00,
}

_GROUP_ALIASES = {
    "own_overtaking": "overtaking",
    "right_crossing": "starboard_crossing",
    "left_crossing": "port_crossing",
    "starboard_beam": "starboard_beam",
    "port_beam": "port_beam",
    "right_beam": "starboard_beam",
    "left_beam": "port_beam",
    "right_rear_overtaking": "starboard_rear_overtaking",
    "left_rear_overtaking": "port_rear_overtaking",
    "left_rear_being_overtaken": "port_rear_being_overtaken",
    "right_rear_being_overtaken": "starboard_rear_being_overtaken",
}


def _linear_interp_extrapolate(x: float, xs, ys) -> float:
    """Piecewise-linear interpolation with linear extrapolation at both ends."""
    x = float(x)
    if x <= xs[0]:
        i = 0
    elif x >= xs[-1]:
        i = len(xs) - 2
    else:
        i = next(i for i in range(len(xs) - 1) if xs[i] <= x <= xs[i + 1])
    x0, x1 = float(xs[i]), float(xs[i + 1])
    y0, y1 = float(ys[i]), float(ys[i + 1])
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def sda_m(
    own_length_m: float,
    target_length_m: float,
    encounter: str,
    pass_side: str = "bow",
    ship_type: str = "other",
) -> tuple[float, bool]:
    """Return (SDA_m, estimated_flag) using Appendix B example tables.

    ``pass_side`` is ``bow`` or ``stern`` for crossing/beam/rear-quarter rows;
    ``conservative`` uses the larger bow/stern row for an uncommitted passage.
    Head-on/overtaking rows whose port/starboard values are equal use ``generic``.
    Unclassified ``none``/``following`` encounters use a conservative maximum
    across Appendix-B examples and are explicitly marked as estimated.
    """
    encounter = str(encounter or "").strip().lower()
    group = _GROUP_ALIASES.get(encounter, encounter)
    own_length = float(own_length_m)
    target_length = float(target_length_m)
    if own_length <= 0 or target_length <= 0:
        raise ValueError("Ship lengths must be positive to calculate SDA")

    estimated = (
        own_length not in OWN_LENGTHS
        or target_length not in TARGET_LENGTHS
    )
    if group not in _TABLE[100.0]:
        # Appendix B does not define a following/unknown row. Retain a visible,
        # conservative estimated boundary rather than silently dropping near-
        # field risk: use the largest interpolated SDA across the example rows.
        estimated = True
        target_rows = {}
        for own_key in OWN_LENGTHS:
            row_values = []
            for group_data in _TABLE[own_key].values():
                for side_values in group_data.values():
                    row_values.append(_linear_interp_extrapolate(
                        target_length, TARGET_LENGTHS, side_values
                    ))
            target_rows[own_key] = max(row_values)
        base_nm = _linear_interp_extrapolate(
            own_length, OWN_LENGTHS,
            [target_rows[own_key] for own_key in OWN_LENGTHS],
        )
    else:
        group_data = _TABLE[100.0][group]
        if "generic" in group_data:
            selected_by_own = {
                own_key: _linear_interp_extrapolate(
                    target_length, TARGET_LENGTHS, _TABLE[own_key][group]["generic"]
                ) for own_key in OWN_LENGTHS
            }
        elif str(pass_side).lower() in ("conservative", "conservative_max", "max", "envelope"):
            # Until a crossing side is unambiguously committed, use the larger
            # of the Appendix-B bow/stern values as the risk-boundary envelope.
            selected_by_own = {
                own_key: max(
                    _linear_interp_extrapolate(target_length, TARGET_LENGTHS,
                                               _TABLE[own_key][group]["bow"]),
                    _linear_interp_extrapolate(target_length, TARGET_LENGTHS,
                                               _TABLE[own_key][group]["stern"]),
                ) for own_key in OWN_LENGTHS
            }
        else:
            side = "stern" if str(pass_side).lower() == "stern" else "bow"
            selected_by_own = {
                own_key: _linear_interp_extrapolate(
                    target_length, TARGET_LENGTHS, _TABLE[own_key][group][side]
                ) for own_key in OWN_LENGTHS
            }
        base_nm = _linear_interp_extrapolate(
            own_length, OWN_LENGTHS,
            [selected_by_own[own_key] for own_key in OWN_LENGTHS],
        )

    ship_type_key = str(ship_type or "other").strip().lower()
    if ship_type_key not in TYPE_FACTOR:
        ship_type_key = "other"
        estimated = True
    value_m = base_nm * TYPE_FACTOR[ship_type_key] * NM_TO_M
    return max(0.0, value_m), estimated
