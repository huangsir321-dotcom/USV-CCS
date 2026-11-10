# Uploaded Dataset Integration

## Inventory

| Group | CSV cases | Data rows per CSV | Meaning |
|---|---:|---:|---|
| 1T | 138 | 1 | One target ship |
| 2T | 99 | 2 | Two target ships |
| 3T | 31 | 3 | Three target ships |
| Total | 268 |  | 268 encounter cases |

## Source fields

The CSV header includes `stage`, `meeting_situation`, `ownship_lat`, `ownship_lon`, `target_lat`, `target_lon`, `ownship_sog(KN)`, `ownship_cog(deg)`, `target_sog(KN)`, `target_cog(deg)`, `distance(NM)`, `bearing(deg)`, `DCPA(NM)`, `TCPA(min)`, `relative_speed(KN)` and `relative_course(deg)`.

The `stage` and `meeting_situation` values are retained in each imported scenario's metadata. 1T has 138 collision-risk rows. 2T has 155 collision-risk and 43 no-risk target rows. 3T has 93 collision-risk target rows. The supplied `target_behavior` and `ownship_behavior` fields are blank in the uploaded CSVs.

## Conversion

- Course/heading: degrees clockwise from north, converted to radians for the simulator.
- Speed: knots converted to m/s using 1 kn = 0.514444 m/s.
- Position: local tangent/equirectangular conversion relative to ownship position; ownship is the local origin.
- Route: a straight planned route extending 15 NM along ownship's initial COG.
- Target motion after initialization: constant velocity/constant course by default, because the source has no time-series data or scripted action field.

## Vessel dimensions

No vessel length, beam, draft, displacement, rudder, propeller, or MMG hydrodynamic coefficients are supplied. The UI lets the operator set ownship length from 50 to 500 m, set a default target length, and edit each target individually. PPO training randomizes ownship and target lengths among values from 50 to 500 m to expose the model to multiple scales. Beam is provisionally estimated as 0.16 * length (minimum 8 m), which is an explicit simulation assumption, not source data.

## Training use

- Warm-start reads all 268 cases and uses source risk-stage labels when creating rule-policy pseudo-labels.
- PPO simulation rollouts draw cases from the same pool, reset ship states to the supplied initial conditions, and simulate subsequent motion using the current motion models.
- The uploaded cases are initial-condition scenarios, not demonstrated expert trajectories; this data alone cannot support behavior cloning from actual human manoeuvre commands.
