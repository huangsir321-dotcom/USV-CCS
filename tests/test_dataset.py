from usv_sim.dataset import default_dataset_root, list_cases, load_case, dataset_summary
from usv_sim.engine import SimulationEngine
from usv_sim.ppo_agent import OBS_SIZE, build_observation


def test_uploaded_dataset_has_all_expected_cases():
    summary = dataset_summary(default_dataset_root())
    assert summary == {"1T": 138, "2T": 99, "3T": 31}
    assert len(list_cases(default_dataset_root())) == 268


def test_uploaded_cases_have_expected_target_counts_and_units():
    specs = list_cases(default_dataset_root())
    expected = {"1T": 1, "2T": 2, "3T": 3}
    for spec in specs:
        scene = load_case(spec, own_length_m=250, default_target_length_m=150)
        assert len(scene.targets) == expected[spec.group]
        assert 50 <= scene.own_config.length_m <= 500
        assert 50 <= scene.targets[0].length_m <= 500
        assert scene.own_state.u > 0
        assert all(t.state.u >= 0 for t in scene.targets)


def test_dataset_case_runs_and_observation_has_expected_shape():
    spec = next(s for s in list_cases(default_dataset_root()) if s.group == "3T")
    scene = load_case(spec, own_length_m=300, target_lengths_m=[50, 200, 500])
    engine = SimulationEngine(scene)
    obs = build_observation(engine.own, engine.config, engine.targets, engine.last_snapshot,
                            engine.route_x, engine.route_y, engine.route_start_x, engine.route_start_y)
    assert len(obs) == OBS_SIZE
    engine.step()
    assert engine.metrics.elapsed_s == engine.dt
    assert [round(t.length_m) for t in engine.targets] == [50, 200, 500]


def test_rule_policy_selects_same_feasible_bank_as_planner():
    from usv_sim.ppo_agent import RuleBasedPolicy
    scene = load_case(list_cases(default_dataset_root())[0])
    engine = SimulationEngine(scene)
    obs, mask = engine.prepare_decision()
    policy = RuleBasedPolicy()
    assert policy.predict(obs, engine.last_snapshot, mask) == engine.baseline_action


def test_long_tcpa_case_extends_generated_route_and_keeps_source_values():
    specs = list_cases(default_dataset_root(), "1T")
    spec = next(s for s in specs if s.scene_id == "050")
    scene = load_case(spec, own_length_m=200)
    tcpa_s = max(scene.metadata["source_tcpa_min"]) * 60.0
    route_len = (scene.route[0][0] ** 2 + scene.route[0][1] ** 2) ** 0.5
    assert scene.metadata["max_source_tcpa_s"] == tcpa_s
    assert route_len >= scene.own_state.u * (tcpa_s + 600.0) - 1e-6


def test_training_action_override_respects_monitor_phase_policy_gate():
    scene = load_case(next(s for s in list_cases(default_dataset_root()) if s.group == "1T"))
    engine = SimulationEngine(scene)
    mask = engine.action_mask(engine.last_snapshot)
    assert sum(mask) == 1
    assert mask[engine.baseline_action]
    engine.step(action_override=engine.baseline_action)
    assert engine.last_action == engine.baseline_action
    assert engine.last_policy_source == 'training_override'


def test_rule_action_mask_starts_head_on_at_critical_window_not_eight_nm():
    from usv_sim.actions import ACTION_SPECS, ACTION_COUNT
    from usv_sim.geometry import velocity_from_state
    scene = load_case(next(s for s in list_cases(default_dataset_root())
                           if s.group == "1T" and s.scene_id == "001"))
    engine = SimulationEngine(scene)
    far_mask = engine.action_mask(engine.last_snapshot)
    assert len(far_mask) == ACTION_COUNT and any(far_mask)
    assert not engine.maneuver_active
    assert all(not ACTION_SPECS[i].speed_mode.startswith('accelerate_')
               for i, allowed in enumerate(far_mask) if allowed)

    own_v = velocity_from_state(scene.own_state)
    target_v = velocity_from_state(scene.targets[0].state)
    scene.own_state.x += own_v[0] * 400.0
    scene.own_state.y += own_v[1] * 400.0
    scene.targets[0].state.x += target_v[0] * 400.0
    scene.targets[0].state.y += target_v[1] * 400.0
    engine = SimulationEngine(scene)
    near_mask = engine.action_mask(engine.last_snapshot)
    assert engine.maneuver_active
    assert all(ACTION_SPECS[i].offset_deg >= 10
               for i, allowed in enumerate(near_mask) if allowed)


def test_engine_rejects_action_outside_rule_mask():
    scene = load_case(next(s for s in list_cases(default_dataset_root()) if s.group == "1T"))
    engine = SimulationEngine(scene)
    try:
        engine.step(action_override=-1)
    except ValueError as exc:
        assert "action mask" in str(exc)
    else:
        raise AssertionError("Rule-disallowed action should be rejected")


def test_head_on_encounter_is_not_released_before_target_passes():
    # Regression for the old bug: at about t=430 s, a temporarily large DCPA
    # caused the avoidance manoeuvre to end even though the target was still
    # ahead (TCPA remained strongly positive). This then caused route recovery
    # to recreate the head-on collision geometry.
    spec = next(
        s for s in list_cases(default_dataset_root(), "1T")
        if s.scene_id == "001"
    )
    engine = SimulationEngine(load_case(spec, own_length_m=200, default_target_length_m=100))
    for _ in range(225):  # 450 s of simulated time
        engine.step()
    assessment = next(a for a in engine.last_snapshot.assessments if a.target_id == "TV1")
    assert engine.maneuver_active
    assert assessment.encounter == "head_on"
    assert assessment.tcpa_s > 0.0


def test_head_on_scene_reaches_safe_pass_before_route_recovery():
    spec = next(
        s for s in list_cases(default_dataset_root(), "1T")
        if s.scene_id == "001"
    )
    engine = SimulationEngine(load_case(spec, own_length_m=200, default_target_length_m=100))
    encounter_sda = engine.last_snapshot.assessments[0].sda_m
    for _ in range(700):
        engine.step()
    # After clear passing, the live classifier can assign another geometry and
    # another SDA. Do not apply that later threshold to the earlier head-on pass.
    # Also verify every interval's margin against the SDA then in force.
    assert engine.metrics.min_separation_m >= encounter_sda
    assert engine.metrics.minimum_domain_clearance_m >= 0
    assert engine.metrics.domain_intrusion_s == 0
    assert engine.metrics.collision_count == 0
    assert not engine.maneuver_active
