from usv_sim.engine import SimulationEngine
from usv_sim.scenarios import make_scenario


def test_clear_tracks_keeps_current_state_and_restarts_history():
    engine = SimulationEngine(make_scenario(2))
    engine.own.x += 123.0
    engine.own.y += 456.0
    engine.targets[0].state.x += 50.0
    engine.own_track.extend([(1.0, 2.0), (3.0, 4.0)])
    engine.target_tracks[engine.targets[0].target_id].append((8.0, 9.0))

    own_position = (engine.own.x, engine.own.y)
    target_positions = {
        t.target_id: (t.state.x, t.state.y) for t in engine.targets
    }
    engine.clear_tracks()

    assert engine.own_track == [own_position]
    for target in engine.targets:
        assert engine.target_tracks[target.target_id] == [target_positions[target.target_id]]
    assert (engine.own.x, engine.own.y) == own_position
