import unittest
import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from usv_sim import ppo_agent as p
from usv_sim.models import ShipState, ShipConfig, RiskSnapshot, TargetShip

class LearningContractTests(unittest.TestCase):
    def test_candidate_contract_and_context(self):
        self.assertEqual(len(p.ACTION_NAMES), p.ACTION_COUNT)
        self.assertEqual(p.ACTION_COUNT, 125)
        own, cfg = ShipState(), ShipConfig()
        snap = RiskSnapshot([], None, 0, False)
        a = p.build_observation(own, cfg, [], snap, context={'maneuver_active': False})
        b = p.build_observation(own, cfg, [], snap, context={'maneuver_active': True})
        self.assertEqual(len(a), p.OBS_SIZE)
        self.assertNotEqual(a, b)
    def test_none_references_and_actual_phase_names(self):
        own, cfg, snap = ShipState(), ShipConfig(), RiskSnapshot([],None,0,False)
        values=[]
        for phase in ('monitor','give_way','stand_on_intervene','emergency','passing'):
            values.append(p.build_observation(own,cfg,[],snap,context={
                'reference_heading':None,'reference_speed':None,'phase':phase}))
        self.assertEqual(len({tuple(v) for v in values}),5)
    def test_checkpoint_rejects_old_contract(self):
        with self.assertRaisesRegex(ValueError,'incompatible checkpoint'):
            p.validate_checkpoint({'obs_size':42,'action_count':5})
        p.validate_checkpoint(p.checkpoint_metadata())
    def test_teacher_selects_lowest_allowed_candidate(self):
        candidates = [NS(candidate_id=i, score=float(21-i), offset_deg=0, speed_cmd_mps=7,
                         minimum_clearance_m=100, preference_clearance_m=200, violation=False,
                         feasible=True, status='safe') for i in range(p.ACTION_COUNT)]
        obs = p.build_observation(ShipState(), ShipConfig(), [], RiskSnapshot([],None,0,False), candidates=candidates)
        self.assertEqual(p.RuleBasedPolicy().predict(obs, action_mask=[True]*(p.ACTION_COUNT-1)+[False]),p.ACTION_COUNT-2)
    def test_reward_events_and_unclipped_collision(self):
        from usv_sim.rewards import compute_reward
        own=ShipState(); engine=NS(own=own, config=ShipConfig(), route_ux=0, route_uy=1,
          route_start_x=0, route_start_y=0, dt=2, last_snapshot=RiskSnapshot([],None,0,False),
          metrics=NS(collision_count=1,obstacle_collision_count=0,route_returned=True,destination_reached=True),
          last_intervention=False, last_plan=None)
        value, parts=compute_reward(engine,own.copy(),0,0,0,dt_s=2)
        self.assertEqual(parts['collision'],-100)
        self.assertLess(value,-80)
        _,parts2=compute_reward(engine,own.copy(),0,1,0,True,True,dt_s=2)
        self.assertEqual(parts2['route_return'],0)
        self.assertEqual(parts2['destination'],0)
    def test_split_groups_duplicate_geometry(self):
        from usv_sim.data_split import split_cases, geometry_fingerprint
        from usv_sim.dataset import list_cases
        cases=list_cases()
        split=split_cases(cases,7)
        locations={}
        for name,items in split.items():
            for case in items:
                key=geometry_fingerprint(case)
                self.assertIn(locations.get(key,name),(name,))
                locations[key]=name
        self.assertEqual(sum(map(len,split.values())),len(cases))
        self.assertEqual(split,split_cases(cases,7))
    def test_rule_cost_is_audited_even_when_rule_mask_is_disabled(self):
        from usv_sim.rewards import compute_reward
        own=ShipState()
        engine=NS(own=own,config=ShipConfig(),route_ux=0,route_uy=1,route_start_x=0,
                  route_start_y=0,dt=2,last_snapshot=RiskSnapshot([],None,0,False),
                  metrics=NS(collision_count=0,obstacle_collision_count=0,route_returned=False,
                             destination_reached=False),last_intervention=False,
                  last_plan=NS(reasons=(),rule_reasons=('T:R14_starboard_required',)))
        _,parts=compute_reward(engine,own,0,0,0)
        self.assertLess(parts['rule'],0)

@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed')
class TorchCheckpointTests(unittest.TestCase):
    def test_training_cli_independent_ablation_flags(self):
        import subprocess, sys
        result=subprocess.run([sys.executable,'train_ppo.py','--help'],capture_output=True,text=True,check=True)
        self.assertIn('--no-rule-mask',result.stdout)
        self.assertIn('--no-monitor',result.stdout)

    def test_shared_factory_and_checkpoint_rejection(self):
        import torch
        model=p.make_actor_critic()
        logits, values=model(torch.zeros((2,p.OBS_SIZE)))
        self.assertEqual(tuple(logits.shape),(2,p.ACTION_COUNT))
        self.assertEqual(tuple(values.shape),(2,))
        with tempfile.TemporaryDirectory() as folder:
            checkpoint=Path(folder)/'policy.pt'
            torch.save({**p.checkpoint_metadata(),'model':model.state_dict()},checkpoint)
            policy=p.PPOPolicy(str(checkpoint))
            self.assertTrue(policy.available,policy.load_error)
            action=policy.predict([0.]*p.OBS_SIZE,action_mask=[False]*(p.ACTION_COUNT-1)+[True])
            self.assertEqual(action,p.ACTION_COUNT-1)
            torch.save({'model':model.state_dict(),'obs_size':42},checkpoint)
            policy=p.PPOPolicy(str(checkpoint))
            self.assertFalse(policy.available)
            self.assertIn('incompatible checkpoint',policy.load_error)

if __name__=='__main__': unittest.main()
