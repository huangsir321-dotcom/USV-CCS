"""Versioned candidate observation and shared actor/critic architecture."""
from __future__ import annotations
import math
import os
from typing import Optional
from .actions import ACTION_COUNT, ACTION_NAMES, ACTION_INTENTS, ACTION_SPECS, CONTRACT_VERSION

OBS_SCHEMA_VERSION = 'candidate-observation-v4'
CHECKPOINT_SCHEMA_VERSION = 2
TARGET_SLOTS = 3
TARGET_FEATURES = 12
CONTEXT_FEATURES = 14
AGGREGATE_FEATURES = 5
CANDIDATE_FEATURES = 9
CANDIDATE_OFFSET = 9 + TARGET_SLOTS * TARGET_FEATURES + CONTEXT_FEATURES + AGGREGATE_FEATURES
OBS_SIZE = CANDIDATE_OFFSET + ACTION_COUNT * CANDIDATE_FEATURES


def _clip(value, limit=3.0):
    return max(-limit, min(limit, float(value)))


def build_observation(own, own_config, targets, risk_snapshot, route_x=0., route_y=12000.,
                      route_start_x=0., route_start_y=0., context=None, candidates=None):
    """Stable target-ID slots plus all-target aggregates and all-hazard candidate scores.

    Candidate slots come directly from planner sweeps over every target and obstacle;
    scores use signed log1p to retain ranking without unbounded network inputs.
    """
    context = context or {}
    dx, dy = route_x-route_start_x, route_y-route_start_y
    length=max(1.,math.hypot(dx,dy)); ux,uy=dx/length,dy/length
    lateral=(own.x-route_start_x)*-uy+(own.y-route_start_y)*ux
    progress=(own.x-route_start_x)*ux+(own.y-route_start_y)*uy
    obs=[own.u/max(own_config.max_speed_mps,1), own.v/max(own_config.max_speed_mps,1),
         math.sin(own.psi),math.cos(own.psi),own.r/max(math.radians(own_config.max_yaw_rate_deg_s),.001),
         own.delta/max(math.radians(own_config.max_rudder_deg),.001),_clip(lateral/5000),
         _clip((length-progress)/30000),own_config.length_m/500]
    assessments={a.target_id:a for a in risk_snapshot.assessments}
    ordered=sorted((t for t in targets if t.active),key=lambda t:str(t.target_id))
    for i in range(TARGET_SLOTS):
        if i>=len(ordered): obs.extend([0.]*TARGET_FEATURES); continue
        t=ordered[i]; a=assessments.get(t.target_id)
        dx,dy=t.state.x-own.x,t.state.y-own.y; dist=math.hypot(dx,dy)
        bearing=math.atan2(dx,dy)-own.psi; heading=t.state.psi-own.psi
        obs.extend([1.,_clip(dist/12000),math.sin(bearing),math.cos(bearing),
                    _clip((t.state.u-own.u)/15),math.sin(heading),math.cos(heading),
                    _clip((a.dcpa_m if a else dist)/12000),_clip((a.tcpa_s if a else 0)/3600),
                    a.risk_score if a else 0,_clip((a.sda_m if a else 0)/5000),float(bool(a and a.own_give_way))])
    phase=str(context.get('phase',''))
    heading=context.get('reference_heading')
    heading=own.psi if heading is None else heading
    speed=context.get('reference_speed')
    speed=own.u if speed is None else speed
    obs.extend([float(bool(context.get('maneuver_active'))),math.sin(heading),math.cos(heading),
                speed/max(own_config.max_speed_mps,1),
                _clip(context.get('elapsed_s',0)/3600),
                float(phase=='stand_on_intervene'),float(phase=='give_way'),float(phase=='emergency'),
                float(phase=='passing'),float(phase=='monitor'),
                float(phase=='stand_on'),
                float(context.get('last_action',0))/max(ACTION_COUNT-1,1),
                float(bool(context.get('intervention'))),float(bool(context.get('odd_degraded')))])
    aa=list(assessments.values())
    obs.extend([_clip(len(ordered)/10),max((a.risk_score for a in aa),default=0),
                _clip(min((a.range_m/max(a.sda_m,1) for a in aa),default=3)),
                _clip(sum(a.own_give_way for a in aa)/10),float(any(a.tcpa_s<=0 and a.range_m<a.sda_m for a in aa))])
    by_id={c.candidate_id:c for c in (candidates or [])}
    for i in range(ACTION_COUNT):
        c=by_id.get(i)
        if c is None: obs.extend([0.]*CANDIDATE_FEATURES); continue
        score=float(c.score)
        if not math.isfinite(score): score=1e12 if score>=0 else -1e12
        obs.extend([1.,math.copysign(math.log1p(abs(score)),score),c.offset_deg/60,
                    c.speed_cmd_mps/max(own_config.max_speed_mps,1),
                    _clip(c.minimum_clearance_m/5000),_clip(c.preference_clearance_m/5000),
                    float(bool(c.violation)),float(c.feasible),float(c.status=='minimum_risk')])
    return obs


class RuleBasedPolicy:
    """Choose the lowest-scored allowed candidate; empty masks are a contract error."""
    def predict(self, observation, risk_snapshot=None, action_mask=None):
        allowed=[i for i in range(ACTION_COUNT) if action_mask is None or action_mask[i]]
        if not allowed: raise ValueError('action mask has no allowed candidate')
        if len(observation)==OBS_SIZE:
            populated=[i for i in allowed if observation[CANDIDATE_OFFSET+i*CANDIDATE_FEATURES]>.5]
            if populated:
                return min(populated,key=lambda i:(observation[CANDIDATE_OFFSET+i*CANDIDATE_FEATURES+1],i))
        return 0 if 0 in allowed else allowed[0]


def make_actor_critic():
    """Single architecture factory used by training and inference (lazy torch import)."""
    import torch.nn as nn
    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=nn.Sequential(nn.Linear(OBS_SIZE,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh())
            self.actor=nn.Linear(128,ACTION_COUNT); self.critic=nn.Linear(128,1)
        def forward(self,x):
            h=self.backbone(x)
            return self.actor(h),self.critic(h).squeeze(-1)
    return ActorCritic()


def checkpoint_metadata():
    return dict(schema_version=CHECKPOINT_SCHEMA_VERSION,contract_version=CONTRACT_VERSION,
                obs_schema_version=OBS_SCHEMA_VERSION,obs_size=OBS_SIZE,action_count=ACTION_COUNT,
                action_specs=[(a.offset_deg,a.speed_mode) for a in ACTION_SPECS])


def validate_checkpoint(state):
    if not isinstance(state,dict): raise ValueError('incompatible checkpoint: metadata missing; retrain candidate-v4')
    for key,value in checkpoint_metadata().items():
        if state.get(key)!=value: raise ValueError(f'incompatible checkpoint {key}: expected {value!r}; retrain candidate-v4')


class PPOPolicy:
    def __init__(self,checkpoint: Optional[str]=None):
        self.checkpoint=checkpoint; self.fallback=RuleBasedPolicy(); self.model=None
        self.available=False; self.load_error=None; self.metadata={}
        if not checkpoint: return
        try:
            if not os.path.isfile(checkpoint): raise FileNotFoundError(checkpoint)
            import torch
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            validate_checkpoint(state)
            self.metadata={key:state.get(key) for key in ('data_split','config','training_steps')}
            model=make_actor_critic(); model.load_state_dict(state['model']); model.eval()
            self.model=model; self.torch=torch; self.available=True
        except Exception as exc:
            self.load_error=f'PPO checkpoint rejected: {exc}'
    def predict(self,observation,risk_snapshot=None,action_mask=None):
        if not self.available: return self.fallback.predict(observation,risk_snapshot,action_mask)
        if len(observation)!=OBS_SIZE: raise ValueError('observation schema mismatch')
        if action_mask is not None and not any(action_mask): raise ValueError('empty action mask')
        with self.torch.no_grad():
            logits,_=self.model(self.torch.tensor(observation,dtype=self.torch.float32).unsqueeze(0))
            if action_mask is not None:
                logits=logits.masked_fill(~self.torch.as_tensor(action_mask,dtype=self.torch.bool).unsqueeze(0),-1e9)
            return int(logits.argmax(-1).item())
