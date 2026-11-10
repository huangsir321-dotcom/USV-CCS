"""Dataset-driven PPO training on the user's 1T/2T/3T encounter cases.

Optional candidate-baseline warmup and PPO rollouts use only geometry-grouped
training cases. Validation/test groups are reserved and stored in the checkpoint.
Each policy decision advances the shared simulator clock by up to 10 seconds,
using the same 2-second dynamics as interactive inference.

The uploaded CSVs contain initial encounter states, not continuous trajectories. The
simulator generates subsequent motion using its vessel/target motion models.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from usv_sim.dataset import list_cases, load_case, default_dataset_root
from usv_sim.engine import SimulationEngine
from usv_sim.ppo_agent import OBS_SIZE, RuleBasedPolicy, make_actor_critic, checkpoint_metadata, validate_checkpoint
from usv_sim.actions import CONTROL_DT_S, DECISION_PERIOD_S
from usv_sim.data_split import split_cases, split_manifest
from usv_sim.models import RiskSnapshot

LENGTH_CHOICES = [50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]


ActorCritic = make_actor_critic


class SamplingPolicy:
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        self.last_log_prob = None
        self.last_value = None
        self.last_action = 0

    def predict(self, observation, risk_snapshot=None, action_mask=None):
        x = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(x)
        if action_mask is not None:
            mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            logits = logits.masked_fill(~mask, -1e9)
        dist = Categorical(logits=logits)
        action = dist.sample()
        self.last_log_prob = dist.log_prob(action).squeeze(0)
        self.last_value = value.squeeze(0)
        self.last_action = int(action.item())
        return self.last_action


def make_dataset_scenario(spec, episode_index: int, random_lengths: bool = True):
    if random_lengths:
        own_length = random.choice(LENGTH_CHOICES)
        target_lengths = [random.choice(LENGTH_CHOICES) for _ in range(3)]
    else:
        own_length, target_lengths = 200.0, [100.0, 100.0, 100.0]
    return load_case(spec, own_length_m=own_length, target_lengths_m=target_lengths, dt_s=CONTROL_DT_S)


def build_warmup_data(specs, root: Path, use_rule_mask=True, use_safety_monitor=True):
    observations, labels = [], []
    for index, spec in enumerate(specs):
        engine = SimulationEngine(make_dataset_scenario(spec, index, random_lengths=False))
        engine.use_rule_mask = use_rule_mask
        engine.use_safety_monitor = use_safety_monitor
        obs, mask = engine.prepare_decision()
        observations.append(obs)
        labels.append(engine.baseline_action)
    return np.asarray(observations, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def warm_start(model, optimizer, observations, labels, epochs, device):
    x = torch.as_tensor(observations, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    model.train()
    for epoch in range(epochs):
        idx = torch.randperm(len(x), device=device)
        losses = []
        for start in range(0, len(x), 64):
            batch = idx[start:start + 64]
            logits, _ = model(x[batch])
            loss = nn.functional.cross_entropy(logits, y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"WARMUP_EPOCH {epoch + 1}/{epochs} loss={np.mean(losses):.5f}", flush=True)
    model.eval()


def ppo_update(model, optimizer, obs, actions, old_logp, action_masks, advantages, returns,
               device, epochs, batch_size):
    x = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)
    act = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=device)
    old_lp = torch.as_tensor(np.asarray(old_logp), dtype=torch.float32, device=device)
    masks = torch.as_tensor(np.asarray(action_masks), dtype=torch.bool, device=device)
    adv = torch.as_tensor(np.asarray(advantages), dtype=torch.float32, device=device)
    ret = torch.as_tensor(np.asarray(returns), dtype=torch.float32, device=device)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    idx = np.arange(len(x))
    model.train()
    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            b = idx[start:start + batch_size]
            logits, value = model(x[b])
            logits = logits.masked_fill(~masks[b], -1e9)
            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(act[b])
            ratio = torch.exp(new_lp - old_lp[b])
            s1 = ratio * adv[b]
            s2 = torch.clamp(ratio, 0.8, 1.2) * adv[b]
            policy_loss = -torch.minimum(s1, s2).mean()
            value_loss = 0.5 * (ret[b] - value).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + value_loss - 0.01 * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
    model.eval()


def main():
    parser = argparse.ArgumentParser(description="Train PPO on the uploaded maritime encounter cases")
    parser.add_argument("--steps", type=int, default=150000, help="PPO interaction steps")
    parser.add_argument("--episode-steps", type=int, default=120, help="Minimum episode steps; longer source TCPA extends an episode")
    parser.add_argument("--max-episode-steps", type=int, default=1500, help="Safety cap on episode length")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="models/ppo_policy.pt")
    parser.add_argument("--dataset-root", default=str(default_dataset_root()))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-rule-mask", action="store_true", help="Offline ablation: disable COLREG candidate masking")
    parser.add_argument("--no-monitor", action="store_true", help="Offline ablation: disable runtime safety monitor")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda",
                        help="Training device. Defaults to CUDA and fails rather than silently falling back.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA training was requested but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch wheel and verify the NVIDIA driver with nvidia-smi."
            )
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device_name = torch.cuda.get_device_name(0)
        print(f"DEVICE cuda | {device_name} | torch_cuda={torch.version.cuda}", flush=True)
    else:
        device_name = "CPU"
        print("DEVICE cpu | GPU training disabled by selection", flush=True)
    root = Path(args.dataset_root)
    specs = list_cases(root)
    if not specs:
        raise SystemExit(f"No case CSV files found under {root}. Extract the uploaded 1T/2T/3T datasets first.")

    by_group = {group: sum(1 for s in specs if s.group == group) for group in ("1T", "2T", "3T")}
    print(f"DATASET total={len(specs)} 1T={by_group['1T']} 2T={by_group['2T']} 3T={by_group['3T']}", flush=True)
    print("NOTE source CSVs are initial-state scenarios, not time-series trajectories.", flush=True)

    splits = split_cases(specs, args.seed)
    manifest = split_manifest(splits, args.seed)
    specs = splits['train']
    if not specs:
        raise SystemExit('No training groups in this split; choose another seed or add data')
    print('SPLIT ' + ' '.join(f'{key}={len(value)}' for key, value in splits.items()), flush=True)
    model = make_actor_critic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    out = Path(args.out)
    total_steps = 0
    episode = 0
    resumed_statistics = {}
    clock = {"control_dt_s": CONTROL_DT_S, "decision_period_s": DECISION_PERIOD_S}
    case_order = list(specs)
    case_cursor = 0
    if args.resume:
        if not out.exists():
            raise SystemExit(f'Resume checkpoint does not exist: {out}')
        checkpoint = torch.load(out, map_location=device, weights_only=False)
        validate_checkpoint(checkpoint)
        if checkpoint.get('clock') != clock:
            raise SystemExit('Resume physical/decision clock mismatch')
        for key in ('no_rule_mask', 'no_monitor'):
            if checkpoint.get('config', {}).get(key) != getattr(args, key):
                raise SystemExit(f'Resume configuration mismatch: {key}')
        resumed_statistics = checkpoint.get('statistics', {})
        if checkpoint.get('data_split') != manifest:
            raise SystemExit('Resume dataset split/content changed')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        total_steps = checkpoint['training_steps']
        episode = checkpoint.get('episode', 0)
        random.setstate(checkpoint['rng']['python'])
        np.random.set_state(checkpoint['rng']['numpy'])
        torch.set_rng_state(checkpoint['rng']['torch'].cpu())
        if device.type == 'cuda' and checkpoint['rng'].get('cuda'):
            torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
        saved_order = checkpoint.get('case_order', [s.key for s in specs])
        lookup = {s.key:s for s in specs}
        case_order = [lookup[key] for key in saved_order]
        case_cursor = checkpoint.get('case_cursor', 0)
        print(f'RESUME {out} steps={total_steps}', flush=True)
    else:
        random.shuffle(case_order)
        if args.warmup_epochs > 0:
            warm_obs, warm_labels = build_warmup_data(specs, root, not args.no_rule_mask, not args.no_monitor)
            warm_start(model, optimizer, warm_obs, warm_labels, args.warmup_epochs, device)
    model.eval()
    buf_obs, buf_actions, buf_logp, buf_masks, buf_adv, buf_returns = [], [], [], [], [], []
    decision_count = resumed_statistics.get('decisions', 0)
    valid_action_count = resumed_statistics.get('valid_action_total', 0)
    intervention_count = resumed_statistics.get('interventions', 0)
    reward_components = dict(resumed_statistics.get('reward_components', {}))
    update_count = resumed_statistics.get('ppo_updates', 0)
    bootstrap_count = resumed_statistics.get('truncation_bootstraps', 0)
    episode_rewards = []
    print(f"TRAINING_START steps={args.steps} device={device}", flush=True)
    while total_steps < args.steps:
        if case_cursor >= len(case_order):
            random.shuffle(case_order)
            case_cursor = 0
        spec = case_order[case_cursor]
        case_cursor += 1
        episode += 1
        scenario = make_dataset_scenario(spec, episode, random_lengths=True)
        engine = SimulationEngine(scenario)
        engine.use_rule_mask = not args.no_rule_mask
        engine.use_safety_monitor = not args.no_monitor
        source_tcpa_s = [float(v) * 60.0 for v in scenario.metadata.get("source_tcpa_min", []) if float(v) > 0.0]
        max_tcpa_s = max(source_tcpa_s, default=float(scenario.metadata.get("max_source_tcpa_s", 0.0)))
        required_steps = int(math.ceil((max_tcpa_s + 600.0) / DECISION_PERIOD_S))
        episode_limit = min(args.max_episode_steps, max(args.episode_steps, required_steps))
        sampler = SamplingPolicy(model, device)


        ep_obs, ep_actions, ep_logp, ep_masks = [], [], [], []
        ep_rewards, ep_dones, ep_values, ep_elapsed = [], [], [], []
        ep_reward = 0.0
        for _ in range(episode_limit):
            if total_steps >= args.steps:
                break
            obs, action_mask = engine.prepare_decision()
            action = sampler.predict(obs, engine.last_snapshot, action_mask)
            logp = sampler.last_log_prob.detach().cpu().item()
            value = sampler.last_value.detach().cpu().item()
            reward, terminal, info = engine.advance_decision(action)
            ep_elapsed.append(float(info['elapsed_s']))
            for key, component in info.get('reward_components', {}).items():
                reward_components[key] = reward_components.get(key, 0.) + float(component)
            valid_action_count += sum(action_mask)
            intervention_count += int(bool(info.get('intervention', getattr(engine, 'last_intervention', False))))
            decision_count += 1
            ep_obs.append(obs)
            ep_actions.append(action)
            ep_logp.append(logp)
            ep_masks.append(action_mask)
            ep_rewards.append(reward)
            ep_dones.append(float(terminal))
            ep_values.append(value)
            ep_reward += reward
            total_steps += 1
            if terminal:
                break

        if ep_obs:
            # Bootstrap at artificial episode truncation, but not at collision/destination.
            if ep_dones[-1] > 0.5:
                next_value = 0.0
            else:
                last_obs, _ = engine.prepare_decision()
                with torch.no_grad():
                    _, nv = model(torch.as_tensor(last_obs, dtype=torch.float32, device=device).unsqueeze(0))
                next_value = float(nv.item())
                bootstrap_count += 1
            adv = np.zeros(len(ep_rewards), dtype=np.float32)
            gae = 0.0
            for t in reversed(range(len(ep_rewards))):
                nonterminal = 1.0 - ep_dones[t]
                v_next = next_value if t == len(ep_rewards) - 1 else ep_values[t + 1]
                discount = 0.99 ** (ep_elapsed[t] / DECISION_PERIOD_S)
                trace = 0.95 ** (ep_elapsed[t] / DECISION_PERIOD_S)
                delta = ep_rewards[t] + discount * v_next * nonterminal - ep_values[t]
                gae = delta + discount * trace * nonterminal * gae
                adv[t] = gae
            ret = adv + np.asarray(ep_values, dtype=np.float32)
            buf_obs.extend(ep_obs); buf_actions.extend(ep_actions); buf_logp.extend(ep_logp)
            buf_masks.extend(ep_masks)
            buf_adv.extend(adv.tolist()); buf_returns.extend(ret.tolist())
            episode_rewards.append(ep_reward)

        if len(buf_obs) >= 512 or total_steps >= args.steps:
            ppo_update(model, optimizer, buf_obs, buf_actions, buf_logp, buf_masks,
                       buf_adv, buf_returns, device, args.ppo_epochs, 128)
            update_count += 1
            buf_obs, buf_actions, buf_logp, buf_masks, buf_adv, buf_returns = [], [], [], [], [], []

        if total_steps % 512 < max(1, args.episode_steps) or total_steps >= args.steps:
            avg_r = float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0
            print(f"PROGRESS {total_steps} {args.steps} episodes={episode} avg_reward20={avg_r:.3f} valid_actions={valid_action_count/max(decision_count,1):.2f} interventions={intervention_count}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        **checkpoint_metadata(),
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': vars(args),
        'clock': clock,
        'data_split': manifest,
        'training_steps': total_steps,
        'episode': episode,
        'case_order': [s.key for s in case_order],
        'case_cursor': case_cursor,
        'training_device': str(device),
        'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        'statistics': {'decisions': decision_count, 'interventions': intervention_count,
                       'valid_action_total': valid_action_count, 'reward_components': reward_components,
                       'ppo_updates': update_count, 'truncation_bootstraps': bootstrap_count,
                       'pending_rollout_steps': len(buf_obs),
                       'valid_action_mean': valid_action_count/max(decision_count,1)},
    }, out)
    print(f"TRAINING_COMPLETE checkpoint={out.resolve()} cases={len(specs)} steps={total_steps}", flush=True)


if __name__ == "__main__":
    main()
