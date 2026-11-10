PPO checkpoint directory.

After installing requirements-ml.txt, run:
  python train_ppo.py --steps 30000 --out models/ppo_policy.pt

The GUI auto-loads models/ppo_policy.pt when it exists. No model is bundled: the first run uses
an explicit rule-based fallback rather than pretending that an untrained PPO network is useful.
