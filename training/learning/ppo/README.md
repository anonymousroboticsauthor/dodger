# Project-local PPO trainer

`train.py` is based on Brax 0.14.2's
`brax.training.agents.ppo.train` module.

The local copy keeps the upstream training behavior and adds one optional
feature: when `enable_kl_limit=True`, a minibatch whose mean KL exceeds
`hard_kl_limit` is skipped together with every remaining minibatch update for
that rollout.  The stop state is reset explicitly after fresh rollout data is
collected.  This avoids guessing rollout boundaries from a numerically small KL
value and does not enable Brax's adaptive-KL learning-rate schedule.
