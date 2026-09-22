# DODGER G1 navigation training

`G1MultiObstacleNavigation` trains a 10 Hz high-level G1 navigation actor over
a frozen 50 Hz locomotion policy. MuJoCo integrates the robot and applies the
deployment PD gains at 500 Hz. A batched JAX primal-dual interior-point solver
evaluates a 50 Hz multi-obstacle DPCBF QP during training.

For DODGER, the policy action—not the QP-filtered action—updates the
environment. The filtered action is a counterfactual safe reference used by
the tracking reward. Two soft constraints determine the fractional
continuation discount:

- `dpcbf_constraint`: violation of
  `Lf h + Lg h u_policy + alpha h >= 0`;
- `low_level_command_constraint`: violation of the sagittal, lateral, and yaw
  command ranges used to train the frozen locomotion policy.

Each constraint has its own EMA normalization scale. Their probabilities share
`dodger.probability_ramp_end_steps`, and the largest normalized constraint is
used at each step. A shared critic MLP has positive- and negative-return heads:
the fractional continuation affects only the positive return, while physical
terminations and penalty returns use the hard environment termination.

The fixed implementation choices intentionally omitted from the public config
are: 50 Hz QP evaluation, surface-clearance GAT edges without an appended node
radius, the derivative DPCBF constraint above, both DODGER soft constraints,
the split value critic, dynamic curriculum transitions, and no high-level
action randomization or yaw-magnitude penalty.

## Installation

From the repository's `training/` directory:

```bash
conda create -n dodger-training python=3.11 -y
conda activate dodger-training
pip install -U "jax[cuda12]"
pip install -e ".[learning]"
python -c "import jax, training; print(jax.default_backend())"
```

The final command should print `gpu`. The first locomotion environment load may
download MuJoCo Menagerie assets.

## Train

```bash
train-jax-ppo --env_name=G1MultiObstacleNavigation --use_wandb
```

The default PPO configuration uses 1,024 parallel environments,
`unroll_length=32`, and at most 600 million environment transitions. Logs and
checkpoints are stored in `logs/G1MultiObstacleNavigation-<timestamp>/`.
Environment overrides remain available for experimental quantities, for
example:

```bash
train-jax-ppo \
  --env_name=G1MultiObstacleNavigation \
  --config_overrides='{"obstacle.count":15}'
```

The actor receives the ten highest-priority detected obstacles. Priority 1
ranks obstacles by closing alignment first and distance second. Unused static
training slots are masked. The actor GAT computes only the robot-node
embedding; the critic receives privileged local-frame state.

## Play a checkpoint

The dedicated player selects the latest numbered checkpoint when given a run
or `checkpoints/` directory, uses the final curriculum stage, chains random
goals, and disables the runtime QP filter by default:

```bash
python -m learning.play_jax_policy \
  --checkpoint logs/G1MultiObstacleNavigation-YYYYMMDD-HHMMSS
```

Videos are written to `<run>/play/`. Pass `--runtime_filter` only for a
diagnostic rollout with the QP-filtered command.

## Export the deployment actor to ONNX

```bash
python -m learning.export_jax_policy_onnx \
  --checkpoint logs/G1MultiObstacleNavigation-YYYYMMDD-HHMMSS
```

The latest completed checkpoint is selected automatically. By default the two
models are written to `<run>/onnx/checkpoint_<step>/`:

- `gat_encoder.onnx`: dynamic `[batch, detected_nodes, 8]` input and a
  16-dimensional robot embedding output. The robot must be node 0; detected
  obstacles and the goal may appear in any order. No padded nodes or validity
  mask are passed at deployment.
- `policy_head.onnx`: the robot embedding plus the 13-dimensional local state,
  producing `[sagittal_acceleration, lateral_acceleration, yaw_rate]`.

The ONNX models contain only the high-level actor. They do not contain the QP,
the frozen low-level locomotion policy, PD gains, or observation construction.

## Frozen low-level policy asset

The repository includes the converted `low_level_policy.npz` used by the
environment. To reproduce that conversion from the original ONNX policy:

```bash
python tools/convert_g1_navigation_policy.py \
  --input /path/to/velocity_policy.onnx
```

The converter validates the 98-to-29 architecture, ELU layers, normalization,
and numerical equivalence before writing the asset.
