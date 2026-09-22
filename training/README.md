# DODGER training

This directory contains the MJX/JAX training implementation for DODGER, a
constraint-aware high-level navigation policy for Unitree G1. The simulator
and environment API are derived from
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground).

The Python package is named `training`; no checkout-specific paths are used.
The detailed environment, reproduction, playback, and ONNX export guide is in
[the navigation README](training/_src/locomotion/g1/multi_obstacle_navigation/README.md).

## Quick start

```bash
conda create -n dodger-training python=3.11 -y
conda activate dodger-training
pip install -U "jax[cuda12]"
pip install -e ".[learning]"
python -c "import jax, training; print(jax.default_backend())"
train-jax-ppo --env_name=G1MultiObstacleNavigation
```

The first environment load downloads MuJoCo Menagerie assets when they are not
already available. Training checkpoints and W&B logs are written below
`logs/` and `wandb/`, which are excluded from version control.

## License

The inherited MuJoCo Playground code remains under the Apache License 2.0.
See [LICENSE](LICENSE).
