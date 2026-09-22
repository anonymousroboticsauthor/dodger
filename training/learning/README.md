# Learning entry points

The public DODGER workflow uses these commands:

```bash
train-jax-ppo --env_name=G1MultiObstacleNavigation
python -m learning.play_jax_policy --checkpoint /path/to/run
python -m learning.export_jax_policy_onnx --checkpoint /path/to/run
```

See the [navigation guide](../training/_src/locomotion/g1/multi_obstacle_navigation/README.md)
for installation, configuration overrides, checkpoint layout, and deployment
model interfaces.
