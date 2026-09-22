# DODGER

**DODGER**, **D**ynamic **O**bstacle Avoidance with **D**PCBF-**G**uided
**E**xploration in **R**einforcement Learning, is a safety-guided RL framework
for goal-directed robot navigation among multiple dynamic obstacles.

## Training

The high-level navigation policy is trained with MuJoCo Playground. Training
code, environment configuration, and policy-export instructions are located in
the [training README](training/README.md).

## Deployment

The Unitree G1 simulation and real-robot runtime is located in
[`deployment/`](deployment/). It includes the exported policies, MuJoCo
deployment simulator, ROS 2 perception stack, G1 controller, and interactive
goal interface.

- [Deployment setup](deployment/doc/setup_en.md)
- [Perception setup](deployment/ros2/README.md)
- [Simulation and hardware operation](deployment/deploy/robots/g1/config/policy/navigation/v1/RUN.md)

## License

See [`deployment/LICENCE`](deployment/LICENCE) and the license files included
with individual third-party components. Training dependencies and code retain
their respective upstream licenses.
