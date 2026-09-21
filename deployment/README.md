# DODGER deployment

This directory contains the simulation and real-robot runtime for DODGER on
the Unitree G1. It includes the MuJoCo deployment simulator, the G1 controller,
the exported high- and low-level ONNX policies, dynamic-obstacle perception,
the interactive goal interface, and DPCBF visualization.

The navigation training environment is intentionally not included here.
Training is maintained separately under [`../training`](../training/).

## Setup

1. [General deployment and simulation setup](doc/setup_en.md)
2. [ROS 2 perception setup](ros2/README.md)
3. [ROS 2 Foxy setup for the G1 onboard computer](doc/ros2_foxy_setup.md)
4. [Network and remote visualization setup](doc/ros2_teleop.md)

## Run DODGER

Use the [Navigation runtime guide](deploy/robots/g1/config/policy/navigation/v1/RUN.md)
for the complete simulation and hardware procedures, safety checks, FSM
transitions, goal-interface controls, and shutdown sequence.

## Policy artifacts

The deployable policy bundle is stored in
`deploy/robots/g1/config/policy/navigation/v1/exported/`:

- `high_level_navigation_policy/gat_encoder.onnx`
- `high_level_navigation_policy/policy_head.onnx`
- `low_level_locomotion_policy.onnx`

See the policy bundle [README](deploy/robots/g1/config/policy/navigation/v1/README.md)
for the observation contract and checksums.

## License

This deployment code retains the licenses and notices of the upstream
projects from which it is derived. See [`LICENCE`](LICENCE) and the license
files shipped with the relevant third-party components.

