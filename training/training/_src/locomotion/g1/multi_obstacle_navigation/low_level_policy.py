"""Frozen Unitree RL MjLab G1 velocity policy implemented in pure JAX."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Exact values from unitree_rl_mjlab/.../velocity/v1/params/deploy.yaml.
STIFFNESS = jp.asarray((
    40.2, 99.1, 40.2, 99.1, 28.5, 28.5, 40.2, 99.1, 40.2, 99.1, 28.5, 28.5,                 # Lower body
    40.2, 28.5, 28.5,                                                                       # Waist
    14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8, 14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8,     # Upper body
))        

DAMPING = jp.asarray((
    2.6, 6.3, 2.6, 6.3, 1.8, 1.8, 2.6, 6.3, 2.6, 6.3, 1.8, 1.8,                             # Lower body
    2.6, 1.8, 1.8,                                                                          # Waist
    0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1, 0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1,                   # Upper body
))

DEFAULT_JOINT_POSITION = jp.asarray((
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,                         # Lower body
    0.0, 0.0, 0.0,                                                                          # Waist
    0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0, 0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0,            # Upper body
))

ACTION_SCALE = jp.asarray((
    0.55, 0.35, 0.55, 0.35, 0.44, 0.44, 0.55, 0.35, 0.55, 0.35, 0.44, 0.44,                 # Lower body
    0.55, 0.44, 0.44,                                                                       # Waist
    0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07, 0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07,     # Upper body
))


class FrozenVelocityPolicy:
  """98->29 ELU MLP whose parameters are constant JAX arrays."""

  def __init__(self, weights_path: str | Path):
    weights_path = Path(weights_path)
    if not weights_path.is_file():
      raise FileNotFoundError(
          f"Missing converted low-level policy: {weights_path}. Run "
          "`python tools/convert_g1_navigation_policy.py`."
      )
    with np.load(weights_path) as archive:
      self.mean = jp.asarray(archive["mean"])
      self.std = jp.asarray(archive["std"])
      self.weights = tuple(jp.asarray(archive[f"weight_{i}"]) for i in range(4))
      self.biases = tuple(jp.asarray(archive[f"bias_{i}"]) for i in range(4))
    if self.mean.shape != (98,) or self.std.shape != (98,):
      raise ValueError("low-level normalizer must contain 98 values")
    expected = ((512, 98), (256, 512), (128, 256), (29, 128))
    actual = tuple(weight.shape for weight in self.weights)
    if actual != expected:
      raise ValueError(
          f"unexpected low-level MLP shapes: {actual}, expected {expected}"
      )

  def __call__(self, observation: jax.Array) -> jax.Array:
    if observation.shape != (98,):
      raise ValueError(
          f"expected low-level observation (98,), got {observation.shape}"
      )
    hidden = (observation - self.mean) / jp.maximum(self.std, 1.0e-6)
    for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
      hidden = jax.nn.elu(weight @ hidden + bias)
    return self.weights[-1] @ hidden + self.biases[-1]


def apply_deploy_pd_to_model(model) -> None:
  """Mutates an MjModel to match the Unitree RL MjLab deploy controller."""
  names = tuple(model.actuator(i).name for i in range(model.nu))
  if names != JOINT_NAMES:
    raise ValueError(
        "MuJoCo actuator order differs from the deployed low-level policy:\n"
        f"model={names}\npolicy={JOINT_NAMES}"
    )
  stiffness = np.asarray(STIFFNESS)
  damping = np.asarray(DAMPING)
  model.actuator_gainprm[:, 0] = stiffness
  model.actuator_biasprm[:, 1] = -stiffness
  model.actuator_biasprm[:, 2] = -damping
  # Damping is supplied by actuator_biasprm.  Avoid double-counting the
  # original MuJoCo Playground joint damping.
  model.dof_damping[6:] = 0.0
