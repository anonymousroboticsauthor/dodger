"""Exports a trained G1 high-level navigation actor as two ONNX models.

The GAT encoder accepts only the nodes available at deployment and produces the
robot embedding.  The policy head combines that embedding with the local state
and produces the deterministic high-level action.  Neither model contains
observation construction, the DPCBF/QP safety filter, the frozen low-level
velocity policy, or the robot PD controller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
import onnx
import torch
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from onnx.reference import ReferenceEvaluator
from torch import nn

from training._src.locomotion.g1.multi_obstacle_navigation import (
    networks as navigation_networks,
)


def _numbered_checkpoints(checkpoints_dir: Path) -> list[Path]:
  return [
      path
      for path in checkpoints_dir.iterdir()
      if path.is_dir()
      and path.name.isdigit()
      and (path / "ppo_network_config.json").is_file()
  ]


def resolve_checkpoint(path: Path) -> tuple[Path, Path]:
  """Resolves a run, checkpoints, or numbered checkpoint directory."""
  path = path.expanduser().resolve()
  if not path.exists():
    raise FileNotFoundError(path)

  if path.is_dir() and path.name.isdigit():
    checkpoints_dir = path.parent
    if checkpoints_dir.name != "checkpoints":
      raise ValueError(
          f"Numbered checkpoint must be inside a checkpoints directory: {path}"
      )
    if not (path / "ppo_network_config.json").is_file():
      raise FileNotFoundError(f"Incomplete checkpoint: {path}")
    return path, checkpoints_dir.parent

  checkpoints_dir = path if path.name == "checkpoints" else path / "checkpoints"
  if not checkpoints_dir.is_dir():
    raise FileNotFoundError(
        "Expected a training run directory, checkpoints directory, or "
        f"numbered checkpoint; got {path}"
    )
  candidates = _numbered_checkpoints(checkpoints_dir)
  if not candidates:
    raise FileNotFoundError(
        f"No completed numbered checkpoints found in {checkpoints_dir}"
    )
  return max(candidates, key=lambda item: int(item.name)), checkpoints_dir.parent


def _linear(in_features: int, out_features: int, params: Any) -> nn.Linear:
  layer = nn.Linear(in_features, out_features)
  with torch.no_grad():
    layer.weight.copy_(
        torch.from_numpy(np.asarray(params["kernel"])).transpose(0, 1)
    )
    layer.bias.copy_(torch.from_numpy(np.asarray(params["bias"])))
  return layer


class GatEncoder(nn.Module):
  """Robot-node GAT encoder with a dynamic, mask-free neighbor dimension."""

  def __init__(self, policy_params: Any):
    super().__init__()
    params = policy_params["params"]

    edge_dim = int(np.asarray(params["psi1_dense0"]["kernel"]).shape[0])
    if edge_dim != 11:
      raise ValueError(f"Expected an 11-dimensional DODGER edge, got {edge_dim}")
    self.psi1_dense0 = _linear(edge_dim, 64, params["psi1_dense0"])
    self.psi1_dense1 = _linear(64, 16, params["psi1_dense1"])
    self.psi2_dense0 = _linear(16, 16, params["psi2_dense0"])
    self.psi2_dense1 = _linear(16, 1, params["psi2_dense1"])
    self.psi3_dense0 = _linear(16, 64, params["psi3_dense0"])
    self.psi3_dense1 = _linear(64, 16, params["psi3_dense1"])

  def forward(self, nodes: torch.Tensor) -> torch.Tensor:
    labels = nodes[..., :3]
    position = nodes[..., 3:5]
    radius = nodes[..., 5]
    velocity = nodes[..., 6:8]

    labels_j = labels[..., 1:, :]
    labels_i = labels[..., :1, :].expand_as(labels_j)
    delta_position = position[..., 1:, :] - position[..., :1, :]
    center_distance = torch.linalg.vector_norm(
        delta_position, dim=-1, keepdim=True
    )
    surface_distance = center_distance - (
        radius[..., :1] + radius[..., 1:]
    ).unsqueeze(-1)
    delta_velocity = velocity[..., 1:, :] - velocity[..., :1, :]
    edge = torch.cat(
        (labels_i, labels_j, delta_position, surface_distance, delta_velocity),
        dim=-1,
    )

    message = self.psi1_dense1(torch.relu(self.psi1_dense0(edge)))
    logits = self.psi2_dense1(torch.relu(self.psi2_dense0(message)))[..., 0]
    attention = torch.softmax(logits, dim=-1)
    encoded = self.psi3_dense1(torch.relu(self.psi3_dense0(message)))
    return (attention.unsqueeze(-1) * encoded).sum(dim=-2)


class PolicyHead(nn.Module):
  """Policy MLP and physical action transform after graph encoding."""

  def __init__(
      self,
      policy_params: Any,
      *,
      action_size: int,
      acceleration_and_yaw_ranges: np.ndarray,
      alpha_nominal: float | None = None,
      alpha_range: tuple[float, float] | None = None,
  ):
    super().__init__()
    params = policy_params["params"]
    self.action_size = action_size
    hidden_names = sorted(
        (name for name in params if name.startswith("policy_hidden_")),
        key=lambda name: int(name.rsplit("_", maxsplit=1)[1]),
    )
    hidden_layers = []
    for name in hidden_names:
      kernel = np.asarray(params[name]["kernel"])
      hidden_layers.append(_linear(kernel.shape[0], kernel.shape[1], params[name]))
    self.hidden_layers = nn.ModuleList(hidden_layers)
    output_kernel = np.asarray(params["policy_output"]["kernel"])
    self.policy_output = _linear(
        output_kernel.shape[0], output_kernel.shape[1], params["policy_output"]
    )

    ranges = torch.as_tensor(acceleration_and_yaw_ranges, dtype=torch.float32)
    self.register_buffer("action_midpoint", ranges.mean(dim=-1))
    self.register_buffer(
        "action_half_range", 0.5 * (ranges[:, 1] - ranges[:, 0])
    )
    if action_size == 4:
      if alpha_nominal is None or alpha_range is None:
        raise ValueError("Four-action checkpoints require alpha configuration")
      self.register_buffer(
          "alpha_nominal", torch.tensor(alpha_nominal, dtype=torch.float32)
      )
      self.register_buffer(
          "alpha_range", torch.tensor(alpha_range, dtype=torch.float32)
      )

  def forward(
      self, robot_embedding: torch.Tensor, local_state: torch.Tensor
  ) -> torch.Tensor:
    hidden = torch.cat((robot_embedding, local_state), dim=-1)
    for layer in self.hidden_layers:
      hidden = torch.nn.functional.elu(layer(hidden))
    distribution_parameters = self.policy_output(hidden)
    normalized_action = torch.tanh(
        distribution_parameters[..., : self.action_size]
    )
    physical_action = (
        self.action_midpoint
        + self.action_half_range * normalized_action[..., :3]
    )
    if self.action_size == 3:
      return physical_action

    normalized_alpha = normalized_action[..., 3]
    alpha = self.alpha_nominal + torch.where(
        normalized_alpha >= 0.0,
        normalized_alpha * (self.alpha_range[1] - self.alpha_nominal),
        normalized_alpha * (self.alpha_nominal - self.alpha_range[0]),
    )
    return torch.cat((physical_action, alpha.unsqueeze(-1)), dim=-1)


def _physical_action_numpy(
    normalized_action: np.ndarray,
    ranges: np.ndarray,
    *,
    alpha_nominal: float | None,
    alpha_range: tuple[float, float] | None,
) -> np.ndarray:
  physical = ranges.mean(axis=-1) + 0.5 * (
      ranges[:, 1] - ranges[:, 0]
  ) * normalized_action[..., :3]
  if normalized_action.shape[-1] == 3:
    return physical
  assert alpha_nominal is not None and alpha_range is not None
  raw_alpha = normalized_action[..., 3]
  alpha = alpha_nominal + np.where(
      raw_alpha >= 0.0,
      raw_alpha * (alpha_range[1] - alpha_nominal),
      raw_alpha * (alpha_nominal - alpha_range[0]),
  )
  return np.concatenate((physical, alpha[..., None]), axis=-1)


def _add_metadata(model: onnx.ModelProto, metadata: dict[str, str]) -> None:
  for key, value in metadata.items():
    entry = model.metadata_props.add()
    entry.key = key
    entry.value = value


def export_policy(
    checkpoint_arg: Path, output_dir: Path | None
) -> tuple[Path, Path]:
  checkpoint_path, run_dir = resolve_checkpoint(checkpoint_arg)
  with (checkpoint_path / "ppo_network_config.json").open(encoding="utf-8") as fp:
    network_config = json.load(fp)
  config_path = run_dir / "checkpoints" / "config.json"
  if not config_path.is_file():
    raise FileNotFoundError(f"Environment config not found: {config_path}")
  with config_path.open(encoding="utf-8") as fp:
    env_config = json.load(fp)

  action_size = int(network_config["action_size"])
  if action_size not in (3, 4):
    raise ValueError(f"Expected a 3- or 4-action navigation actor, got {action_size}")
  factory = network_config["network_factory_kwargs"]
  num_nodes = int(factory["num_nodes"])
  node_dim = int(factory["node_dim"])
  if node_dim != 9:
    raise ValueError(
        "Expected 9 training node features (including valid mask), got "
        f"{node_dim}"
    )
  local_state_dim = int(factory["local_state_dim"])
  observation_size = int(
      network_config["observation_size"][factory["policy_obs_key"]]["shape"][0]
  )
  expected_observation_size = num_nodes * node_dim + local_state_dim
  if observation_size != expected_observation_size:
    raise ValueError(
        f"Observation size {observation_size} does not match GAT layout "
        f"{expected_observation_size}"
    )

  robot_config = env_config["robot"]
  action_ranges = np.asarray(
      (
          robot_config["sagittal_acceleration_range"],
          robot_config["lateral_acceleration_range"],
          robot_config["yaw_rate_range"],
      ),
      dtype=np.float32,
  )
  alpha_nominal = None
  alpha_range = None
  if action_size == 4:
    dpcbf_config = env_config["dpcbf"]
    alpha_nominal = float(dpcbf_config["alpha_nominal"])
    alpha_range = tuple(map(float, dpcbf_config["alpha_range"]))

  params = ppo_checkpoint.load(checkpoint_path)
  policy_params = params[1]
  edge_dim = int(
      np.asarray(policy_params["params"]["psi1_dense0"]["kernel"]).shape[0]
  )
  if edge_dim != 11:
    raise ValueError(f"Expected an 11-dimensional DODGER edge, got {edge_dim}")
  gat_encoder = GatEncoder(policy_params).eval()
  policy_head = PolicyHead(
      policy_params,
      action_size=action_size,
      acceleration_and_yaw_ranges=action_ranges,
      alpha_nominal=alpha_nominal,
      alpha_range=alpha_range,
  ).eval()

  if output_dir is None:
    output_dir = run_dir / "onnx" / f"checkpoint_{checkpoint_path.name}"
  output_dir = output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  gat_path = output_dir / "gat_encoder.onnx"
  policy_head_path = output_dir / "policy_head.onnx"

  # Deployment supplies only actual nodes: robot first, followed by detected
  # obstacles and the goal in any order.  The validity column used by
  # fixed-shape JAX training is absent.
  example_nodes = torch.zeros((1, num_nodes, node_dim - 1), dtype=torch.float32)
  example_nodes[..., 0, 0] = 1.0
  example_nodes[..., 1:-1, 1] = 1.0
  example_nodes[..., -1, 2] = 1.0
  torch.onnx.export(
      gat_encoder,
      (example_nodes,),
      gat_path,
      input_names=["nodes"],
      output_names=["robot_embedding"],
      dynamic_axes={
          "nodes": {0: "batch", 1: "detected_nodes"},
          "robot_embedding": {0: "batch"},
      },
      opset_version=17,
      do_constant_folding=True,
      dynamo=False,
  )
  example_embedding = torch.zeros(
      (1, int(factory["gat_embedding_dim"])), dtype=torch.float32
  )
  example_local_state = torch.zeros((1, local_state_dim), dtype=torch.float32)
  torch.onnx.export(
      policy_head,
      (example_embedding, example_local_state),
      policy_head_path,
      input_names=["robot_embedding", "local_state"],
      output_names=["policy_action"],
      dynamic_axes={
          "robot_embedding": {0: "batch"},
          "local_state": {0: "batch"},
          "policy_action": {0: "batch"},
      },
      opset_version=17,
      do_constant_folding=True,
      dynamo=False,
  )

  gat_model = onnx.load(gat_path)
  common_metadata = {
      "checkpoint_step": str(int(checkpoint_path.name)),
      "policy_hz": str(env_config["policy_hz"]),
      "action_frame": "body",
      "low_level_policy_included": "false",
      "pd_gains_included": "false",
  }
  _add_metadata(
      gat_model,
      {
          **common_metadata,
          "model_type": "g1_navigation_gat_encoder",
          "input": "body-frame nodes [robot,(obstacles and goal in any order)]",
          "input_node_features": (
              "robot_label,obstacle_label,goal_label,relative_x,relative_y,"
              "radius,relative_vx,relative_vy"
          ),
          "input_valid_mask_included": "false",
          "maximum_nodes_during_training": str(num_nodes),
          "output_embedding_dim": str(int(factory["gat_embedding_dim"])),
          "gat_edge_dimension": str(edge_dim),
          "gat_edge_distance": "surface_clearance",
      },
  )
  onnx.checker.check_model(gat_model)
  onnx.save(gat_model, gat_path)

  head_model = onnx.load(policy_head_path)
  _add_metadata(
      head_model,
      {
          **common_metadata,
          "model_type": "g1_navigation_policy_head",
          "input_robot_embedding_dim": str(int(factory["gat_embedding_dim"])),
          "input_local_state_dim": str(local_state_dim),
          "output": "[sagittal_acceleration,lateral_acceleration,yaw_rate]"
          + (",alpha" if action_size == 4 else ""),
      },
  )
  onnx.checker.check_model(head_model)
  onnx.save(head_model, policy_head_path)

  rng = np.random.default_rng(0)
  test_observation = rng.normal(size=(4, observation_size)).astype(np.float32)
  graph_size = num_nodes * node_dim
  test_nodes = test_observation[..., :graph_size].reshape(
      4, num_nodes, node_dim
  )
  test_nodes[..., 8] = 1.0
  test_observation[..., :graph_size] = test_nodes.reshape(4, graph_size)
  compact_nodes = test_nodes[..., :8]
  local_state = test_observation[..., graph_size:]
  with torch.no_grad():
    torch_embedding = gat_encoder(torch.from_numpy(compact_nodes)).numpy()
    torch_output = policy_head(
        torch.from_numpy(torch_embedding), torch.from_numpy(local_state)
    ).numpy()
  flax_module = navigation_networks.DenseGatActor(
      output_size=2 * action_size,
      num_nodes=num_nodes,
      node_dim=node_dim,
      local_state_dim=local_state_dim,
      embedding_dim=int(factory["gat_embedding_dim"]),
      hidden_sizes=tuple(factory["policy_hidden_layer_sizes"]),
  )
  with jax.default_matmul_precision("highest"):
    flax_parameters = np.asarray(
        flax_module.apply(policy_params, jp.asarray(test_observation))
    )
  normalized_action = np.tanh(flax_parameters[..., :action_size])
  flax_output = _physical_action_numpy(
      normalized_action,
      action_ranges,
      alpha_nominal=alpha_nominal,
      alpha_range=alpha_range,
  )
  np.testing.assert_allclose(torch_output, flax_output, rtol=2.0e-5, atol=2.0e-5)
  reference_embedding = ReferenceEvaluator(gat_model).run(
      ["robot_embedding"], {"nodes": compact_nodes}
  )[0]
  reference_output = ReferenceEvaluator(head_model).run(
      ["policy_action"],
      {
          "robot_embedding": reference_embedding,
          "local_state": local_state,
      },
  )[0]
  np.testing.assert_allclose(
      reference_output, flax_output, rtol=3.0e-5, atol=3.0e-5
  )

  print(f"Resolved checkpoint: {checkpoint_path}")
  print(f"Exported GAT encoder: {gat_path}")
  print(f"Exported policy head: {policy_head_path}")
  print("GAT input shape: [batch, detected_nodes, 8]")
  print(
      "Policy head input shapes: "
      f"[batch, {int(factory['gat_embedding_dim'])}], "
      f"[batch, {local_state_dim}]"
  )
  print(f"Output shape: [batch, {action_size}]")
  print("Verified chained PyTorch, Flax/JAX, and ONNX outputs.")
  return gat_path, policy_head_path


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--checkpoint",
      type=Path,
      required=True,
      help=(
          "Training run directory, checkpoints directory, or numbered "
          "checkpoint directory. The latest completed checkpoint is selected."
      ),
  )
  parser.add_argument(
      "--output_dir",
      type=Path,
      default=None,
      help=(
          "Directory for gat_encoder.onnx and policy_head.onnx "
          "(default: <run>/onnx/checkpoint_<step>/)."
      ),
  )
  args = parser.parse_args()
  export_policy(args.checkpoint, args.output_dir)


if __name__ == "__main__":
  main()
