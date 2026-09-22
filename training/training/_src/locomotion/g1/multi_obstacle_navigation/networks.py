"""Dense Flax GAT actor and local-state critic for Brax PPO."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jp
from brax.training import distribution, networks
from brax.training.agents.ppo import networks as ppo_networks
from flax import linen


class DenseGatActor(linen.Module):
  """psi1/psi2/psi3 graph attention followed by an MLP policy head."""

  output_size: int
  num_nodes: int = 12
  node_dim: int = 9
  local_state_dim: int = 13
  embedding_dim: int = 16
  hidden_sizes: Sequence[int] = (256, 128, 64)

  @linen.compact
  def __call__(self, observation: jax.Array) -> jax.Array:
    graph_size = self.num_nodes * self.node_dim
    nodes = observation[..., :graph_size].reshape(
        observation.shape[:-1] + (self.num_nodes, self.node_dim)
    )
    labels = nodes[..., :3]
    position = nodes[..., 3:5]
    radius = nodes[..., 5]
    velocity = nodes[..., 6:8]
    valid = nodes[..., 8] > 0.5
    # Only the robot embedding is consumed by the policy head.  Since this is
    # a single message-passing layer, embeddings for obstacle and goal source
    # nodes cannot influence the robot and need not be computed.  Neighbor
    # slots contain the selected obstacles followed by the goal.
    labels_j = labels[..., 1:, :]
    labels_i = jp.broadcast_to(labels[..., :1, :], labels_j.shape)
    delta_position = position[..., 1:, :] - position[..., :1, :]
    center_distance = jp.linalg.norm(delta_position, axis=-1, keepdims=True)
    surface_distance = (
        center_distance
        - (radius[..., :1] + radius[..., 1:])[..., None]
    )
    delta_velocity = velocity[..., 1:, :] - velocity[..., :1, :]
    edge = jp.concatenate(
        (labels_i, labels_j, delta_position, surface_distance, delta_velocity),
        axis=-1,
    )

    message = linen.Dense(64, name="psi1_dense0")(edge)
    message = linen.relu(message)
    message = linen.Dense(self.embedding_dim, name="psi1_dense1")(message)

    logits = linen.Dense(16, name="psi2_dense0")(message)
    logits = linen.relu(logits)
    logits = linen.Dense(1, name="psi2_dense1")(logits)[..., 0]
    edge_valid = valid[..., :1] & valid[..., 1:]
    logits = jp.where(edge_valid, logits, -1.0e9)

    attention = jax.nn.softmax(logits, axis=-1) * edge_valid
    attention /= jp.maximum(jp.sum(attention, axis=-1, keepdims=True), 1.0e-8)

    encoded_message = linen.Dense(64, name="psi3_dense0")(message)
    encoded_message = linen.relu(encoded_message)
    encoded_message = linen.Dense(self.embedding_dim, name="psi3_dense1")(
        encoded_message
    )

    robot_embedding = jp.sum(
        attention[..., None] * encoded_message, axis=-2
    )
    local = observation[..., graph_size : graph_size + self.local_state_dim]
    hidden = jp.concatenate((robot_embedding, local), axis=-1)
    for index, size in enumerate(self.hidden_sizes):
      hidden = linen.Dense(size, name=f"policy_hidden_{index}")(hidden)
      hidden = linen.elu(hidden)
    return linen.Dense(self.output_size, name="policy_output")(hidden)


class SplitValueCritic(linen.Module):
  """One critic MLP with positive- and negative-return scalar outputs."""

  hidden_sizes: Sequence[int]
  kernel_init: linen.initializers.Initializer = (
      jax.nn.initializers.lecun_uniform()
  )

  @linen.compact
  def __call__(self, observation: jax.Array) -> jax.Array:
    hidden = observation
    for index, size in enumerate(self.hidden_sizes):
      hidden = linen.Dense(
          size,
          kernel_init=self.kernel_init,
          name=f"value_hidden_{index}",
      )(hidden)
      hidden = linen.elu(hidden)
    positive = linen.Dense(
        1, kernel_init=self.kernel_init, name="positive_value_head"
    )(hidden)
    negative = linen.Dense(
        1, kernel_init=self.kernel_init, name="negative_value_head"
    )(hidden)
    return jp.concatenate((positive, negative), axis=-1)


def make_gat_ppo_networks(
    observation_size,
    action_size: int,
    preprocess_observations_fn,
    *,
    num_nodes: int = 12,
    node_dim: int = 9,
    local_state_dim: int = 13,
    gat_embedding_dim: int = 16,
    policy_hidden_layer_sizes: Sequence[int] = (256, 128, 64),
    value_hidden_layer_sizes: Sequence[int] = (256, 256, 128),
    policy_obs_key: str = "state",
    value_obs_key: str = "privileged_state",
    split_value_critic: bool = False,
) -> ppo_networks.PPONetworks:
  """Builds PPO networks while preserving Brax's normalizer API."""
  action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size
  )
  policy_module = DenseGatActor(
      output_size=action_distribution.param_size,
      num_nodes=num_nodes,
      node_dim=node_dim,
      local_state_dim=local_state_dim,
      embedding_dim=gat_embedding_dim,
      hidden_sizes=policy_hidden_layer_sizes,
  )
  policy_obs_size = math.prod(observation_size[policy_obs_key])
  dummy_policy_obs = jp.zeros((1, policy_obs_size))

  def policy_init(key):
    return policy_module.init(key, dummy_policy_obs)

  def policy_apply(processor_params, policy_params, observation):
    value = (
        observation[policy_obs_key]
        if isinstance(observation, Mapping)
        else observation
    )
    # Navigation deliberately disables global observation normalization so
    # one-hot labels and validity masks retain their semantics.
    del processor_params
    return policy_module.apply(policy_params, value)

  policy_network = networks.FeedForwardNetwork(
      init=policy_init, apply=policy_apply
  )
  if split_value_critic:
    value_module = SplitValueCritic(value_hidden_layer_sizes)
    value_obs_size = math.prod(observation_size[value_obs_key])
    dummy_value_obs = jp.zeros((1, value_obs_size))

    def value_init(key):
      return value_module.init(key, dummy_value_obs)

    def value_apply(processor_params, value_params, observation):
      value = (
          observation[value_obs_key]
          if isinstance(observation, Mapping)
          else observation
      )
      if isinstance(observation, Mapping):
        value_processor_params = networks.normalizer_select(
            processor_params, value_obs_key
        )
      else:
        value_processor_params = processor_params
      value = preprocess_observations_fn(value, value_processor_params)
      return value_module.apply(value_params, value)

    value_network = networks.FeedForwardNetwork(
        init=value_init, apply=value_apply
    )
  else:
    value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=linen.elu,
        obs_key=value_obs_key,
    )
  return ppo_networks.PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=action_distribution,
  )
