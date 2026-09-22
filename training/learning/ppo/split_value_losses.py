"""PPO loss for DODGER with sign-decomposed value estimation."""

# The signature intentionally mirrors Brax's PPO loss configuration.
# ruff: noqa: PLR0917

from typing import Any, Tuple

import jax
import jax.numpy as jp
from brax.training import types
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks


def _explained_variance(target: jax.Array, prediction: jax.Array) -> jax.Array:
  target_variance = jp.var(target)
  residual_variance = jp.var(target - prediction)
  return 1.0 - residual_variance / (target_variance + 1.0e-8)


def compute_split_value_ppo_loss(
    params: ppo_losses.PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jax.Array,
    ppo_network: ppo_networks.PPONetworks,
    entropy_cost: float = 1.0e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    vf_coefficient: float = 0.5,
    clipping_epsilon_value: float | None = None,
) -> Tuple[jax.Array, types.Metrics]:
  """Computes one PPO loss using positive and negative value targets.

  The critic returns ``[..., 2]`` as ``[V_positive, V_negative]``.  Positive
  GAE uses DODGER's soft continuation, while negative GAE uses only genuine hard
  termination.  Their raw advantages are summed and normalized once for the
  actor.  The two squared value errors are averaged into one value loss.
  """
  action_distribution = ppo_network.parametric_action_distribution
  policy_apply = ppo_network.policy_network.apply
  value_apply = ppo_network.value_network.apply

  # Put time before batch: [B, T, ...] -> [T, B, ...].
  data = jax.tree_util.tree_map(lambda x: jp.swapaxes(x, 0, 1), data)
  policy_logits = policy_apply(
      normalizer_params, params.policy, data.observation
  )
  baseline = value_apply(normalizer_params, params.value, data.observation)
  terminal_observation = jax.tree_util.tree_map(
      lambda x: x[-1], data.next_observation
  )
  bootstrap_value = value_apply(
      normalizer_params, params.value, terminal_observation
  )
  if baseline.shape[-1] != 2 or bootstrap_value.shape[-1] != 2:
    raise ValueError("The split value critic must output exactly two values")

  positive_value = baseline[..., 0]
  negative_value = baseline[..., 1]
  positive_bootstrap = bootstrap_value[..., 0]
  negative_bootstrap = bootstrap_value[..., 1]

  state_extras = data.extras["state_extras"]
  truncation = state_extras["truncation"]
  hard_done = state_extras["dodger_hard_done"]
  dodger_delta = state_extras["dodger_delta"]
  positive_done = 1.0 - (1.0 - hard_done) * (1.0 - dodger_delta)
  positive_termination = positive_done * (1.0 - truncation)
  negative_termination = hard_done * (1.0 - truncation)
  positive_reward = state_extras["dodger_positive_reward"] * reward_scaling
  negative_reward = state_extras["dodger_negative_reward"] * reward_scaling

  positive_vs, positive_advantage = ppo_losses.compute_gae(
      truncation=truncation,
      termination=positive_termination,
      rewards=positive_reward,
      values=positive_value,
      bootstrap_value=positive_bootstrap,
      lambda_=gae_lambda,
      discount=discounting,
  )
  negative_vs, negative_advantage = ppo_losses.compute_gae(
      truncation=truncation,
      termination=negative_termination,
      rewards=negative_reward,
      values=negative_value,
      bootstrap_value=negative_bootstrap,
      lambda_=gae_lambda,
      discount=discounting,
  )

  raw_advantage = positive_advantage + negative_advantage
  advantage = raw_advantage
  if normalize_advantage:
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1.0e-8)

  target_log_prob = action_distribution.log_prob(
      policy_logits, data.extras["policy_extras"]["raw_action"]
  )
  behavior_log_prob = data.extras["policy_extras"]["log_prob"]
  probability_ratio = jp.exp(target_log_prob - behavior_log_prob)
  surrogate_loss_1 = probability_ratio * advantage
  surrogate_loss_2 = (
      jp.clip(probability_ratio, 1.0 - clipping_epsilon, 1.0 + clipping_epsilon)
      * advantage
  )
  policy_loss = -jp.mean(jp.minimum(surrogate_loss_1, surrogate_loss_2))

  def component_value_loss(
      target: jax.Array,
      prediction: jax.Array,
      old_prediction: jax.Array | None,
  ) -> jax.Array:
    squared_error = (target - prediction) ** 2
    if clipping_epsilon_value is not None:
      if old_prediction is None:
        raise ValueError("Clipped value loss requires rollout value estimates")
      clipped_prediction = old_prediction + jp.clip(
          prediction - old_prediction,
          -clipping_epsilon_value,
          clipping_epsilon_value,
      )
      clipped_error = (target - clipped_prediction) ** 2
      squared_error = jp.maximum(squared_error, clipped_error)
    return 0.5 * jp.mean(squared_error)

  old_value = data.extras["policy_extras"].get("value")
  old_positive = None if old_value is None else old_value[..., 0]
  old_negative = None if old_value is None else old_value[..., 1]
  positive_value_error = component_value_loss(
      positive_vs, positive_value, old_positive
  )
  negative_value_error = component_value_loss(
      negative_vs, negative_value, old_negative
  )
  positive_value_loss = 0.5 * positive_value_error * vf_coefficient
  negative_value_loss = 0.5 * negative_value_error * vf_coefficient
  value_loss = positive_value_loss + negative_value_loss

  entropy = jp.mean(action_distribution.entropy(policy_logits, rng))
  entropy_loss = -entropy_cost * entropy
  total_loss = policy_loss + value_loss + entropy_loss

  new_distribution = action_distribution.create_dist(policy_logits)
  if hasattr(new_distribution, "kl_divergence"):
    old_distribution = action_distribution.create_dist(
        data.extras["policy_extras"]["distribution_params"]
    )
    kl = jp.mean(new_distribution.kl_divergence(old_distribution))
  else:
    kl = jp.asarray(0.0)

  return total_loss, {
      "total_loss": total_loss,
      "policy_loss": policy_loss,
      "v_loss": value_loss,
      "v_loss_positive": positive_value_loss,
      "v_loss_negative": negative_value_loss,
      "entropy_loss": entropy_loss,
      "kl_mean": kl,
      "policy_dist_mean_std": jp.mean(new_distribution.scale),
      "policy_dist_max_std": jp.max(new_distribution.scale),
      "policy_dist_min_std": jp.min(new_distribution.scale),
      "policy_dist_mean_loc": jp.mean(new_distribution.loc),
      "policy_dist_max_loc": jp.max(new_distribution.loc),
      "policy_dist_min_loc": jp.min(new_distribution.loc),
      "value_decomposition/positive_value_mean": jp.mean(positive_value),
      "value_decomposition/negative_value_mean": jp.mean(negative_value),
      "value_decomposition/positive_explained_variance": _explained_variance(
          positive_vs, positive_value
      ),
      "value_decomposition/negative_explained_variance": _explained_variance(
          negative_vs, negative_value
      ),
      "value_decomposition/positive_advantage_mean": jp.mean(
          positive_advantage
      ),
      "value_decomposition/negative_advantage_mean": jp.mean(
          negative_advantage
      ),
      "value_decomposition/positive_advantage_std": jp.std(positive_advantage),
      "value_decomposition/negative_advantage_std": jp.std(negative_advantage),
  }
