# Copyright 2026 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

# This file intentionally stays close to the vendored Brax implementation.
# ruff: noqa: E731, I001, PLR0917

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import logger as metric_logger
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import optimizer as ppo_optimizer
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from learning.ppo import split_value_losses

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""

  optimizer_state: optax.OptState
  params: ppo_losses.PPONetworkParams
  normalizer_params: running_statistics.RunningStatisticsState
  env_steps: types.UInt64
  kl_policy_update_scale: jax.Array


def _apply_transactional_kl_guard(
    optimizer_state_before,
    params_before,
    candidate_optimizer_state,
    candidate_params,
    committed_kl,
    candidate_kl,
    hard_kl_limit,
    policy_update_scale,
    backoff_factor,
    recovery_factor,
    minimum_update_scale,
):
  """Commits a safe PPO pass and adapts the next policy-update scale."""
  invalid_candidate = jnp.logical_not(jnp.isfinite(candidate_kl))
  limit_exceeded = jnp.logical_or(
      invalid_candidate, candidate_kl > hard_kl_limit
  )
  rollback = limit_exceeded
  accept_candidate = jnp.logical_not(limit_exceeded)

  # A lax.cond over the complete model and Adam trees creates a dynamic GPU
  # branch inside nested PPO scans and is the common suspect in the observed
  # Blackwell Xid 8 failures.  Both states already exist here, so elementwise
  # selection provides the same commit without accelerator control flow.
  optimizer_state = jax.tree_util.tree_map(
      lambda candidate, previous: jnp.where(
          accept_candidate, candidate, previous
      ),
      candidate_optimizer_state,
      optimizer_state_before,
  )
  params = jax.tree_util.tree_map(
      lambda candidate, previous: jnp.where(
          accept_candidate, candidate, previous
      ),
      candidate_params,
      params_before,
  )
  committed_kl = jnp.where(accept_candidate, candidate_kl, committed_kl)
  next_update_scale = jnp.where(
      rollback,
      jnp.maximum(policy_update_scale * backoff_factor, minimum_update_scale),
      jnp.minimum(policy_update_scale * recovery_factor, 1.0),
  )
  return optimizer_state, params, committed_kl, rollback, next_update_scale


def _scale_policy_update(params_update, policy_update_scale):
  """Scales only the actor update, leaving the critic update unchanged."""
  return params_update.replace(
      policy=jax.tree_util.tree_map(
          lambda update: update * policy_update_scale,
          params_update.policy,
      )
  )


def _unpmap(v):
  # Avoid degraded performance under the new jax.pmap.
  return jax.tree_util.tree_map(
      lambda x: x.addressable_shards[0].data.squeeze(0), v
  )


def _strip_weak_type(tree):
  # brax user code is sometimes ambiguous about weak_type.  in order to
  # avoid extra jit recompilations we strip all weak types from user input
  def f(leaf):
    leaf = jnp.asarray(leaf)
    return jnp.astype(leaf, leaf.dtype)

  return jax.tree_util.tree_map(f, tree)


def _maybe_wrap_env(
    env: envs.Env,
    wrap_env: bool,
    num_envs: int,
    episode_length: Optional[int],
    action_repeat: int,
    device_count: int,
    key_env: PRNGKey,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):
  """Wraps the environment for training/eval if wrap_env is True."""
  if not wrap_env:
    return env
  if episode_length is None:
    raise ValueError('episode_length must be specified in ppo.train')
  v_randomization_fn = None
  if randomization_fn is not None:
    randomization_batch_size = num_envs // device_count
    # all devices gets the same randomization rng
    randomization_rng = jax.random.split(key_env, randomization_batch_size)
    v_randomization_fn = functools.partial(
        randomization_fn, rng=randomization_rng
    )
  if wrap_env_fn is not None:
    wrap_for_training = wrap_env_fn
  else:
    wrap_for_training = envs.training.wrap
  env = wrap_for_training(
      env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=v_randomization_fn,
  )  # pytype: disable=wrong-keyword-args
  return env


def _random_translate_pixels(
    obs: Mapping[str, jax.Array], key: PRNGKey
) -> Mapping[str, jax.Array]:
  """Apply random translations to B x T x ... pixel observations.

  The same shift is applied across the unroll_length (T) dimension.

  Args:
    obs: a dictionary of observations
    key: a PRNGKey

  Returns:
    A dictionary of observations with translated pixels
  """

  @jax.vmap
  def rt_all_views(
      ub_obs: Mapping[str, jax.Array], key: PRNGKey
  ) -> Mapping[str, jax.Array]:
    # Expects dictionary of unbatched observations.
    def rt_view(
        img: jax.Array, padding: int, key: PRNGKey
    ) -> jax.Array:  # TxHxWxC
      # Randomly translates a set of pixel inputs.
      # Adapted from
      # https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/agents/drq/augmentations.py
      crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
      zero = jnp.zeros((1,), dtype=jnp.int32)
      crop_from = jnp.concatenate([zero, crop_from, zero])
      padded_img = jnp.pad(
          img,
          ((0, 0), (padding, padding), (padding, padding), (0, 0)),
          mode='edge',
      )
      return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

    out = {}
    for k_view, v_view in ub_obs.items():
      if k_view.startswith('pixels/'):
        key, key_shift = jax.random.split(key)
        out[k_view] = rt_view(v_view, 4, key_shift)
    return {**ub_obs, **out}

  bdim = next(iter(obs.items()), None)[1].shape[0]
  keys = jax.random.split(key, bdim)
  obs = rt_all_views(obs, keys)
  return obs


def _remove_pixels(
    obs: Union[jnp.ndarray, Mapping[str, jax.Array]],
) -> Union[jnp.ndarray, Mapping[str, jax.Array]]:
  """Removes pixel observations from the observation dict."""
  if not isinstance(obs, Mapping):
    return obs
  return {k: v for k, v in obs.items() if not k.startswith('pixels/')}


def train(
    environment: envs.Env,
    num_timesteps: int,
    max_devices_per_host: Optional[int] = None,
    # high-level control flow
    wrap_env: bool = True,
    vision: bool = False,
    augment_pixels: bool = False,
    # environment wrapper
    num_envs: int = 1,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    # ppo params
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    normalize_observations_std_eps: float = 0.0,
    normalize_observations_mode: str = 'welford',
    normalize_until_count: Optional[int] = None,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    clipping_epsilon_value: float | None = None,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = None,
    normalize_advantage: bool = True,
    vf_loss_coefficient: float = 0.5,
    bootstrap_on_timeout: bool = False,
    use_distributional_critic: bool = False,
    use_split_value_critic: bool = False,
    desired_kl: float = 0.01,
    enable_kl_limit: bool = False,
    hard_kl_limit: float = 0.2,
    kl_backoff_factor: float = 0.5,
    kl_recovery_factor: float = 1.05,
    kl_min_policy_update_scale: float = 1.0 / 64.0,
    learning_rate_schedule: Optional[
        Union[str, ppo_optimizer.LRSchedule]
    ] = None,
    learning_rate_schedule_min_lr: float = 1e-5,
    learning_rate_schedule_max_lr: float = 1e-2,
    network_factory: types.NetworkFactory[
        ppo_networks.PPONetworks
    ] = ppo_networks.make_ppo_networks,
    seed: int = 0,
    use_pmap_on_reset: bool = True,
    # eval
    num_evals: int = 1,
    eval_env: Optional[envs.Env] = None,
    num_eval_envs: int = 128,
    deterministic_eval: bool = False,
    # training metrics
    log_training_metrics: bool = False,
    training_metrics_steps: Optional[int] = None,
    # callbacks
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    # checkpointing
    save_checkpoint_path: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    restore_params: Optional[Any] = None,
    restore_value_fn: bool = True,
    run_evals: bool = True,
):
  """PPO training.

  Args:
    environment: the environment to train
    num_timesteps: the total number of environment steps to use during training
    max_devices_per_host: maximum number of chips to use per host process
    wrap_env: If True, wrap the environment for training. Otherwise use the
      environment as is.
    vision: whether to use vision observations.
    augment_pixels: whether to add image augmentation to pixel inputs
    num_envs: the number of parallel environments to use for rollouts
      NOTE `num_envs` must be divisible by the total number of chips since each
        chip gets `num_envs // total_number_of_chips` environments to roll out
      NOTE `batch_size * num_minibatches` must be divisible by `num_envs` since
        data generated by `num_envs` parallel envs gets used for gradient
        updates over `num_minibatches` of data, where each minibatch has a
        leading dimension of `batch_size`
    episode_length: the length of an environment episode
    action_repeat: the number of timesteps to repeat an action
    wrap_env_fn: a custom function that wraps the environment for training. If
      not specified, the environment is wrapped with the default training
      wrapper.
    randomization_fn: a user-defined callback function that generates randomized
      environments
    learning_rate: learning rate for ppo loss
    entropy_cost: entropy reward for ppo loss, higher values increase entropy of
      the policy
    discounting: discounting rate
    unroll_length: the number of timesteps to unroll in each environment. The
      PPO loss is computed over `unroll_length` timesteps
    batch_size: the batch size for each minibatch SGD step
    num_minibatches: the number of times to run the SGD step, each with a
      different minibatch with leading dimension of `batch_size`
    num_updates_per_batch: the number of times to run the gradient update over
      all minibatches before doing a new environment rollout
    num_resets_per_eval: the number of environment resets to run between each
      eval. The environment resets occur on the host
    normalize_observations: whether to normalize observations
    normalize_observations_std_eps: small value added to the standard deviation
      for obs normalization to improve numerical stability
    normalize_observations_mode: method to use for running statistics, welford
      is the default, but ema is more numerically stable for long training runs
    normalize_until_count: the number of environment steps to normalize
      observations until
    reward_scaling: float scaling for reward
    clipping_epsilon: clipping epsilon for PPO loss
    clipping_epsilon_value: Value function loss clipping epsilon
    gae_lambda: General advantage estimation lambda
    max_grad_norm: gradient clipping norm value. If None, no clipping is done
    normalize_advantage: whether to normalize advantage estimate
    vf_loss_coefficient: Coefficient for value function loss.
    bootstrap_on_timeout: if True, bootstrap value on time_out steps using
      reward += gamma * V(s) * time_out. Environments should set
      state.info['time_out'] = 1.0 and done=True for steps where the episode
      ends due to a time_out.
    use_distributional_critic: whether to use a distributional critic
    use_split_value_critic: whether the value network predicts positive and
      negative DODGER returns with two outputs and one averaged value loss.
    desired_kl: Desired KL divergence for adaptive KL divergence learning rate
      schedule.
    enable_kl_limit: whether to transactionally reject a complete PPO update
      pass when its full-rollout KL exceeds `hard_kl_limit`. Rejected passes
      reduce the policy-update scale and the remaining passes retry from the
      last accepted parameters.
    hard_kl_limit: hard full-rollout mean-KL threshold.  This is independent of
      `desired_kl` and does not enable adaptive learning-rate control.
    kl_backoff_factor: multiplier applied to the policy-update scale after a
      rejected KL candidate.
    kl_recovery_factor: multiplier applied after an accepted KL candidate,
      capped at one.
    kl_min_policy_update_scale: minimum policy-update scale used by the KL
      guard. The critic continues to use its configured learning rate.
    learning_rate_schedule: Learning rate schedule for the optimizer.
    learning_rate_schedule_min_lr: Minimum learning rate for adaptive KL
      learning rate schedule.
    learning_rate_schedule_max_lr: Maximum learning rate for adaptive KL
      learning rate schedule.
    network_factory: function that generates networks for policy and value
      functions
    seed: random seed
    num_evals: the number of evals to run during the entire training run.
      Increasing the number of evals increases total training time
    eval_env: an optional environment for eval only, defaults to `environment`
    num_eval_envs: the number of envs to use for evluation. Each env will run 1
      episode, and all envs run in parallel during eval.
    deterministic_eval: whether to run the eval with a deterministic policy
    log_training_metrics: whether to log training metrics and callback to
      progress_fn
    training_metrics_steps: the number of environment steps between logging
      training metrics
    progress_fn: a user-defined callback function for reporting/plotting metrics
    policy_params_fn: a user-defined callback function that can be used for
      saving custom policy checkpoints or creating policy rollouts and videos
    save_checkpoint_path: the path used to save checkpoints. If None, no
      checkpoints are saved.
    restore_checkpoint_path: the path used to restore previous model params
    restore_params: raw network parameters to restore the TrainingState from.
      These override `restore_checkpoint_path`. These paramaters can be obtained
      from the return values of ppo.train().
    restore_value_fn: whether to restore the value function from the checkpoint
      or use a random initialization
    run_evals: if True, use the evaluator num_eval times to collect distinct
      eval rollouts. If False, num_eval_envs and eval_env are ignored.
      progress_fn is then expected to use training_metrics.
    use_pmap_on_reset: default to True. if True, use pmap instead of vmap for
      env.reset across devices.

  Returns:
    Tuple of (make_policy function, network params, metrics)
  """
  assert batch_size * num_minibatches % num_envs == 0

  if enable_kl_limit and hard_kl_limit <= 0.0:
    raise ValueError(
        'hard_kl_limit must be positive when enable_kl_limit is True'
    )
  if not 0.0 < kl_backoff_factor < 1.0:
    raise ValueError('kl_backoff_factor must be in (0, 1)')
  if kl_recovery_factor < 1.0:
    raise ValueError('kl_recovery_factor must be at least 1')
  if not 0.0 < kl_min_policy_update_scale <= 1.0:
    raise ValueError('kl_min_policy_update_scale must be in (0, 1]')
  if use_split_value_critic and use_distributional_critic:
    raise ValueError(
        'Split-value and distributional critics cannot be enabled together'
    )
  if use_split_value_critic and bootstrap_on_timeout:
    raise ValueError(
        'bootstrap_on_timeout is not implemented for the split-value critic'
    )

  if vision and action_repeat != 1:
    raise ValueError(
        "Implement action_repeat using PipelineEnv's _n_frames to avoid"
        ' unnecessary rendering!'
    )

  xt = time.time()

  process_count = jax.process_count()
  process_id = jax.process_index()
  local_device_count = jax.local_device_count()
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d',
      jax.device_count(),
      process_count,
      process_id,
      local_device_count,
      local_devices_to_use,
  )
  device_count = local_devices_to_use * process_count

  # The number of environment steps executed for every training step.
  env_step_per_training_step = (
      batch_size * unroll_length * num_minibatches * action_repeat
  )
  num_evals_after_init = max(num_evals - 1, 1)
  # The number of training_step calls per training_epoch call.
  # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
  #                                 num_resets_per_eval))
  num_training_steps_per_epoch = np.ceil(
      num_timesteps
      / (
          num_evals_after_init
          * env_step_per_training_step
          * max(num_resets_per_eval, 1)
      )
  ).astype(int)

  key = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(key)
  del key
  local_key = jax.random.fold_in(local_key, process_id)
  local_key, key_env, eval_key = jax.random.split(local_key, 3)
  # key_networks should be global, so that networks are initialized the same
  # way for different processes.
  key_policy, key_value = jax.random.split(global_key)
  del global_key

  assert num_envs % device_count == 0

  env = _maybe_wrap_env(
      environment,
      wrap_env,
      num_envs,
      episode_length,
      action_repeat,
      device_count,
      key_env,
      wrap_env_fn,
      randomization_fn,
  )

  def reset_fn_donated_env_state(env_state_donated, key_envs):
    return env.reset(key_envs)

  key_envs = jax.random.split(key_env, num_envs // process_count)
  key_envs = jnp.reshape(
      key_envs, (local_devices_to_use, -1) + key_envs.shape[1:]
  )
  if local_devices_to_use > 1 or use_pmap_on_reset:
    reset_fn_ = jax.pmap(env.reset, axis_name=_PMAP_AXIS_NAME)
    env_state = reset_fn_(key_envs)
    reset_fn = jax.pmap(
        reset_fn_donated_env_state,
        axis_name=_PMAP_AXIS_NAME,
        donate_argnums=(0,),
    )
  else:
    reset_fn_ = jax.jit(jax.vmap(env.reset))
    env_state = reset_fn_(key_envs)
    reset_fn = jax.jit(
        reset_fn_donated_env_state, donate_argnums=(0,), keep_unused=True
    )

  # Discard the batch axes over devices and envs.
  obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)

  normalize = lambda x, y: x
  if normalize_observations:
    normalize = running_statistics.normalize
  if use_distributional_critic and clipping_epsilon_value is None:
    raise AssertionError(
        'clipping_epsilon_value must not be None when '
        'use_distributional_critic=True (it serves as kappa for quantile '
        'Huber loss)'
    )

  ppo_network = network_factory(
      obs_shape, env.action_size, preprocess_observations_fn=normalize
  )
  make_policy = ppo_networks.make_inference_fn(
      ppo_network,
      compute_value=bootstrap_on_timeout or clipping_epsilon_value is not None,
      use_distributional_critic=use_distributional_critic,
  )

  # Optimizer.
  base_optimizer = optax.adam(learning_rate=learning_rate)
  lr_schedule = learning_rate_schedule or ppo_optimizer.LRSchedule.NONE
  lr_schedule = ppo_optimizer.LRSchedule(lr_schedule)
  lr_is_adaptive_kl = lr_schedule == ppo_optimizer.LRSchedule.ADAPTIVE_KL
  if lr_is_adaptive_kl:
    base_optimizer = optax.inject_hyperparams(optax.adam)(
        learning_rate=learning_rate
    )
  if max_grad_norm is not None:
    # TODO(btaba): Move gradient clipping to `training/gradients.py`.
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        base_optimizer,
    )
  else:
    optimizer = base_optimizer

  if use_split_value_critic:
    loss_fn = functools.partial(
        split_value_losses.compute_split_value_ppo_loss,
        ppo_network=ppo_network,
        entropy_cost=entropy_cost,
        discounting=discounting,
        reward_scaling=reward_scaling,
        gae_lambda=gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage,
        vf_coefficient=vf_loss_coefficient,
        clipping_epsilon_value=clipping_epsilon_value,
    )
  else:
    loss_fn = functools.partial(
        ppo_losses.compute_ppo_loss,
        ppo_network=ppo_network,
        entropy_cost=entropy_cost,
        discounting=discounting,
        reward_scaling=reward_scaling,
        gae_lambda=gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage,
        vf_coefficient=vf_loss_coefficient,
        clipping_epsilon_value=clipping_epsilon_value,
        use_distributional_critic=use_distributional_critic,
    )

  loss_and_pgrad_fn = gradients.loss_and_pgrad(
      loss_fn, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
  )

  def full_rollout_policy_kl(
      params: ppo_losses.PPONetworkParams,
      normalizer_params: running_statistics.RunningStatisticsState,
      observations,
      behavior_distribution_params,
  ) -> jax.Array:
    """Returns mean candidate-to-behavior KL over every rollout sample."""

    def minibatch_kl(inputs) -> jax.Array:
      observation, behavior_params = inputs
      policy_logits = ppo_network.policy_network.apply(
          normalizer_params, params.policy, observation
      )
      action_distribution = ppo_network.parametric_action_distribution
      candidate_distribution = action_distribution.create_dist(policy_logits)
      behavior_distribution = action_distribution.create_dist(behavior_params)
      if hasattr(candidate_distribution, 'kl_divergence'):
        return jnp.mean(
            candidate_distribution.kl_divergence(behavior_distribution)
        )
      return jnp.asarray(0.0, dtype=jnp.float32)

    kl_mean = jnp.mean(
        jax.lax.map(minibatch_kl, (observations, behavior_distribution_params))
    )
    return jax.lax.pmean(kl_mean, axis_name=_PMAP_AXIS_NAME)

  steps_between_logging = training_metrics_steps or env_step_per_training_step
  metrics_aggregator = metric_logger.EpisodeMetricsLogger(
      steps_between_logging=steps_between_logging,
      progress_fn=progress_fn,
  )

  def minibatch_step(
      carry,
      data: types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState,
  ):
    optimizer_state, params, key, policy_update_scale = carry
    key, key_loss = jax.random.split(key)
    (_, metrics), grads = loss_and_pgrad_fn(
        params, normalizer_params, data, key_loss
    )

    if lr_is_adaptive_kl:
      kl_mean = metrics['kl_mean']
      kl_mean = jax.lax.pmean(kl_mean, axis_name=_PMAP_AXIS_NAME)
      optimizer_state, lr = ppo_optimizer.adaptive_kl_learning_rate(
          optimizer_state,
          kl_mean,
          desired_kl,
          min_learning_rate=learning_rate_schedule_min_lr,
          max_learning_rate=learning_rate_schedule_max_lr,
      )
    else:
      lr = jnp.array(learning_rate)

    params_update, optimizer_state = optimizer.update(grads, optimizer_state)
    params_update = _scale_policy_update(params_update, policy_update_scale)
    params = optax.apply_updates(params, params_update)
    metrics['learning_rate'] = lr * policy_update_scale
    metrics['kl_policy_update_scale'] = policy_update_scale
    # Pass-level values are filled after validating the candidate parameters.
    metrics['kl_candidate'] = metrics['kl_mean']
    metrics['kl_limit_triggered'] = jnp.asarray(0.0, dtype=jnp.float32)
    metrics['kl_update_skipped'] = jnp.asarray(0.0, dtype=jnp.float32)
    metrics['effective_minibatch_update'] = jnp.asarray(1.0, dtype=jnp.float32)

    return (optimizer_state, params, key, policy_update_scale), metrics

  def sgd_step(
      carry,
      unused_t,
      data: types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState,
  ):
    optimizer_state, params, key, committed_kl, policy_update_scale = carry
    optimizer_state_before = optimizer_state
    params_before = params
    key, key_perm, key_grad = jax.random.split(key, 3)

    if augment_pixels:
      key, key_rt = jax.random.split(key)
      r_translate = functools.partial(_random_translate_pixels, key=key_rt)
      data = types.Transition(
          observation=r_translate(
              data.observation
          ),  # pytype: disable=wrong-arg-types
          action=data.action,
          reward=data.reward,
          discount=data.discount,
          next_observation=r_translate(
              data.next_observation
          ),  # pytype: disable=wrong-arg-types
          extras=data.extras,
      )

    def convert_data(x: jnp.ndarray):
      x = jax.random.permutation(key_perm, x)
      x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
      return x

    shuffled_data = jax.tree_util.tree_map(convert_data, data)
    (optimizer_state, params, _, _), metrics = jax.lax.scan(
        functools.partial(minibatch_step, normalizer_params=normalizer_params),
        (optimizer_state, params, key_grad, policy_update_scale),
        shuffled_data,
        length=num_minibatches,
    )

    if enable_kl_limit:
      # Always evaluate the candidate.  Avoiding data-dependent conditionals
      # keeps the complete PPO update path static on GPU. A rejected pass is
      # discarded transactionally, then the next pass retries at a lower
      # policy-update scale.
      candidate_kl = full_rollout_policy_kl(
          params,
          normalizer_params,
          shuffled_data.observation,
          shuffled_data.extras['policy_extras']['distribution_params'],
      )
      (
          optimizer_state,
          params,
          committed_kl,
          rollback,
          policy_update_scale,
      ) = _apply_transactional_kl_guard(
          optimizer_state_before,
          params_before,
          optimizer_state,
          params,
          committed_kl,
          candidate_kl,
          hard_kl_limit,
          policy_update_scale,
          kl_backoff_factor,
          kl_recovery_factor,
          kl_min_policy_update_scale,
      )
      metric_shape = metrics['kl_mean'].shape
      metrics['kl_mean'] = jnp.full(metric_shape, committed_kl)
      metrics['kl_candidate'] = jnp.full(metric_shape, candidate_kl)
      metrics['kl_limit_triggered'] = jnp.full(
          metric_shape, rollback, dtype=jnp.float32
      )
      metrics['kl_update_skipped'] = jnp.full(
          metric_shape, rollback, dtype=jnp.float32
      )
      metrics['effective_minibatch_update'] = jnp.full(
          metric_shape, jnp.logical_not(rollback), dtype=jnp.float32
      )
      metrics['kl_policy_update_scale'] = jnp.full(
          metric_shape, policy_update_scale, dtype=jnp.float32
      )

    return (
        optimizer_state,
        params,
        key,
        committed_kl,
        policy_update_scale,
    ), metrics

  def training_step(
      carry: Tuple[TrainingState, envs.State, PRNGKey], unused_t
  ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
    training_state, state, key = carry
    key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

    policy = make_policy((
        training_state.normalizer_params,
        training_state.params.policy,
        training_state.params.value,
    ))

    def f(carry, unused_t):
      current_state, current_key = carry
      current_key, next_key = jax.random.split(current_key)
      extra_fields = ['truncation', 'episode_metrics', 'episode_done']
      if use_split_value_critic:
        extra_fields.extend((
            'dodger_hard_done',
            'dodger_delta',
            'dodger_positive_reward',
            'dodger_negative_reward',
        ))
      if bootstrap_on_timeout:
        extra_fields.append('time_out')
      next_state, data = acting.generate_unroll(
          env,
          current_state,
          policy,
          current_key,
          unroll_length,
          extra_fields=tuple(extra_fields),
      )
      return (next_state, next_key), data

    (state, _), data = jax.lax.scan(
        f,
        (state, key_generate_unroll),
        (),
        length=batch_size * num_minibatches // num_envs,
    )
    # Have leading dimensions (batch_size * num_minibatches, unroll_length)
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
    data = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
    )
    assert data.discount.shape[1:] == (unroll_length,)

    if bootstrap_on_timeout:  # bootstrap reward on timeout
      time_out = data.extras['state_extras']['time_out']
      value = data.extras['policy_extras']['value']
      data = types.Transition(
          observation=data.observation,
          action=data.action,
          reward=data.reward + discounting * time_out * value,
          discount=data.discount,
          next_observation=data.next_observation,
          extras=data.extras,
      )

    normalizer_params = training_state.normalizer_params
    if not lr_is_adaptive_kl:
      # Update normalization params before SGD for backwards compatibility.
      normalizer_params = running_statistics.update(
          normalizer_params,
          _remove_pixels(data.observation),
          pmap_axis_name=_PMAP_AXIS_NAME,
          until_count=normalize_until_count,
      )

    # Each pass is committed only after its KL over the complete fresh rollout
    # is validated. A rejected pass restores both parameters and Adam state,
    # backs off the policy-update scale, and lets a later pass retry.
    committed_kl = jnp.asarray(0.0, dtype=jnp.float32)
    (optimizer_state, params, _, _, policy_update_scale), metrics = (
        jax.lax.scan(
            functools.partial(
                sgd_step, data=data, normalizer_params=normalizer_params
            ),
            (
                training_state.optimizer_state,
                training_state.params,
                key_sgd,
                committed_kl,
                training_state.kl_policy_update_scale,
            ),
            (),
            length=num_updates_per_batch,
        )
    )

    if lr_is_adaptive_kl:
      # For adaptive KL, normalization params should be updated after SGD s.t.
      # old distribution outputs are valid for KL computation.
      normalizer_params = running_statistics.update(
          normalizer_params,
          _remove_pixels(data.observation),
          pmap_axis_name=_PMAP_AXIS_NAME,
          until_count=normalize_until_count,
      )

    new_training_state = TrainingState(
        optimizer_state=optimizer_state,
        params=params,
        normalizer_params=normalizer_params,
        env_steps=training_state.env_steps + env_step_per_training_step,
        kl_policy_update_scale=policy_update_scale,
    )

    if log_training_metrics:  # log unroll metrics
      jax.debug.callback(
          metrics_aggregator.update_episode_metrics,
          data.extras['state_extras']['episode_metrics'],
          data.extras['state_extras']['episode_done'],
          metrics,
      )

    return (new_training_state, state, new_key), metrics

  def training_epoch(
      training_state: TrainingState, state: envs.State, key: PRNGKey
  ) -> Tuple[TrainingState, envs.State, Metrics]:
    (training_state, state, _), loss_metrics = jax.lax.scan(
        training_step,
        (training_state, state, key),
        (),
        length=num_training_steps_per_epoch,
    )
    if enable_kl_limit:
      # Preserve the peak candidate KL before reducing the pass/minibatch axes.
      # This makes a rejected overshoot directly visible in W&B.
      candidate_kl_max = jnp.max(loss_metrics['kl_candidate'])
    loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
    if enable_kl_limit:
      loss_metrics['kl_candidate_max'] = candidate_kl_max
    return training_state, state, loss_metrics

  training_epoch = jax.pmap(
      training_epoch,
      axis_name=_PMAP_AXIS_NAME,
      donate_argnums=(
          0,
          1,
      ),
  )

  # Note that this is NOT a pure jittable method.
  def training_epoch_with_timing(
      training_state: TrainingState, env_state: envs.State, key: PRNGKey
  ) -> Tuple[TrainingState, envs.State, Metrics]:
    nonlocal training_walltime
    t = time.time()
    training_state, env_state = _strip_weak_type((training_state, env_state))
    result = training_epoch(training_state, env_state, key)
    training_state, env_state, metrics = _strip_weak_type(result)

    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (
        num_training_steps_per_epoch
        * env_step_per_training_step
        * max(num_resets_per_eval, 1)
    ) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()},
    }
    return (
        training_state,
        env_state,
        metrics,
    )  # pytype: disable=bad-return-type  # py311-upgrade

  # Initialize model params and training state.
  init_params = ppo_losses.PPONetworkParams(
      policy=ppo_network.policy_network.init(key_policy),
      value=ppo_network.value_network.init(key_value),
  )

  obs_shape = jax.tree_util.tree_map(
      lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs
  )
  training_state = (
      TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
          optimizer_state=optimizer.init(
              init_params
          ),  # pytype: disable=wrong-arg-types  # numpy-scalars
          params=init_params,
          normalizer_params=running_statistics.init_state(
              _remove_pixels(obs_shape),
              std_eps=normalize_observations_std_eps,
              mode=normalize_observations_mode,
          ),
          env_steps=types.UInt64(hi=0, lo=0),
          kl_policy_update_scale=jnp.asarray(1.0, dtype=jnp.float32),
      )
  )

  if restore_checkpoint_path is not None:
    params = checkpoint.load(restore_checkpoint_path)
    value_params = params[2] if restore_value_fn else init_params.value
    training_state = training_state.replace(
        normalizer_params=params[0],
        params=training_state.params.replace(
            policy=params[1], value=value_params
        ),
    )

  if restore_params is not None:
    logging.info('Restoring TrainingState from `restore_params`.')
    value_params = restore_params[2] if restore_value_fn else init_params.value
    training_state = training_state.replace(
        normalizer_params=restore_params[0],
        params=training_state.params.replace(
            policy=restore_params[1], value=value_params
        ),
    )

  if num_timesteps == 0:
    return (
        make_policy,
        (
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        ),
        {},
    )

  training_state = jax.device_put_replicated(
      training_state, jax.local_devices()[:local_devices_to_use]
  )

  eval_env = _maybe_wrap_env(
      eval_env or environment,
      wrap_env,
      num_eval_envs,
      episode_length,
      action_repeat,
      device_count=1,  # eval on the host only
      key_env=eval_key,
      wrap_env_fn=wrap_env_fn,
      randomization_fn=randomization_fn,
  )
  evaluator = acting.Evaluator(
      eval_env,
      functools.partial(make_policy, deterministic=deterministic_eval),
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key,
  )

  training_metrics = {}
  training_walltime = 0
  current_step = 0

  # Run initial eval
  metrics = {}
  if process_id == 0 and num_evals > 1 and run_evals:
    metrics = evaluator.run_evaluation(
        _unpmap((
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        )),
        training_metrics={},
    )
    logging.info(metrics)
    progress_fn(0, metrics)

  # Run initial policy_params_fn.
  params = _unpmap((
      training_state.normalizer_params,
      training_state.params.policy,
      training_state.params.value,
  ))
  policy_params_fn(current_step, make_policy, params)

  for it in range(num_evals_after_init):
    logging.info('starting iteration %s %s', it, time.time() - xt)

    for _ in range(max(num_resets_per_eval, 1)):
      # optimization
      epoch_key, local_key = jax.random.split(local_key)
      epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
      training_state, env_state, training_metrics = training_epoch_with_timing(
          training_state, env_state, epoch_keys
      )
      current_step = int(_unpmap(training_state.env_steps))

      key_envs = jax.vmap(
          lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
      )(key_envs, key_envs.shape[1])
      # TODO(brax-team): move extra reset logic to the AutoResetWrapper.
      if num_resets_per_eval > 0:
        env_state = reset_fn(env_state, key_envs)

    if process_id != 0:
      continue

    # Process id == 0.
    params = _unpmap((
        training_state.normalizer_params,
        training_state.params.policy,
        training_state.params.value,
    ))

    policy_params_fn(current_step, make_policy, params)

    if save_checkpoint_path is not None:
      ckpt_config = checkpoint.network_config(
          observation_size=obs_shape,
          action_size=env.action_size,
          normalize_observations=normalize_observations,
          network_factory=network_factory,
      )
      checkpoint.save(save_checkpoint_path, current_step, params, ckpt_config)

    if num_evals > 0:
      metrics = training_metrics
      if run_evals:
        metrics = evaluator.run_evaluation(
            params,
            training_metrics,
        )
      logging.info(metrics)
      progress_fn(current_step, metrics)

  total_steps = current_step
  if not total_steps >= num_timesteps:
    raise AssertionError(
        f'Total steps {total_steps} is less than `num_timesteps`='
        f' {num_timesteps}.'
    )

  # If there was no mistakes the training_state should still be identical on all
  # devices.
  pmap.assert_is_replicated(training_state)
  params = _unpmap((
      training_state.normalizer_params,
      training_state.params.policy,
      training_state.params.value,
  ))
  logging.info('total steps: %s', total_steps)
  pmap.synchronize_hosts()
  return (make_policy, params, metrics)
