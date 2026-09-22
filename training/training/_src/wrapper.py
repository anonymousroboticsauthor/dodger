# Copyright 2025 DeepMind Technologies Limited
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
# ==============================================================================
"""Wrappers for MuJoCo Playground environments."""

import contextlib
from typing import Any, Callable, List, Optional, Sequence, Tuple

import jax
import mujoco
import numpy as np
from brax.envs.wrappers import training as brax_training
from jax import numpy as jp
from mujoco import mjx

from training._src import mjx_env


class Wrapper(mjx_env.MjxEnv):
  """Wraps an environment to allow modular transformations."""

  def __init__(self, env: Any):  # pylint: disable=super-init-not-called
    self.env = env

  def reset(self, rng: jax.Array) -> mjx_env.State:
    return self.env.reset(rng)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    return self.env.step(state, action)

  @property
  def observation_size(self) -> mjx_env.ObservationSize:
    return self.env.observation_size

  @property
  def action_size(self) -> int:
    return self.env.action_size

  @property
  def unwrapped(self) -> Any:
    return self.env.unwrapped

  def __getattr__(self, name):
    if name == '__setstate__':
      raise AttributeError(name)
    return getattr(self.env, name)

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self.env.mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self.env.mjx_model

  @property
  def xml_path(self) -> str:
    return self.env.xml_path

  def render(  # noqa: PLR0917
      self,
      trajectory: List[mjx_env.State],
      height: int = 240,
      width: int = 320,
      camera: Optional[str] = None,
      scene_option: Optional[mujoco.MjvOption] = None,
      modify_scene_fns: Optional[
          Sequence[Callable[[mujoco.MjvScene], None]]
      ] = None,
  ) -> Sequence[np.ndarray]:
    return self.env.render(
        trajectory, height, width, camera, scene_option, modify_scene_fns
    )


def wrap_for_brax_training(
    env: mjx_env.MjxEnv,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]]
    ] = None,
    full_reset: bool = False,
) -> Wrapper:
  """Common wrapper pattern for all brax training agents.

  Args:
    env: environment to be wrapped
    episode_length: length of episode
    action_repeat: how many repeated actions to take per step
    randomization_fn: randomization function that produces a vectorized model
      and in_axes to vmap over
    full_reset: whether to call `env.reset` during `env.step` on done rather
      than resetting to a cached first state. Setting full_reset=True may
      increase wallclock time because it forces full resets to random states.

  Returns:
    An environment that is wrapped with Episode and AutoReset wrappers.  If the
    environment did not already have batch dimensions, it is additional Vmap
    wrapped.
  """
  curriculum_cfg = getattr(env, 'curriculum_config', None)
  dodger_cfg = getattr(env, 'dodger_config', None)
  dodger_apply_soft_discount = getattr(env, 'dodger_apply_soft_discount', False)
  if curriculum_cfg is not None and randomization_fn is not None:
    raise ValueError(
        'G1 navigation curriculum currently requires its curriculum-aware '
        'vmap wrapper; model domain randomization must be integrated there.'
    )
  if curriculum_cfg is not None:
    env = CurriculumVmapWrapper(env)
  elif randomization_fn is None:
    env = brax_training.VmapWrapper(env)  # pytype: disable=wrong-arg-types
  else:
    env = BraxDomainRandomizationVmapWrapper(env, randomization_fn)
  if curriculum_cfg is not None:
    env = CurriculumEpisodeWrapper(env, episode_length, action_repeat)
    env = FixedWindowCurriculumWrapper(env, curriculum_cfg)
    full_reset = True
  else:
    env = brax_training.EpisodeWrapper(
        env, episode_length, action_repeat
    )  # pyrefly: ignore[bad-argument-type, bad-assignment]
  env = BraxAutoResetWrapper(env, full_reset=full_reset)
  if dodger_cfg is not None:
    env = ConstraintsAsTerminationsWrapper(
        env, dodger_cfg, apply_soft_discount=dodger_apply_soft_discount
    )
  return env


class ConstraintsAsTerminationsWrapper(Wrapper):
  """Applies DODGER reward/discount while preserving hard physical resets.

  The DPCBF and three low-level command violations are independently normalized
  with exponential moving maxima.  Their maximum fractional termination affects
  only the positive-return bootstrap; the original hard ``done`` is restored
  before the next environment step, so DODGER never resets a world by itself.
  """

  def __init__(self, env, config, *, apply_soft_discount):
    super().__init__(env)
    self._dodger_config = config
    self._apply_soft_discount = apply_soft_discount

  def _initialize_statistics(self, state):
    batch_shape = state.done.shape
    state.info['dodger_hard_done'] = jp.zeros(batch_shape)
    constraint_shape = batch_shape + (4,)
    state.info['dodger_cmax'] = jp.zeros(constraint_shape)
    state.info['dodger_cmax_initialized'] = jp.zeros(
        constraint_shape, dtype=jp.bool_
    )
    state.info['dodger_global_steps'] = jp.zeros(batch_shape, dtype=jp.int32)
    state.info['dodger_delta'] = jp.zeros(batch_shape)
    state.info['dodger_positive_reward'] = jp.zeros(batch_shape)
    state.info['dodger_negative_reward'] = jp.zeros(batch_shape)
    return state

  def reset(self, rng):
    return self._initialize_statistics(self.env.reset(rng))

  def step(self, state, action):
    # AutoReset and curriculum accounting must see only genuine environment
    # terminations, never the fractional DODGER continuation probability.
    state = state.replace(done=state.info['dodger_hard_done'])
    state = self.env.step(state, action)
    hard_done = state.done
    config = self._dodger_config

    violation = state.info['dodger_constraint_violations']
    batch_max = jp.max(violation, axis=0)
    previous_cmax = state.info['dodger_cmax'][0]
    initialized = state.info['dodger_cmax_initialized'][0]
    cmax = jp.where(
        initialized,
        config.cmax_ema_tau * previous_cmax
        + (1.0 - config.cmax_ema_tau) * batch_max,
        batch_max,
    )
    cmax = jp.maximum(cmax, config.normalization_epsilon)
    batch_size = state.done.shape[0]
    global_steps = state.info['dodger_global_steps'][0] + jp.asarray(
        batch_size, dtype=jp.int32
    )
    ramp_steps = jp.asarray(config.probability_ramp_end_steps, dtype=jp.float32)
    ramp_progress = jp.clip(global_steps / ramp_steps, 0.0, 1.0)
    dpcbf = config.dpcbf_constraint
    command = config.low_level_command_constraint
    p_min = jp.asarray((
        dpcbf.probability_min,
        command.probability_min,
        command.probability_min,
        command.probability_min,
    ))
    p_max = jp.asarray((
        dpcbf.probability_max,
        command.probability_max,
        command.probability_max,
        command.probability_max,
    ))
    probability = p_min + (p_max - p_min) * ramp_progress
    normalized_violation = jp.clip(violation / cmax, 0.0, 1.0)
    delta = jp.max(probability * normalized_violation, axis=-1)

    original_reward = state.reward
    reward_metric_updates = {}
    cbf_reward = state.metrics['reward/cbf']
    positive_reward = jp.zeros_like(original_reward)
    penalty_reward = jp.zeros_like(original_reward)
    for name, value in state.metrics.items():
      if not name.startswith('reward/') or name == 'reward/cbf':
        continue
      positive_reward += jp.maximum(value, 0.0)
      penalty_reward += jp.minimum(value, 0.0)
      reward_metric_updates[name] = jp.where(
          value > 0.0, (1.0 - delta) * value, value
      )
    reward = cbf_reward + (1.0 - delta) * positive_reward + penalty_reward

    metric_updates = {
        **reward_metric_updates,
        'navigation/dodger_delta_per_step': delta,
        'navigation/dodger_cmax_per_step': jp.full_like(delta, jp.max(cmax)),
        'navigation/dodger_probability_per_step': jp.full_like(
            delta, jp.max(probability)
        ),
    }
    episode_metrics = state.info.get('episode_metrics')
    for name, value in metric_updates.items():
      previous_value = state.metrics[name]
      state.metrics[name] = value
      if episode_metrics is not None:
        episode_metrics[name] += value - previous_value
    if episode_metrics is not None:
      episode_metrics['sum_reward'] += reward - original_reward

    state.info['dodger_hard_done'] = hard_done
    state.info['dodger_cmax'] = jp.broadcast_to(cmax, violation.shape)
    state.info['dodger_cmax_initialized'] = jp.ones_like(violation, dtype=jp.bool_)
    state.info['dodger_global_steps'] = jp.full_like(
        state.info['dodger_global_steps'], global_steps
    )
    state.info['dodger_delta'] = delta
    state.info['dodger_positive_reward'] = (1.0 - delta) * positive_reward
    state.info['dodger_negative_reward'] = cbf_reward + penalty_reward
    soft_done = 1.0 - (1.0 - hard_done) * (1.0 - delta)
    outward_done = jp.where(self._apply_soft_discount, soft_done, hard_done)
    return state.replace(reward=reward, done=outward_done)


class CurriculumVmapWrapper(Wrapper):
  """Vmap wrapper exposing stage-conditioned resets for navigation."""

  def reset(self, rng: jax.Array) -> mjx_env.State:
    return jax.vmap(self.env.reset)(rng)

  def reset_at_stage(self, rng: jax.Array, stage: jax.Array) -> mjx_env.State:
    return jax.vmap(self.env.reset_at_stage)(rng, stage)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    return jax.vmap(self.env.step)(state, action)


class CurriculumEpisodeWrapper(brax_training.EpisodeWrapper):
  """Episode accounting that also supports stage-conditioned resets."""

  def _add_episode_info(self, state, rng):
    state.info['steps'] = jp.zeros(rng.shape[:-1])
    state.info['truncation'] = jp.zeros(rng.shape[:-1])
    state.info['episode_done'] = jp.zeros(rng.shape[:-1])
    episode_metrics = {
        'sum_reward': jp.zeros(rng.shape[:-1]),
        'length': jp.zeros(rng.shape[:-1]),
    }
    for metric_name in state.metrics.keys():
      episode_metrics[metric_name] = jp.zeros(rng.shape[:-1])
    state.info['episode_metrics'] = episode_metrics
    return state

  def reset(self, rng: jax.Array) -> mjx_env.State:
    return self._add_episode_info(self.env.reset(rng), rng)

  def reset_at_stage(self, rng, stage):
    return self._add_episode_info(self.env.reset_at_stage(rng, stage), rng)


class FixedWindowCurriculumWrapper(Wrapper):
  """Global success windows with dynamic or timestep-fixed stage selection."""

  def __init__(self, env, config):
    super().__init__(env)
    self._curriculum_config = config

  def _initialize_statistics(self, state):
    batch_shape = state.done.shape
    state.info['curriculum_success_sum'] = jp.zeros(batch_shape)
    state.info['curriculum_episode_count'] = jp.zeros(
        batch_shape, dtype=jp.int32
    )
    state.info['curriculum_success_rate'] = jp.zeros(batch_shape)
    state.info['curriculum_demote_bad_checks'] = jp.zeros(
        batch_shape, dtype=jp.int32
    )
    state.info['curriculum_global_steps'] = jp.zeros(
        batch_shape, dtype=jp.int32
    )
    state.info['curriculum_stage_steps'] = jp.zeros(
        batch_shape, dtype=jp.int32
    )
    return state

  def reset(self, rng):
    return self._initialize_statistics(self.env.reset(rng))

  def reset_at_stage_with_info(self, rng, previous_info):
    state = self.env.reset_at_stage(rng, previous_info['curriculum_stage'])
    for key in (
        'curriculum_success_sum',
        'curriculum_episode_count',
        'curriculum_success_rate',
        'curriculum_demote_bad_checks',
        'curriculum_global_steps',
        'curriculum_stage_steps',
    ):
      state.info[key] = previous_info[key]
    return state

  def step(self, state, action):
    state = self.env.step(state, action)
    done = state.done.astype(bool)
    previous_count = state.info['curriculum_episode_count'][0]
    previous_sum = state.info['curriculum_success_sum'][0]
    target = jp.asarray(
        self._curriculum_config.evaluation_episodes, dtype=jp.int32
    )
    needed = jp.maximum(target - previous_count, 0)
    rank = jp.cumsum(done.astype(jp.int32))
    selected = done & (rank <= needed)
    added_count = jp.sum(selected.astype(jp.int32))
    added_success = jp.sum(
        selected * state.info['goal_reached'].astype(jp.float32)
    )
    count = previous_count + added_count
    success_sum = previous_sum + added_success
    ready = count >= target
    rate = success_sum / jp.maximum(count, 1)
    current_stage = state.info['curriculum_stage'][0]
    previous_bad_checks = state.info['curriculum_demote_bad_checks'][0]
    max_stage = len(self._curriculum_config.stages) - 1
    global_steps = state.info['curriculum_global_steps'][0] + jp.asarray(
        state.done.shape[0], dtype=jp.int32
    )
    stage_steps = state.info['curriculum_stage_steps'][0] + jp.asarray(
        state.done.shape[0], dtype=jp.int32
    )

    minimum_stage_steps_met = (
        ~jp.asarray(
            self._curriculum_config.minimum_stage_steps_enabled,
            dtype=jp.bool_,
        )
        | (
            stage_steps
            >= jp.asarray(
                self._curriculum_config.minimum_stage_steps,
                dtype=jp.int32,
            )
        )
    )
    promoted = (
        ready
        & minimum_stage_steps_met
        & (current_stage < max_stage)
        & (rate >= self._curriculum_config.promote_threshold)
    )
    demote_signal = ready & (rate <= self._curriculum_config.demote_threshold)
    bad_checks = jp.where(
        ready,
        jp.where(demote_signal, previous_bad_checks + 1, 0),
        previous_bad_checks,
    )
    demoted = demote_signal & (
        bad_checks >= self._curriculum_config.demote_required_checks
    )
    next_stage = jp.where(
        promoted,
        jp.minimum(current_stage + 1, max_stage),
        jp.where(demoted, jp.maximum(current_stage - 1, 0), current_stage),
    )
    stage_changed = next_stage != current_stage
    stage_steps = jp.where(stage_changed, 0, stage_steps)
    reset_window = ready
    count = jp.where(reset_window, 0, count)
    success_sum = jp.where(reset_window, 0.0, success_sum)
    logged_rate = jp.where(
        ready, rate, state.info['curriculum_success_rate'][0]
    )
    bad_checks = jp.where(promoted | demoted, 0, bad_checks)
    state.info['curriculum_stage'] = jp.full_like(
        state.info['curriculum_stage'], next_stage
    )
    state.info['curriculum_episode_count'] = jp.full_like(
        state.info['curriculum_episode_count'], count
    )
    state.info['curriculum_success_sum'] = jp.full_like(
        state.info['curriculum_success_sum'], success_sum
    )
    state.info['curriculum_success_rate'] = jp.full_like(
        state.info['curriculum_success_rate'], logged_rate
    )
    state.info['curriculum_demote_bad_checks'] = jp.full_like(
        state.info['curriculum_demote_bad_checks'], bad_checks
    )
    state.info['curriculum_global_steps'] = jp.full_like(
        state.info['curriculum_global_steps'], global_steps
    )
    state.info['curriculum_stage_steps'] = jp.full_like(
        state.info['curriculum_stage_steps'], stage_steps
    )
    state.metrics['navigation/curriculum_stage_per_step'] = jp.full_like(
        state.metrics['navigation/curriculum_stage_per_step'], next_stage
    )
    state.metrics['navigation/curriculum_success_rate_per_step'] = jp.full_like(
        state.metrics['navigation/curriculum_success_rate_per_step'],
        logged_rate,
    )
    # EpisodeWrapper accumulated the step metrics before this global curriculum
    # decision.  Replace completed-episode values so the training logger reports
    # the newly selected stage and completed-window rate without one-episode lag.
    episode_metrics = state.info['episode_metrics']
    episode_length = episode_metrics['length']
    for name, value in (
        ('navigation/curriculum_stage_per_step', next_stage),
        ('navigation/curriculum_success_rate_per_step', logged_rate),
    ):
      episode_metrics[name] = jp.where(
          done,
          value.astype(episode_metrics[name].dtype) * episode_length,
          episode_metrics[name],
      )
    return state


class BraxAutoResetWrapper(Wrapper):
  """Automatically resets Brax envs that are done.

  If `full_reset` is disabled (default):
    * the environment will reset to a cached first state.
    * only data and obs are reset, not the environment info.

  If `full_reset` is enabled:
    * the environment will call env.reset during env.step on done.
    * `full_reset` will thus incur a penalty in wallclock time depending on the
      complexity of the reset function.
    * info is fully reset, except for info under the key
      `AutoResetWrapper_preserve_info`, which is passed through from the prior
      step. This can be used for curriculum learning.

  Attributes:
    env: The wrapped environment.
    full_reset: Whether to call `env.reset` during `env.step` on done.
  """

  def __init__(self, env: Any, full_reset: bool = False):
    super().__init__(env)
    self._full_reset = full_reset
    self._info_key = 'AutoResetWrapper'

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng_key = jax.vmap(jax.random.split)(rng)
    rng, key = rng_key[..., 0], rng_key[..., 1]
    state = self.env.reset(key)
    state.info[f'{self._info_key}_first_data'] = state.data
    state.info[f'{self._info_key}_first_obs'] = state.obs
    state.info[f'{self._info_key}_rng'] = rng
    state.info[f'{self._info_key}_done_count'] = jp.zeros(
        key.shape[:-1], dtype=int
    )
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if self._full_reset and hasattr(self.env, 'reset_at_stage_with_info'):
      return self._step_with_curriculum_reset(state, action)
    # grab the reset state.
    reset_state = None
    rng_key = jax.vmap(jax.random.split)(state.info[f'{self._info_key}_rng'])
    reset_rng, reset_key = rng_key[..., 0], rng_key[..., 1]
    if self._full_reset:
      reset_state = self.reset(reset_key)
      reset_data = reset_state.data
      reset_obs = reset_state.obs
    else:
      reset_data = state.info[f'{self._info_key}_first_data']
      reset_obs = state.info[f'{self._info_key}_first_obs']

    if 'steps' in state.info:
      # reset steps to 0 if done.
      steps = state.info['steps']
      steps = jp.where(state.done, jp.zeros_like(steps), steps)
      state.info.update(steps=steps)

    state = state.replace(
        done=jp.zeros_like(state.done)
    )  # pyrefly: ignore[missing-attribute]
    state = self.env.step(state, action)

    def where_done(x, y):
      done = state.done
      if done.shape and done.shape[0] != x.shape[0]:
        return y
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
      return jp.where(done, x, y)

    data = jax.tree.map(where_done, reset_data, state.data)
    obs = jax.tree.map(where_done, reset_obs, state.obs)

    next_info = state.info
    done_count_key = f'{self._info_key}_done_count'
    if self._full_reset and reset_state:
      next_info = jax.tree.map(where_done, reset_state.info, state.info)
      next_info[done_count_key] = state.info[done_count_key]

      if 'steps' in next_info:
        next_info['steps'] = state.info['steps']
      preserve_info_key = f'{self._info_key}_preserve_info'
      if preserve_info_key in next_info:
        next_info[preserve_info_key] = state.info[preserve_info_key]

    next_info[done_count_key] += state.done.astype(int)
    next_info[f'{self._info_key}_rng'] = reset_rng

    return state.replace(data=data, obs=obs, info=next_info)

  def _step_with_curriculum_reset(self, state, action):
    """Resets completed worlds using the newly evaluated global stage."""
    rng_key = jax.vmap(jax.random.split)(state.info[f'{self._info_key}_rng'])
    reset_rng, reset_key = rng_key[..., 0], rng_key[..., 1]
    if 'steps' in state.info:
      state.info['steps'] = jp.where(
          state.done, jp.zeros_like(state.info['steps']), state.info['steps']
      )
    state = state.replace(done=jp.zeros_like(state.done))
    state = self.env.step(state, action)
    reset_state = self.env.reset_at_stage_with_info(reset_key, state.info)

    # AutoReset-owned fields are absent below this wrapper.  Copy them into the
    # reset tree so JAX sees identical structures, then explicitly advance the
    # RNG and done count below.
    for key in state.info:
      if key not in reset_state.info:
        reset_state.info[key] = state.info[key]

    def where_done(x, y):
      done = state.done
      if done.shape and done.shape[0] != x.shape[0]:
        return y
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
      return jp.where(done, x, y)

    data = jax.tree.map(where_done, reset_state.data, state.data)
    obs = jax.tree.map(where_done, reset_state.obs, state.obs)
    next_info = jax.tree.map(where_done, reset_state.info, state.info)
    # Acting reads these fields from the returned terminal transition.  Keep
    # them for one step so the training episode logger sees completed rewards;
    # CurriculumEpisodeWrapper clears them at the start of the next episode.
    for key in ('episode_done', 'episode_metrics', 'truncation'):
      if key in state.info:
        next_info[key] = state.info[key]
    done_count_key = f'{self._info_key}_done_count'
    next_info[done_count_key] = state.info[done_count_key] + state.done.astype(
        int
    )
    next_info[f'{self._info_key}_rng'] = reset_rng
    return state.replace(data=data, obs=obs, info=next_info)


class BraxDomainRandomizationVmapWrapper(Wrapper):
  """Brax wrapper for domain randomization."""

  def __init__(
      self,
      env: mjx_env.MjxEnv,
      randomization_fn: Callable[[mjx.Model], Tuple[mjx.Model, mjx.Model]],
  ):
    super().__init__(env)
    self._mjx_model_v, self._in_axes = randomization_fn(self.mjx_model)

  @contextlib.contextmanager
  def v_env_fn(self, mjx_model: mjx.Model):
    env = self.env.unwrapped
    old_mjx_model = env._mjx_model
    try:
      env.unwrapped._mjx_model = mjx_model
      yield env
    finally:
      env.unwrapped._mjx_model = old_mjx_model

  def reset(self, rng: jax.Array) -> mjx_env.State:
    def reset(mjx_model, rng):
      with self.v_env_fn(mjx_model) as v_env:
        return v_env.reset(rng)

    state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._mjx_model_v, rng)
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    def step(mjx_model, s, a):
      with self.v_env_fn(mjx_model) as v_env:
        return v_env.step(s, a)

    res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
        self._mjx_model_v, state, action
    )
    return res
