"""Compatibility helpers for visualizing Brax rollouts with rscope.

The upstream rscope saver slices every rollout leaf along its leading axis.
That fails when an environment exposes scalar metrics.  Non-vision rollouts
are already reset with exactly ``rscope_envs`` environments, so this adapter
only slices leaves when necessary and broadcasts scalar leaves explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from rscope import brax as rscope_brax
from rscope import rscope_utils


class BraxRolloutSaver:
  """Collects fixed-size policy rollouts for the interactive rscope viewer."""

  def __init__(  # noqa: PLR0917
      self,
      trace_env,
      ppo_params: config_dict.ConfigDict,
      vision: bool,
      rscope_envs: int,
      deterministic: bool,
      key: jax.Array,
      callback_fn: Callable | None = None,
  ):
    if rscope_envs <= 0:
      raise ValueError("rscope_envs must be positive")
    if vision and rscope_envs > ppo_params.num_envs:
      raise ValueError(
          "rscope_envs cannot exceed num_envs for a vision environment"
      )
    self.trace_env = trace_env
    self.ppo_params = ppo_params
    self.vision = vision
    self.rscope_envs = rscope_envs
    self.deterministic = deterministic
    self.make_policy = None
    self.key = key
    self.callback_fn = callback_fn

    # Navigation constructs its scene in memory, so ``xml_path`` may only be a
    # descriptive filename.  Save the compiled model to a real temporary XML
    # that rscope can package together with the model assets.
    xml_path = Path(trace_env.xml_path)
    if not xml_path.is_file():
      generated_dir = Path("/tmp/rscope/generated_models")
      generated_dir.mkdir(parents=True, exist_ok=True)
      xml_path = generated_dir / xml_path.name
      mujoco.mj_saveLastXML(str(xml_path), trace_env.mj_model)
    rscope_utils.rscope_init(xml_path, dict(trace_env.model_assets))

  def set_make_policy(self, new_make_policy) -> None:
    if self.make_policy is None:
      self.make_policy = new_make_policy

  def _take_envs(self, value, source_envs: int):
    if value is None:
      return None
    value = jp.asarray(value)
    if value.ndim == 0:
      return jp.broadcast_to(value, (self.rscope_envs,))
    if value.shape[0] == source_envs:
      return value[: self.rscope_envs]
    # A global vector/array has no environment axis.  Replicate it so that
    # every saved rollout leaf has a consistent leading environment axis.
    return jp.broadcast_to(value, (self.rscope_envs,) + value.shape)

  def _rollout(self, params):
    if self.make_policy is None:
      raise RuntimeError("set_make_policy must be called before dump_rollout")
    key_unroll, key_reset = jax.random.split(self.key)
    source_envs = (
        self.ppo_params.num_envs if self.vision else self.rscope_envs
    )
    reset_keys = jax.random.split(key_reset, source_envs)
    policy = self.make_policy(params, deterministic=self.deterministic)
    state = self.trace_env.reset(reset_keys)

    def step_fn(carry, _):
      state, key = carry
      key, action_key = jax.random.split(key)
      action, _ = policy(state.obs, action_key)
      state = self.trace_env.step(state, action)
      output = (
          rscope_brax.make_raw_rollout(state),
          state.obs,
          state.reward,
          state.done,
      )
      output = jax.tree.map(
          lambda value: self._take_envs(value, source_envs), output
      )
      return (state, key), output

    _, (trace, obs, reward, done) = jax.lax.scan(
        step_fn,
        (state, key_unroll),
        None,
        length=self.ppo_params.episode_length
        // self.ppo_params.action_repeat,
    )
    return trace, obs, reward, done

  def dump_rollout(self, params) -> None:
    trace, obs, reward, done = jax.jit(self._rollout)(params)
    if self.callback_fn is not None:
      self.callback_fn(trace, obs, reward, done)
    rscope_utils.dump_eval(trace, obs, reward)
