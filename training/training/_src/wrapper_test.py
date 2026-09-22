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
"""Tests for the wrapper module."""

import functools

import jax
import jax.numpy as jp
import numpy as np
from absl.testing import absltest, parameterized
from brax.envs.wrappers import training as brax_training

from training._src import dm_control_suite, mjx_env, wrapper


class WrapperTest(parameterized.TestCase):

  def test_dynamic_curriculum_requires_minimum_stage_steps(self):
    class CurriculumEnv:

      def reset(self, rng):
        batch_shape = rng.shape[:-1]
        zeros = jp.zeros(batch_shape)
        metrics = {
            'navigation/curriculum_stage_per_step': zeros,
            'navigation/curriculum_success_rate_per_step': zeros,
        }
        info = {
            'curriculum_stage': jp.zeros(batch_shape, dtype=jp.int32),
            'goal_reached': jp.ones(batch_shape, dtype=jp.bool_),
            'episode_metrics': {
                'length': jp.ones(batch_shape),
                **metrics,
            },
        }
        return mjx_env.State(zeros, zeros, zeros, zeros, metrics, info)

      def step(self, state, action):
        del action
        state.info['goal_reached'] = jp.ones_like(
            state.info['goal_reached'], dtype=jp.bool_
        )
        state.info['episode_metrics']['length'] = jp.ones_like(state.done)
        return state.replace(done=jp.ones_like(state.done))

    config = type(
        'CurriculumConfig',
        (),
        {
            'evaluation_episodes': 2,
            'minimum_stage_steps_enabled': True,
            'minimum_stage_steps': 4,
            'promote_threshold': 1.0,
            'demote_threshold': -1.0,
            'demote_required_checks': 3,
            'stages': (None, None),
        },
    )()
    env = wrapper.FixedWindowCurriculumWrapper(CurriculumEnv(), config)
    rng = jax.random.split(jax.random.PRNGKey(0), 2)
    state = jax.jit(env.reset)(rng)
    step = jax.jit(env.step)

    state = step(state, jp.zeros((2, 1)))
    np.testing.assert_array_equal(state.info['curriculum_stage'], (0, 0))
    np.testing.assert_array_equal(state.info['curriculum_stage_steps'], (2, 2))

    state = step(state, jp.zeros((2, 1)))
    np.testing.assert_array_equal(state.info['curriculum_stage'], (1, 1))
    np.testing.assert_array_equal(state.info['curriculum_stage_steps'], (0, 0))

  def test_dodger_wrapper_uses_soft_discount_without_soft_reset(self):
    class DodgerEnv:

      action_size = 1

      def reset(self, rng):
        batch_shape = rng.shape[:-1]
        zeros = jp.zeros(batch_shape)
        metrics = {
            'reward/cbf': zeros,
            'reward/progress': zeros,
            'reward/time': zeros,
            'navigation/dodger_delta_per_step': zeros,
            'navigation/dodger_cmax_per_step': zeros,
            'navigation/dodger_probability_per_step': zeros,
        }
        episode_metrics = {'sum_reward': zeros, **metrics}
        info = {
            'dodger_constraint_violations': jp.asarray(
                ((1.0, 0.0, 0.0, 0.0), (0.5, 0.0, 0.0, 0.0))
            ),
            'episode_metrics': episode_metrics,
            'seen_done': zeros,
        }
        return mjx_env.State(zeros, zeros, zeros, zeros, metrics, info)

      def step(self, state, action):
        state.info['seen_done'] = state.done
        state.metrics['reward/cbf'] = jp.full_like(state.done, -1.0)
        state.metrics['reward/progress'] = jp.full_like(state.done, 4.0)
        state.metrics['reward/time'] = jp.full_like(state.done, -2.0)
        return state.replace(
            reward=jp.ones_like(state.done), done=jp.zeros_like(state.done)
        )

    config = type(
        'DodgerConfig',
        (),
        {
            'cmax_ema_tau': 0.95,
            'normalization_epsilon': 1.0e-6,
            'probability_ramp_end_steps': 70,
            'dpcbf_constraint': type(
                'Constraint', (), {'probability_min': 0.25, 'probability_max': 0.25}
            )(),
            'low_level_command_constraint': type(
                'Constraint', (), {'probability_min': 0.25, 'probability_max': 0.25}
            )(),
        },
    )()
    env = wrapper.ConstraintsAsTerminationsWrapper(
        DodgerEnv(), config, apply_soft_discount=True
    )
    rng = jax.random.split(jax.random.PRNGKey(0), 2)
    state = jax.jit(env.reset)(rng)
    state = jax.jit(env.step)(state, jp.zeros((2, 1)))
    np.testing.assert_allclose(state.done, (0.25, 0.125))
    np.testing.assert_allclose(state.reward, (0.0, 0.5))
    np.testing.assert_allclose(
        state.info['dodger_positive_reward'], (3.0, 3.5)
    )
    np.testing.assert_allclose(state.info['dodger_negative_reward'], (-3.0, -3.0))

    # The fractional done values are exposed to PPO for one transition, but
    # the inner environment receives only the saved hard termination flags.
    state = jax.jit(env.step)(state, jp.zeros((2, 1)))
    np.testing.assert_allclose(state.info['seen_done'], (0.0, 0.0))

  @parameterized.named_parameters(
      ('full_reset', True),
      ('cache_reset', False),
  )
  def test_auto_reset_wrapper(self, full_reset):
    """Tests the AutoResetWrapper."""

    class DoneEnv:

      def __init__(self, env):
        self._env = env

      def reset(self, key):
        state = self._env.reset(key)
        state.info['AutoResetWrapper_preserve_info'] = 1
        state.info['other_info'] = 1
        return state

      def step(self, state, action):
        state = self._env.step(state, jp.ones_like(action))
        state = state.replace(done=action[0] > 0)
        state.info['AutoResetWrapper_preserve_info'] = 2
        state.info['other_info'] = 2
        return state

    env = wrapper.BraxAutoResetWrapper(
        brax_training.VmapWrapper(
            DoneEnv(
                dm_control_suite.load(
                    'CartpoleBalance', config_overrides={'impl': 'jax'}
                )
            )
        ),
        full_reset=full_reset,
    )

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    state = jit_reset(jax.random.PRNGKey(0)[None])
    first_qpos = state.data.qpos

    # First step should not be done.
    state = jit_step(state, -jp.ones(env._env.action_size)[None])
    np.testing.assert_allclose(state.info['AutoResetWrapper_done_count'], 0)
    self.assertGreater(np.linalg.norm(state.data.qpos - first_qpos), 1e-3)
    self.assertEqual(state.info['AutoResetWrapper_preserve_info'], 2)
    self.assertEqual(state.info['other_info'], 2)

    for i in range(1, 3):
      state = jit_step(state, jp.ones(env._env.action_size)[None])
      jax.tree.map(lambda x: x.block_until_ready(), state)
      if full_reset:
        self.assertTrue((state.data.qpos != first_qpos).all())
      else:
        np.testing.assert_allclose(state.data.qpos, first_qpos, atol=1e-6)
      np.testing.assert_allclose(state.info['AutoResetWrapper_done_count'], i)
      self.assertEqual(state.info['AutoResetWrapper_preserve_info'], 2)
      expected_other_info = 1 if full_reset else 2
      self.assertEqual(state.info['other_info'], expected_other_info)

  @parameterized.named_parameters(
      ('full_reset', True),
      ('cache_reset', False),
  )
  def test_evalwrapper_with_reset(self, full_reset):
    """Tests EvalWrapper with reset in the AutoResetWrapper."""
    episode_length = 10
    num_envs = 4

    env = dm_control_suite.load(
        'CartpoleBalance', config_overrides={'impl': 'jax'}
    )
    env = wrapper.wrap_for_brax_training(
        env,
        episode_length=episode_length,
        full_reset=full_reset,
    )
    env = brax_training.EvalWrapper(env)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, num_envs)
    state = jit_reset(rng)
    first_obs = state.obs
    action = jp.zeros((num_envs, env.action_size))

    for _ in range(episode_length):
      state = jit_step(state, action)

    # All episodes should finish at episode_length.
    avg_episode_length = state.info['eval_metrics'].episode_steps.mean()
    np.testing.assert_allclose(avg_episode_length, episode_length, atol=1e-6)
    active_episodes = state.info['eval_metrics'].active_episodes
    self.assertTrue(np.all(active_episodes == 0))

    np.testing.assert_array_equal(state.info['steps'], 10 * np.ones(num_envs))
    if full_reset:
      self.assertTrue((state.obs != first_obs).all())
    else:
      np.testing.assert_allclose(state.obs, first_obs, rtol=1e-6)

  def test_domain_randomization_wrapper(self):
    def randomization_fn(model, rng):
      @jax.vmap
      def get_gravity(rng):
        dg = jax.random.uniform(rng, shape=(3,), minval=-10.0, maxval=10.0)
        return model.opt.gravity + dg

      model_v = model.tree_replace({'opt.gravity': get_gravity(rng)})
      in_axes = jax.tree.map(lambda x: None, model)
      in_axes = in_axes.tree_replace({'opt.gravity': 0})
      return model_v, in_axes

    env = dm_control_suite.load(
        'CartpoleBalance', config_overrides={'impl': 'jax'}
    )
    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, 256)
    env = wrapper.wrap_for_brax_training(
        env,
        episode_length=200,
        randomization_fn=functools.partial(randomization_fn, rng=rng),
    )

    # set the same key across the batch for env.reset so that only the
    # randomization wrapper creates variability in the env.step
    key = jp.zeros((256, 2), dtype=jp.uint32)
    state = jax.jit(env.reset)(key)
    self.assertEqual(state.data.qpos[:, 0].shape[0], 256)
    self.assertEqual(np.unique(state.data.qpos[:, 0]).shape[0], 1)

    # test that the DomainRandomizationWrapper creates variability in env.step
    state = jax.jit(env.step)(state, jp.zeros((256, env.action_size)))
    self.assertEqual(state.data.qpos[:, 0].shape[0], 256)
    self.assertEqual(np.unique(state.data.qpos[:, 0]).shape[0], 256)


if __name__ == '__main__':
  absltest.main()
