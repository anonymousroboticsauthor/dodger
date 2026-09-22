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

"""Tests for the project-local PPO trainer."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from brax.training.agents.ppo import losses as ppo_losses

from learning.ppo import train


class TransactionalKlGuardTest(absltest.TestCase):

  def _apply(self, candidate_kl, update_scale=1.0):
    return jax.jit(train._apply_transactional_kl_guard)(  # pylint: disable=protected-access
        {'adam': jnp.asarray(1.0)},
        {'policy': jnp.asarray(2.0)},
        {'adam': jnp.asarray(10.0)},
        {'policy': jnp.asarray(20.0)},
        jnp.asarray(0.1),
        jnp.asarray(candidate_kl),
        0.2,
        jnp.asarray(update_scale),
        0.5,
        1.05,
        1.0 / 64.0,
    )

  def test_commits_candidate_within_limit(self):
    optimizer, params, committed_kl, rollback, scale = self._apply(0.2)

    np.testing.assert_allclose(optimizer['adam'], 10.0)
    np.testing.assert_allclose(params['policy'], 20.0)
    np.testing.assert_allclose(committed_kl, 0.2)
    self.assertFalse(bool(rollback))
    np.testing.assert_allclose(scale, 1.0)

  def test_rolls_back_parameters_and_optimizer_above_limit(self):
    optimizer, params, committed_kl, rollback, scale = self._apply(0.21)

    np.testing.assert_allclose(optimizer['adam'], 1.0)
    np.testing.assert_allclose(params['policy'], 2.0)
    np.testing.assert_allclose(committed_kl, 0.1)
    self.assertTrue(bool(rollback))
    np.testing.assert_allclose(scale, 0.5)

  def test_rolls_back_nonfinite_candidate(self):
    optimizer, params, committed_kl, rollback, scale = self._apply(jnp.nan)

    np.testing.assert_allclose(optimizer['adam'], 1.0)
    np.testing.assert_allclose(params['policy'], 2.0)
    np.testing.assert_allclose(committed_kl, 0.1)
    self.assertTrue(bool(rollback))
    np.testing.assert_allclose(scale, 0.5)

  def test_accepted_retry_recovers_update_scale(self):
    optimizer, params, committed_kl, rollback, scale = self._apply(
        0.15, update_scale=0.5
    )

    np.testing.assert_allclose(optimizer['adam'], 10.0)
    np.testing.assert_allclose(params['policy'], 20.0)
    np.testing.assert_allclose(committed_kl, 0.15)
    self.assertFalse(bool(rollback))
    np.testing.assert_allclose(scale, 0.525)

  def test_backoff_respects_minimum_scale(self):
    *_, rollback, scale = self._apply(0.21, update_scale=1.0 / 64.0)

    self.assertTrue(bool(rollback))
    np.testing.assert_allclose(scale, 1.0 / 64.0)

  def test_update_scale_changes_policy_but_not_value_update(self):
    update = ppo_losses.PPONetworkParams(
        policy={'weight': jnp.asarray(4.0)},
        value={'weight': jnp.asarray(7.0)},
    )

    scaled = train._scale_policy_update(update, 0.25)  # pylint: disable=protected-access

    np.testing.assert_allclose(scaled.policy['weight'], 1.0)
    np.testing.assert_allclose(scaled.value['weight'], 7.0)


if __name__ == '__main__':
  absltest.main()
