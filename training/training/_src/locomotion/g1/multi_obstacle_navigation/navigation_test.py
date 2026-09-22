"""Tests for G1 multi-obstacle navigation."""

import jax
import jax.numpy as jp
import numpy as np
from absl.testing import absltest

from training._src.locomotion.g1.multi_obstacle_navigation import (
    config,
    dpcbf,
    low_level_policy,
    navigation,
    networks,
    qp,
)


def _full_gat_reference(params, observation):
  """Applies the former all-pairs GAT for robot-output equivalence tests."""
  num_nodes, node_dim, local_state_dim = 12, 9, 13
  graph_size = num_nodes * node_dim
  nodes = observation[..., :graph_size].reshape(
      observation.shape[:-1] + (num_nodes, node_dim)
  )
  labels = nodes[..., :3]
  position = nodes[..., 3:5]
  radius = nodes[..., 5]
  velocity = nodes[..., 6:8]
  valid = nodes[..., 8] > 0.5
  labels_i = jp.broadcast_to(
      labels[..., :, None, :],
      labels.shape[:-2] + (num_nodes, num_nodes, 3),
  )
  labels_j = jp.broadcast_to(labels[..., None, :, :], labels_i.shape)
  delta_position = position[..., None, :, :] - position[..., :, None, :]
  center_distance = jp.linalg.norm(delta_position, axis=-1, keepdims=True)
  surface_distance = (
      center_distance - (radius[..., :, None] + radius[..., None, :])[..., None]
  )
  delta_velocity = velocity[..., None, :, :] - velocity[..., :, None, :]
  edge = jp.concatenate(
      (labels_i, labels_j, delta_position, surface_distance, delta_velocity),
      axis=-1,
  )
  layer_params = params["params"]

  def dense(value, name):
    layer = layer_params[name]
    return value @ layer["kernel"] + layer["bias"]

  message = dense(jax.nn.relu(dense(edge, "psi1_dense0")), "psi1_dense1")
  logits = dense(jax.nn.relu(dense(message, "psi2_dense0")), "psi2_dense1")[..., 0]
  edge_valid = valid[..., :, None] & valid[..., None, :]
  edge_valid &= ~jp.eye(num_nodes, dtype=bool)
  logits = jp.where(edge_valid, logits, -1.0e9)
  attention = jax.nn.softmax(logits, axis=-1) * edge_valid
  attention /= jp.maximum(jp.sum(attention, axis=-1, keepdims=True), 1.0e-8)
  encoded = dense(jax.nn.relu(dense(message, "psi3_dense0")), "psi3_dense1")
  robot_embedding = jp.sum(
      attention[..., 0, :, None] * encoded[..., 0, :, :], axis=-2
  )
  hidden = jp.concatenate(
      (robot_embedding, observation[..., graph_size : graph_size + local_state_dim]),
      axis=-1,
  )
  for index in range(3):
    hidden = jax.nn.elu(dense(hidden, f"policy_hidden_{index}"))
  return dense(hidden, "policy_output")


class NavigationTest(absltest.TestCase):

  def test_robot_only_gat_matches_full_reference(self):
    num_nodes, node_dim = 12, 9
    observation = jp.zeros((1, num_nodes * node_dim + 13))
    nodes = observation[..., : num_nodes * node_dim].reshape(1, num_nodes, node_dim)
    nodes = nodes.at[..., 0, 0].set(1.0)
    nodes = nodes.at[..., 1:, 1].set(1.0)
    nodes = nodes.at[..., :, 3].set(jp.arange(num_nodes) * 0.2)
    nodes = nodes.at[..., :, 5].set(0.3)
    nodes = nodes.at[..., :, 8].set(1.0)
    observation = observation.at[..., : num_nodes * node_dim].set(nodes.reshape(1, -1))
    model = networks.DenseGatActor(output_size=6)
    params = model.init(jax.random.PRNGKey(0), observation)
    output = model.apply(params, observation)
    self.assertEqual(params["params"]["psi1_dense0"]["kernel"].shape[0], 11)
    np.testing.assert_allclose(
        output,
        _full_gat_reference(params, observation),
        atol=1.0e-6,
    )

  def test_primal_dual_qp_is_feasible(self):
    q = jp.eye(2)
    p = jp.asarray((-2.0, -2.0))
    g = jp.asarray(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)))
    h = jp.asarray((0.5, 1.0, 0.0))
    result = jax.jit(qp.solve_inequality_qp)(q, p, g, h)
    np.testing.assert_allclose(result.primal, (0.5, 1.0), atol=2e-4)
    self.assertLessEqual(float(jp.max(g @ result.primal - h)), 2e-5)

  def test_multi_constraint_filter_honors_soft_cbf_and_bounds(self):
    cfg = config.default_config()
    count = cfg.obstacle.num_constraints
    angle = jp.linspace(-jp.pi, jp.pi, count, endpoint=False)
    selected = dpcbf.SelectedObstacles(
        relative_position=1.2
        * jp.stack((jp.cos(angle), jp.sin(angle)), axis=-1),
        relative_velocity=-0.4
        * jp.stack((jp.cos(angle), jp.sin(angle)), axis=-1),
        radius=jp.full((count,), 0.25),
        valid=jp.ones((count,), dtype=bool),
        indices=jp.arange(count),
    )
    command = jp.asarray((1.95, 0.95, 0.0))
    result = jax.jit(
        lambda: dpcbf.filter_action(
            selected,
            jp.asarray(0.0),
            command[:2],
            command,
            jp.asarray((4.0, 2.0, 1.3)),
            cfg,
        )
    )()
    next_command = command.at[:2].add(result.safe_action[:2] / 50.0)
    next_command = next_command.at[2].set(result.safe_action[2])
    np.testing.assert_array_less(
        next_command,
        jp.asarray((2.0, 1.0, 1.0)) + 5e-4,
    )
    np.testing.assert_array_less(
        jp.asarray((-1.0, -1.0, -1.0)) - 5e-4,
        next_command,
    )
    self.assertGreaterEqual(float(jp.min(result.safe_condition)), -1e-7)

  def test_hard_feasibility_detects_conflicting_rows(self):
    cfg = config.default_config()
    values = dpcbf.DpcbfValues(
        h=jp.zeros((2,)),
        lf_h=jp.asarray((-1.0, -1.0)),
        lg_h=jp.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
        condition=jp.zeros((2,)),
        valid=jp.ones((2,), dtype=bool),
    )
    diagnostic = dpcbf.diagnose_hard_feasibility(
        values, jp.zeros((3,)), cfg
    )
    self.assertFalse(bool(diagnostic.numerical_failure))
    self.assertFalse(bool(diagnostic.feasible))
    self.assertGreater(float(diagnostic.minimum_relaxation), 0.9)

  def test_qp_numerical_failure_falls_back_to_policy(self):
    cfg = config.default_config()
    count = cfg.obstacle.num_constraints
    selected = dpcbf.SelectedObstacles(
        # Corrupting one sensor row forces a true non-finite numerical solve.
        relative_position=jp.full((count, 2), jp.nan),
        relative_velocity=jp.zeros((count, 2)),
        radius=jp.full((count,), 0.25),
        valid=jp.ones((count,), dtype=bool),
        indices=jp.arange(count),
    )
    policy = jp.asarray((0.4, -0.2, 0.3))
    result = dpcbf.filter_action(
        selected,
        jp.asarray(0.0),
        jp.zeros((2,)),
        jp.zeros((3,)),
        policy,
        cfg,
    )
    self.assertTrue(bool(result.qp_failed))
    np.testing.assert_allclose(result.safe_action, policy)

  def test_deploy_controller_constants_are_applied_to_model(self):
    env = navigation.MultiObstacleNavigation(
        config_overrides={
            "perception.enabled": False,
            "robot_state_randomization.enabled": False,
        }
    )
    np.testing.assert_allclose(
        env.mj_model.actuator_gainprm[:, 0],
        low_level_policy.STIFFNESS,
    )
    np.testing.assert_allclose(
        env.mj_model.actuator_biasprm[:, 2],
        -low_level_policy.DAMPING,
    )
    np.testing.assert_allclose(env.mj_model.dof_damping[6:], 0.0)
    self.assertEqual(
        tuple(env.mj_model.actuator(i).name for i in range(env.mj_model.nu)),
        low_level_policy.JOINT_NAMES,
    )

  def test_reset_starts_in_safe_set_and_observation_is_local(self):
    env = navigation.MultiObstacleNavigation(
        config_overrides={
            "perception.enabled": False,
            "robot_state_randomization.enabled": False,
        }
    )
    state = jax.jit(env.reset)(jax.random.PRNGKey(3))
    surface_clearance = (
        jp.linalg.norm(state.info["obstacle_position"], axis=-1)
        - state.info["obstacle_radius"]
        - env._config.robot.radius
    )
    active_clearance = jp.where(
        state.info["obstacle_active"], surface_clearance, jp.inf
    )
    self.assertGreaterEqual(
        float(jp.min(active_clearance)),
        env._config.obstacle.initial_clearance - 1e-5,
    )
    stationary = state.info["obstacle_active"] & (
        jp.linalg.norm(state.info["obstacle_velocity"], axis=-1)
        <= env._config.goal.stationary_obstacle_speed_threshold
    )
    goal_obstacles = dpcbf.SelectedObstacles(
        relative_position=state.info["obstacle_position"] - state.info["goal"],
        relative_velocity=state.info["obstacle_velocity"],
        radius=state.info["obstacle_radius"],
        valid=stationary,
        indices=jp.arange(env._config.obstacle.max_count),
    )
    goal_barrier = dpcbf.evaluate_dpcbf(
        goal_obstacles,
        jp.asarray(0.0),
        jp.zeros((2,)),
        jp.zeros((3,)),
        env._config,
    ).h
    active_goal_barrier = jp.where(stationary, goal_barrier, jp.inf)
    self.assertGreaterEqual(
        float(jp.min(active_goal_barrier)),
        env._config.goal.safe_set_margin - 1e-5,
    )
    self.assertEqual(state.obs["state"].shape, (121,))
    self.assertEqual(state.obs["privileged_state"].shape, (121,))


if __name__ == "__main__":
  absltest.main()
