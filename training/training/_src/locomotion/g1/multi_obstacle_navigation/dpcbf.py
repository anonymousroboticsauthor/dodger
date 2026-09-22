"""Dynamic parabolic CBF evaluation and multi-constraint QP assembly."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp

from training._src.locomotion.g1.multi_obstacle_navigation import qp

QP_HZ = 50.0


class SelectedObstacles(NamedTuple):
  relative_position: jax.Array
  relative_velocity: jax.Array
  radius: jax.Array
  valid: jax.Array
  indices: jax.Array


class DpcbfValues(NamedTuple):
  h: jax.Array
  lf_h: jax.Array
  lg_h: jax.Array
  condition: jax.Array
  valid: jax.Array


class SafetyFilterResult(NamedTuple):
  safe_action: jax.Array
  slack: jax.Array
  values: DpcbfValues
  safe_condition: jax.Array
  qp_residual: jax.Array
  qp_iterations: jax.Array
  qp_failed: jax.Array


class HardFeasibilityResult(NamedTuple):
  minimum_relaxation: jax.Array
  feasible: jax.Array
  numerical_failure: jax.Array


def world_to_body(vector: jax.Array, yaw: jax.Array) -> jax.Array:
  """Rotates arbitrary leading-dimension planar vectors into robot frame."""
  cosine, sine = jp.cos(yaw), jp.sin(yaw)
  x = cosine * vector[..., 0] + sine * vector[..., 1]
  y = -sine * vector[..., 0] + cosine * vector[..., 1]
  return jp.stack((x, y), axis=-1)


def select_obstacles(  # noqa: PLR0917
    obstacle_position: jax.Array,
    obstacle_velocity: jax.Array,
    obstacle_radius: jax.Array,
    obstacle_active: jax.Array,
    robot_position: jax.Array,
    robot_velocity_world: jax.Array,
    *,
    p_max: float,
    num_constraints: int,
    priority: int,
) -> SelectedObstacles:
  """Selects a fixed number of influential obstacles inside ``p_max``."""
  rel_pos = obstacle_position - robot_position
  rel_vel = obstacle_velocity - robot_velocity_world
  distance = jp.linalg.norm(rel_pos, axis=-1)
  valid = obstacle_active & (distance <= p_max)
  if priority == 0:
    score = distance
  else:
    denominator = jp.maximum(
        jp.linalg.norm(rel_vel, axis=-1) * distance, 1.0e-6
    )
    closing_alignment = -jp.sum(rel_vel * rel_pos, axis=-1) / denominator
    score = -closing_alignment + 1.0e-4 * distance
  score = jp.where(valid, score, 1.0e9)
  _, indices = jax.lax.top_k(-score, num_constraints)
  return SelectedObstacles(
      relative_position=rel_pos[indices],
      relative_velocity=rel_vel[indices],
      radius=obstacle_radius[indices],
      valid=valid[indices],
      indices=indices,
  )


def evaluate_dpcbf(
    selected: SelectedObstacles,
    robot_yaw: jax.Array,
    robot_velocity_body: jax.Array,
    action: jax.Array,
    config,
) -> DpcbfValues:
  """Evaluates all selected DPCBF rows using the deployed C++ formulation."""
  rel_pos = selected.relative_position
  rel_vel = selected.relative_velocity
  px, py = rel_pos[..., 0], rel_pos[..., 1]
  p_squared = jp.maximum(jp.sum(rel_pos * rel_pos, axis=-1), 1.0e-8)
  los_angle = jp.arctan2(py, px)
  cosine_los, sine_los = jp.cos(los_angle), jp.sin(los_angle)
  x_tilde = cosine_los * rel_vel[..., 0] + sine_los * rel_vel[..., 1]
  y_tilde = -sine_los * rel_vel[..., 0] + cosine_los * rel_vel[..., 1]

  safe_velocity = jp.sqrt(
      jp.sum(rel_vel * rel_vel, axis=-1) + config.robot.eps_v**2
  )
  safe_radius = (
      config.robot.radius + selected.radius
  ) * config.robot.safety_factor
  smooth_clearance = jp.sqrt(
      jp.maximum(p_squared - safe_radius**2 + config.robot.eps_d**2, 1.0e-8)
  )
  safe_distance = smooth_clearance - config.robot.eps_d
  adaptive = jp.sqrt(config.robot.safety_factor**2 - 1.0) / safe_radius
  lam = config.dpcbf.k_lambda * safe_distance / safe_velocity
  h = x_tilde + adaptive * (
      lam * y_tilde**2 + config.dpcbf.k_mu * safe_distance
  )

  position_term = (
      config.dpcbf.k_lambda * y_tilde**2 / safe_velocity + config.dpcbf.k_mu
  )
  dh_dx = py * y_tilde / p_squared - adaptive * (
      2.0 * lam * x_tilde * y_tilde * py / p_squared
      + px * position_term / smooth_clearance
  )
  dh_dy = -px * y_tilde / p_squared + adaptive * (
      2.0 * lam * x_tilde * y_tilde * px / p_squared
      - py * position_term / smooth_clearance
  )

  relative_heading = robot_yaw - los_angle
  cosine, sine = jp.cos(relative_heading), jp.sin(relative_heading)
  sagittal, lateral = robot_velocity_body[:2]
  v1 = sagittal * sine + lateral * cosine
  v2 = sagittal * cosine - lateral * sine
  velocity_cubed = safe_velocity**3
  common_x = (
      config.dpcbf.k_lambda
      * safe_distance
      * x_tilde
      * y_tilde**2
      / velocity_cubed
  )
  common_y = (
      config.dpcbf.k_lambda
      * safe_distance
      * (2.0 * y_tilde / safe_velocity - y_tilde**3 / velocity_cubed)
  )
  dh_dyaw = v1 - adaptive * (v1 * common_x + v2 * common_y)
  dh_ds = -cosine + adaptive * (cosine * common_x - sine * common_y)
  dh_dl = sine - adaptive * (sine * common_x + cosine * common_y)
  lg_h = jp.stack((dh_ds, dh_dl, dh_dyaw), axis=-1)

  # p_rel_dot = v_obstacle - v_robot, matching dpcbf_safety_filter.cpp.
  lf_h = -dh_dx * rel_vel[..., 0] - dh_dy * rel_vel[..., 1]
  condition = lf_h + lg_h @ action + config.dpcbf.alpha * h
  return DpcbfValues(
      h=h,
      lf_h=lf_h,
      lg_h=lg_h,
      condition=condition,
      valid=selected.valid,
  )


def filter_action(  # noqa: PLR0917
    selected: SelectedObstacles,
    robot_yaw: jax.Array,
    robot_velocity_body: jax.Array,
    velocity_command: jax.Array,
    policy_action: jax.Array,
    config,
) -> SafetyFilterResult:
  """Solves all valid DPCBF rows with per-row nonnegative slack."""
  values = evaluate_dpcbf(
      selected, robot_yaw, robot_velocity_body, policy_action, config
  )
  count = selected.valid.shape[0]
  control_dim = 3
  variable_dim = control_dim + count
  dtype = policy_action.dtype

  # t_i=sqrt(w)*epsilon_i keeps the Hessian well conditioned while preserving
  # exactly .5*w*epsilon_i^2 in the original variables.
  inv_sqrt_weight = jp.asarray(1.0 / config.qp.slack_weight**0.5, dtype=dtype)
  q_matrix = jp.eye(variable_dim, dtype=dtype)
  linear = jp.concatenate((
      -policy_action,
      jp.zeros((count,), dtype=dtype),
  ))
  slack_eye = jp.eye(count, dtype=dtype)
  cbf_g = jp.concatenate((-values.lg_h, -inv_sqrt_weight * slack_eye), axis=-1)
  cbf_h = values.lf_h + config.dpcbf.alpha * values.h
  cbf_g = jp.where(values.valid[:, None], cbf_g, 0.0)
  cbf_h = jp.where(values.valid, cbf_h, 1.0)

  slack_g = jp.concatenate(
      (jp.zeros((count, control_dim), dtype=dtype), -slack_eye), axis=-1
  )
  slack_h = jp.zeros((count,), dtype=dtype)

  bound_g = jp.zeros((6, variable_dim), dtype=dtype)
  bound_g = bound_g.at[0, 0].set(1.0)
  bound_g = bound_g.at[1, 0].set(-1.0)
  bound_g = bound_g.at[2, 1].set(1.0)
  bound_g = bound_g.at[3, 1].set(-1.0)
  bound_g = bound_g.at[4, 2].set(1.0)
  bound_g = bound_g.at[5, 2].set(-1.0)
  dt = 1.0 / QP_HZ
  sagittal_range = config.robot.sagittal_velocity_range
  lateral_range = config.robot.lateral_velocity_range
  yaw_range = config.robot.yaw_rate_range
  bound_h = jp.asarray(
      (
          (sagittal_range[1] - velocity_command[0]) / dt,
          -(sagittal_range[0] - velocity_command[0]) / dt,
          (lateral_range[1] - velocity_command[1]) / dt,
          -(lateral_range[0] - velocity_command[1]) / dt,
          yaw_range[1],
          -yaw_range[0],
      ),
      dtype=dtype,
  )
  bound_g = jp.where(config.qp.enforce_input_bounds, bound_g, 0.0)
  bound_h = jp.where(config.qp.enforce_input_bounds, bound_h, 1.0)

  g_matrix = jp.concatenate((cbf_g, slack_g, bound_g), axis=0)
  h_vector = jp.concatenate((cbf_h, slack_h, bound_h), axis=0)
  # DPCBF derivatives can differ by several orders of magnitude for close,
  # fast obstacles.  Positive row scaling leaves the feasible set unchanged
  # while preventing those rows from dominating the float32 KKT system.
  row_scale = jp.maximum(jp.linalg.norm(g_matrix, axis=-1), 1.0)
  g_matrix = g_matrix / row_scale[:, None]
  h_vector = h_vector / row_scale
  result = qp.solve_inequality_qp(
      q_matrix,
      linear,
      g_matrix,
      h_vector,
      max_iter=config.qp.max_iter,
      tolerance=config.qp.tolerance,
      regularization=config.qp.regularization,
  )
  solved_action = result.primal[:control_dim]
  action_lower = jp.asarray(
      (
          (sagittal_range[0] - velocity_command[0]) / dt,
          (lateral_range[0] - velocity_command[1]) / dt,
          yaw_range[0],
      ),
      dtype=dtype,
  )
  action_upper = jp.asarray(
      (
          (sagittal_range[1] - velocity_command[0]) / dt,
          (lateral_range[1] - velocity_command[1]) / dt,
          yaw_range[1],
      ),
      dtype=dtype,
  )
  solved_action = jp.where(
      config.qp.enforce_input_bounds,
      jp.clip(solved_action, action_lower, action_upper),
      solved_action,
  )
  slack = jp.maximum(result.primal[control_dim:], 0.0) * inv_sqrt_weight
  # Restore the analytically minimal feasible slack.  This repairs finite
  # float32 IPM iterates without treating a merely unfinished complementarity
  # residual as solver failure.
  required_slack = jp.maximum(
      -(
          values.lf_h
          + values.lg_h @ solved_action
          + config.dpcbf.alpha * values.h
      ),
      0.0,
  )
  required_slack = jp.where(values.valid, required_slack, 0.0)
  slack = jp.maximum(slack, required_slack)
  solved_condition = (
      values.lf_h
      + values.lg_h @ solved_action
      + config.dpcbf.alpha * values.h
      + slack
  )
  primal_violation = jp.max(
      jp.where(values.valid, jp.maximum(-solved_condition, 0.0), 0.0)
  )
  qp_failed = (
      ~jp.all(jp.isfinite(result.primal))
      | ~jp.isfinite(result.residual)
      | ~jp.all(jp.isfinite(solved_condition))
      | (primal_violation > config.qp.feasibility_tolerance)
  )
  # With soft CBF rows and projected input bounds, failure now means a genuine
  # numerical failure.  Pass through the requested nominal policy in that case;
  # the environment command integrator still enforces its configured bounds.
  safe_action = jp.where(qp_failed, policy_action, solved_action)
  slack = jp.where(qp_failed, jp.zeros_like(slack), slack)
  required_slack = jp.maximum(
      -(
          values.lf_h
          + values.lg_h @ safe_action
          + config.dpcbf.alpha * values.h
      ),
      0.0,
  )
  required_slack = jp.where(values.valid, required_slack, 0.0)
  slack = jp.maximum(slack, required_slack)
  safe_condition = (
      values.lf_h
      + values.lg_h @ safe_action
      + config.dpcbf.alpha * values.h
      + slack
  )
  return SafetyFilterResult(
      safe_action=safe_action,
      slack=slack,
      values=values,
      safe_condition=safe_condition,
      qp_residual=result.residual,
      qp_iterations=result.iterations,
      qp_failed=qp_failed,
  )


def diagnose_hard_feasibility(
    values: DpcbfValues,
    velocity_command: jax.Array,
    config,
) -> HardFeasibilityResult:
  """Finds the minimum relaxation required by hard DPCBF and input rows."""
  control_dim = 3
  variable_dim = control_dim + 1
  dtype = values.h.dtype
  diagnostic = config.qp.hard_feasibility
  inv_sqrt_weight = jp.asarray(
      1.0 / diagnostic.relaxation_weight**0.5, dtype=dtype
  )
  relaxation_column = -inv_sqrt_weight * jp.ones(
      (values.h.shape[0], 1), dtype=dtype
  )
  cbf_g = jp.concatenate((-values.lg_h, relaxation_column), axis=-1)
  cbf_h = values.lf_h + config.dpcbf.alpha * values.h
  cbf_g = jp.where(values.valid[:, None], cbf_g, 0.0)
  cbf_h = jp.where(values.valid, cbf_h, 1.0)
  relaxation_g = jp.asarray(((0.0, 0.0, 0.0, -1.0),), dtype=dtype)
  relaxation_h = jp.zeros((1,), dtype=dtype)

  bound_g = jp.zeros((6, variable_dim), dtype=dtype)
  bound_g = bound_g.at[0, 0].set(1.0)
  bound_g = bound_g.at[1, 0].set(-1.0)
  bound_g = bound_g.at[2, 1].set(1.0)
  bound_g = bound_g.at[3, 1].set(-1.0)
  bound_g = bound_g.at[4, 2].set(1.0)
  bound_g = bound_g.at[5, 2].set(-1.0)
  dt = 1.0 / QP_HZ
  sagittal_range = config.robot.sagittal_velocity_range
  lateral_range = config.robot.lateral_velocity_range
  yaw_range = config.robot.yaw_rate_range
  bound_h = jp.asarray(
      (
          (sagittal_range[1] - velocity_command[0]) / dt,
          -(sagittal_range[0] - velocity_command[0]) / dt,
          (lateral_range[1] - velocity_command[1]) / dt,
          -(lateral_range[0] - velocity_command[1]) / dt,
          yaw_range[1],
          -yaw_range[0],
      ),
      dtype=dtype,
  )
  bound_g = jp.where(config.qp.enforce_input_bounds, bound_g, 0.0)
  bound_h = jp.where(config.qp.enforce_input_bounds, bound_h, 1.0)

  g_matrix = jp.concatenate((cbf_g, relaxation_g, bound_g), axis=0)
  h_vector = jp.concatenate((cbf_h, relaxation_h, bound_h), axis=0)
  row_scale = jp.maximum(jp.linalg.norm(g_matrix, axis=-1), 1.0)
  g_matrix = g_matrix / row_scale[:, None]
  h_vector = h_vector / row_scale
  result = qp.solve_inequality_qp(
      jp.eye(variable_dim, dtype=dtype),
      jp.zeros((variable_dim,), dtype=dtype),
      g_matrix,
      h_vector,
      max_iter=diagnostic.max_iter,
      tolerance=diagnostic.solver_tolerance,
      regularization=config.qp.regularization,
  )
  action = result.primal[:control_dim]
  action_lower = jp.asarray(
      (
          (sagittal_range[0] - velocity_command[0]) / dt,
          (lateral_range[0] - velocity_command[1]) / dt,
          yaw_range[0],
      ),
      dtype=dtype,
  )
  action_upper = jp.asarray(
      (
          (sagittal_range[1] - velocity_command[0]) / dt,
          (lateral_range[1] - velocity_command[1]) / dt,
          yaw_range[1],
      ),
      dtype=dtype,
  )
  action = jp.where(
      config.qp.enforce_input_bounds,
      jp.clip(action, action_lower, action_upper),
      action,
  )
  hard_condition = (
      values.lf_h + values.lg_h @ action + config.dpcbf.alpha * values.h
  )
  minimum_relaxation = jp.max(
      jp.where(values.valid, jp.maximum(-hard_condition, 0.0), 0.0)
  )
  numerical_failure = (
      ~jp.all(jp.isfinite(result.primal))
      | ~jp.isfinite(result.residual)
      | ~jp.isfinite(minimum_relaxation)
  )
  minimum_relaxation = jp.where(
      numerical_failure, jp.asarray(jp.inf, dtype=dtype), minimum_relaxation
  )
  feasible = minimum_relaxation <= diagnostic.tolerance
  return HardFeasibilityResult(
      minimum_relaxation=minimum_relaxation,
      feasible=feasible & ~numerical_failure,
      numerical_failure=numerical_failure,
  )
