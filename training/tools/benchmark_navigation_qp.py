"""Benchmark G1 DPCBF QP with velocity/yaw bounds enabled and disabled."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import time

# Leave room for cuBLAS/cuSolver handles and the desktop compositor on GPUs
# shared with the display server.  This must happen before importing JAX.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jp

from training._src.locomotion.g1.multi_obstacle_navigation import (
    config as nav_config,
)
from training._src.locomotion.g1.multi_obstacle_navigation import dpcbf


def make_batch(key, batch_size, count):
  pos_key, vel_key, radius_key, action_key, command_key = jax.random.split(
      key, 5
  )
  angle = jax.random.uniform(pos_key, (batch_size, count), maxval=2 * jp.pi)
  distance = jax.random.uniform(
      pos_key, (batch_size, count), minval=0.7, maxval=3.0
  )
  position = distance[..., None] * jp.stack(
      (jp.cos(angle), jp.sin(angle)), axis=-1
  )
  velocity = 0.6 * jax.random.normal(vel_key, (batch_size, count, 2))
  radius = jax.random.uniform(
      radius_key, (batch_size, count), minval=0.2, maxval=0.3
  )
  unit_action = jax.random.uniform(
      action_key, (batch_size, 3), minval=-1.0, maxval=1.0
  )
  action = jp.asarray((1.0, 0.0, 0.0)) + unit_action * jp.asarray(
      (3.0, 2.0, 1.5)
  )
  command = jax.random.uniform(
      command_key, (batch_size, 3), minval=-1.0, maxval=1.0
  )
  command = command.at[:, 0].set(0.5 + 1.5 * command[:, 0])
  selected = dpcbf.SelectedObstacles(
      relative_position=position,
      relative_velocity=velocity,
      radius=radius,
      valid=jp.ones((batch_size, count), dtype=bool),
      indices=jp.broadcast_to(jp.arange(count), (batch_size, count)),
  )
  return selected, command, action


def benchmark(enforce_bounds: bool, batch_size: int, repeats: int):
  config = nav_config.default_config()
  config.qp.enforce_input_bounds = enforce_bounds
  count = config.obstacle.num_constraints
  selected, command, action = make_batch(
      jax.random.PRNGKey(0), batch_size, count
  )
  yaw = jp.zeros((batch_size,))
  velocity = jp.zeros((batch_size, 2))

  solve = jax.jit(
      jax.vmap(
          lambda selected, yaw, velocity, command, action: dpcbf.filter_action(
              selected, yaw, velocity, command, action, config
          )
      )
  )
  compile_start = time.perf_counter()
  result = solve(selected, yaw, velocity, command, action)
  result.safe_action.block_until_ready()
  compile_seconds = time.perf_counter() - compile_start
  start = time.perf_counter()
  for _ in range(repeats):
    result = solve(selected, yaw, velocity, command, action)
  result.safe_action.block_until_ready()
  elapsed = time.perf_counter() - start
  safe_violation = jp.where(
      result.values.valid, jp.minimum(result.safe_condition, 0.0), 0.0
  )
  dt = 1.0 / dpcbf.QP_HZ
  next_command = command.at[:, :2].set(
      command[:, :2] + result.safe_action[:, :2] * dt
  )
  next_command = next_command.at[:, 2].set(result.safe_action[:, 2])
  lower = jp.asarray((
      config.robot.sagittal_velocity_range[0],
      config.robot.lateral_velocity_range[0],
      config.robot.yaw_rate_range[0],
  ))
  upper = jp.asarray((
      config.robot.sagittal_velocity_range[1],
      config.robot.lateral_velocity_range[1],
      config.robot.yaw_rate_range[1],
  ))
  bound_violation = jp.maximum(
      jp.maximum(lower - next_command, next_command - upper), 0.0
  )
  raw_bound_violation_max = jp.max(bound_violation)
  if enforce_bounds:
    next_command = jp.clip(next_command, lower, upper)
    bound_violation = jp.maximum(
        jp.maximum(lower - next_command, next_command - upper), 0.0
    )
  bounds = "on" if enforce_bounds else "off"
  print(
      f"bounds={bounds} batch={batch_size} compile_s={compile_seconds:.4f} "
      f"step_ms={1e3 * elapsed / repeats:.4f} "
      f"env_qp_per_s={batch_size * repeats / elapsed:.1f} "
      f"residual_max={float(jp.max(result.qp_residual)):.3e} "
      f"failed_rate={float(jp.mean(result.qp_failed)):.3%} "
      f"slack_mean={float(jp.mean(result.slack)):.3e} "
      f"safe_violation_min={float(jp.min(safe_violation)):.3e} "
      f"raw_bound_violation_max={float(raw_bound_violation_max):.3e} "
      f"bound_violation_max={float(jp.max(bound_violation)):.3e}"
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size", type=int, default=1024)
  parser.add_argument("--repeats", type=int, default=100)
  args = parser.parse_args()
  for enforce_bounds in (False, True):
    benchmark(enforce_bounds, args.batch_size, args.repeats)


if __name__ == "__main__":
  main()
