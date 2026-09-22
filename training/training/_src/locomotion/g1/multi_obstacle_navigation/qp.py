"""Fixed-shape primal-dual interior-point QP solver for JAX/MJX.

Environment batching is supplied by
``jax.vmap``.  PPO never differentiates through an MJX transition, so keeping
the solver forward-only avoids carrying an unnecessary custom VJP.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp


class QpResult(NamedTuple):
  primal: jax.Array
  residual: jax.Array
  iterations: jax.Array


def _step_length(value: jax.Array, delta: jax.Array) -> jax.Array:
  """Largest step in ``[0, 1]`` that keeps a positive vector nonnegative."""
  ratio = jp.where(delta < 0.0, -value / delta, jp.inf)
  return jp.minimum(1.0, jp.min(ratio))


def solve_inequality_qp(
    q: jax.Array,
    p: jax.Array,
    g: jax.Array,
    h: jax.Array,
    *,
    max_iter: int = 40,
    tolerance: float = 1.0e-5,
    regularization: float = 1.0e-6,
) -> QpResult:
  """Solves ``min .5*x'Qx+p'x`` subject to ``Gx<=h``.

  The factorization and predictor-corrector equations follow the JAX qpth
  implementation referenced by the navigation design.  All iteration counts
  and shapes are static for XLA compilation.
  """
  variable_count = q.shape[0]
  constraint_count = g.shape[0]
  eye = jp.eye(variable_count, dtype=q.dtype)
  # Symmetrize Q and add a small regularization to avoid singularity.
  q = 0.5 * (q + q.T) + regularization * eye

  # Q and G are constant during an IPM solve.  This Schur base is reused at
  # every predictor/corrector iteration; only diag(1 / d) changes.
  inverse_q_gt = jp.linalg.solve(q, g.T)
  schur_base = g @ inverse_q_gt

  def solve_kkt(d, rx, rs, rz):
    inverse_q_rx = jp.linalg.solve(q, rx)
    schur = schur_base + jp.diag(1.0 / d)
    rhs = g @ inverse_q_rx + rs / d - rz
    dz = jp.linalg.solve(schur, -rhs)
    dx = jp.linalg.solve(q, -rx - g.T @ dz)
    ds = (-rs - dz) / d
    return dx, ds, dz

  # qpth's shifted primal-dual initialization is markedly more reliable than
  # an arbitrary infeasible slack for close-obstacle DPCBF rows.
  ones = jp.ones((constraint_count,), dtype=q.dtype)
  x, s, z = solve_kkt(ones, p, jp.zeros_like(h), -h)
  min_s = jp.min(s)
  s = jp.where(min_s < 0.0, s - min_s + 1.0, s)
  min_z = jp.min(z)
  z = jp.where(min_z < 0.0, z - min_z + 1.0, z)

  best_residual = jp.asarray(jp.inf, dtype=q.dtype)
  initial = (
      x,
      s,
      z,
      x,
      best_residual,
      jp.asarray(0, dtype=jp.int32),
      jp.asarray(True),
  )

  def iteration(_, carry):
    x, s, z, best_x, best_residual, iterations, active = carry
    rx = g.T @ z + q @ x + p
    rz = g @ x + s - h
    mu = jp.abs(jp.sum(s * z) / constraint_count)
    residual = jp.linalg.norm(rz) + jp.linalg.norm(rx) + constraint_count * mu
    finite = jp.isfinite(residual)
    improved = active & finite & (residual < best_residual)
    best_x = jp.where(improved, x, best_x)
    best_residual = jp.where(improved, residual, best_residual)

    d = z / jp.maximum(s, 1.0e-12)
    dx_aff, ds_aff, dz_aff = solve_kkt(d, rx, z, rz)
    alpha_aff = jp.minimum(_step_length(z, dz_aff), _step_length(s, ds_aff))    # Prevent s and z from being negative. (s,z > 0)
    s_aff = s + alpha_aff * ds_aff
    z_aff = z + alpha_aff * dz_aff

    # sigma = (mu_aff / mu) ^ 3
    # small sigma: Newton is making excellent progress => trust Newton and take a large step.
    # large sigma: Newton is drifting away from the central path => emphasize the centering term and take a smaller step.
    sigma = jp.clip(
        (jp.sum(s_aff * z_aff) / jp.maximum(jp.sum(s * z), 1.0e-12)) ** 3,
        0.0,
        1.0,
    )

    # Mehrotra corrector, expressed in qpth's scaled complementarity RHS.
    # centering term (-sigma * mu * 1): wants the corrected step to move toward the central path.
    # second-order correction term (ds_aff * dz_aff): uses the affine direction to estimate the neglected term
    #                                                 in the Newton linearization.
    rs_corrector = (-mu * sigma + ds_aff * dz_aff) / jp.maximum(s, 1.0e-12)

    # Solve the corrector direction
    dx_cor, ds_cor, dz_cor = solve_kkt(
        d, jp.zeros_like(x), rs_corrector, jp.zeros_like(z)
    )

    # Combine predictor and corrector directions
    dx, ds, dz = dx_aff + dx_cor, ds_aff + ds_cor, dz_aff + dz_cor
    alpha = jp.minimum(
        1.0,
        0.999 * jp.minimum(_step_length(z, dz), _step_length(s, ds)),    # Ensure s and z positive. (s,z > 0)
    )
    should_step = active & finite & (best_residual > tolerance)

    x_next = jp.where(should_step, x + alpha * dx, x)
    s_next = jp.where(should_step, s + alpha * ds, s)
    z_next = jp.where(should_step, z + alpha * dz, z)

    return (
        x_next,
        jp.maximum(s_next, 1.0e-12),
        jp.maximum(z_next, 1.0e-12),
        best_x,
        best_residual,
        iterations + should_step.astype(jp.int32),
        should_step,
    )

  _, _, _, best_x, best_residual, iterations, _ = jax.lax.fori_loop(
      0, max_iter, iteration, initial
  )
  return QpResult(best_x, best_residual, iterations)


batched_solve_inequality_qp = jax.vmap(
    solve_inequality_qp, in_axes=(0, 0, 0, 0), out_axes=0
)
