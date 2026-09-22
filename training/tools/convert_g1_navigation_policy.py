"""Convert the Unitree RL MjLab G1 ONNX velocity policy to JAX weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "training/_src/locomotion/g1/multi_obstacle_navigation/assets"
    / "low_level_policy.npz"
)


def convert(input_path: Path, output_path: Path) -> None:
  model = onnx.load(str(input_path))
  initializers = {
      item.name: np.asarray(numpy_helper.to_array(item), dtype=np.float32)
      for item in model.graph.initializer
  }
  inputs = [
      (
          value.name,
          tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim),
      )
      for value in model.graph.input
  ]
  outputs = [
      (
          value.name,
          tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim),
      )
      for value in model.graph.output
  ]
  if not inputs or inputs[0][1][-1] != 98:
    raise ValueError(f"expected ONNX input dimension 98, got {inputs}")
  if not outputs or outputs[0][1][-1] != 29:
    raise ValueError(f"expected ONNX output dimension 29, got {outputs}")
  node_types = [node.op_type for node in model.graph.node]
  if node_types.count("Elu") != 3 and node_types.count("ELU") != 3:
    raise ValueError(f"expected three ELU nodes, got {node_types}")

  names = (0, 2, 4, 6)
  weights = [initializers[f"mlp.{index}.weight"] for index in names]
  biases = [initializers[f"mlp.{index}.bias"] for index in names]
  expected = ((512, 98), (256, 512), (128, 256), (29, 128))
  if tuple(weight.shape for weight in weights) != expected:
    raise ValueError("unexpected ONNX MLP layer shapes")
  mean = initializers["obs_normalizer._mean"].reshape(-1)
  std = initializers["onnx::Div_24"].reshape(-1)
  if mean.shape != (98,) or std.shape != (98,):
    raise ValueError("unexpected observation normalization shape")
  if np.any(std <= 0.0):
    raise ValueError("observation standard deviation must be positive")

  # Numerical equivalence is checked before writing anything.  This catches
  # transposed Gemm weights, a wrong normalizer tensor, or activation drift.
  test_observation = (
      np.random.default_rng(0).normal(size=(1, 98)).astype(np.float32)
  )
  reference = ReferenceEvaluator(model).run(
      None, {inputs[0][0]: test_observation}
  )[0]
  hidden = (test_observation.reshape(-1) - mean) / std
  for weight, bias in zip(weights[:-1], biases[:-1]):
    hidden = weight @ hidden + bias
    hidden = np.where(hidden >= 0.0, hidden, np.expm1(hidden))
  converted = weights[-1] @ hidden + biases[-1]
  np.testing.assert_allclose(
      converted, reference.reshape(-1), rtol=2e-5, atol=2e-5
  )

  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
      output_path,
      mean=mean,
      std=std,
      **{f"weight_{i}": value for i, value in enumerate(weights)},
      **{f"bias_{i}": value for i, value in enumerate(biases)},
  )
  print(f"Converted {input_path} -> {output_path}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  args = parser.parse_args()
  convert(args.input, args.output)


if __name__ == "__main__":
  main()
