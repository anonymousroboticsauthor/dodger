"""Play a trained G1 navigation policy with continuously sampled goals.

This dedicated entry point deliberately supports random chained goals only.
Interactive/manual goal sources belong in the later deployment interface.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument(
      "--output_dir",
      type=Path,
      default=None,
      help=(
          "Root directory for play videos. By default, videos are stored in "
          "the training run's play/ directory."
      ),
  )
  parser.add_argument("--episode_length", type=int, default=900)
  parser.add_argument("--num_videos", type=int, default=1)
  parser.add_argument(
      "--runtime_filter", action=argparse.BooleanOptionalAction, default=False
  )
  parser.add_argument("--impl", choices=("jax", "warp"), default="jax")
  args = parser.parse_args()
  if not args.checkpoint.exists():
    raise FileNotFoundError(args.checkpoint)
  checkpoint = args.checkpoint.resolve()
  checkpoints_dir = next(
      (
          path
          for path in (checkpoint, *checkpoint.parents)
          if path.name == "checkpoints"
      ),
      None,
  )
  if checkpoints_dir is None:
    checkpoint_label = checkpoint.name
  elif checkpoint == checkpoints_dir:
    checkpoint_candidates = [
        path
        for path in checkpoints_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    if not checkpoint_candidates:
      raise FileNotFoundError(
          f"No numbered checkpoints found in {checkpoints_dir}"
      )
    checkpoint_label = max(
        checkpoint_candidates, key=lambda path: int(path.name)
    ).name
  else:
    relative_parts = checkpoint.relative_to(checkpoints_dir).parts
    checkpoint_label = relative_parts[0]
  if args.output_dir is not None:
    output_dir = args.output_dir.resolve()
  elif checkpoints_dir is not None:
    output_dir = checkpoints_dir.parent / "play"
  else:
    output_dir = checkpoint.parent / "play"
  experiment_name = f"checkpoint_{checkpoint_label}"
  overrides = {
      "play_mode": True,
      "dpcbf.filter_actions": args.runtime_filter,
      "perception.enabled": False,
      "robot_state_randomization.enabled": False,
  }
  command = [
      sys.executable,
      "-m",
      "learning.train_jax_ppo",
      "--env_name=G1MultiObstacleNavigation",
      "--play_only",
      "--timestamp_videos",
      f"--experiment_name={experiment_name}",
      f"--load_checkpoint_path={checkpoint}",
      f"--logdir={output_dir}",
      f"--episode_length={args.episode_length}",
      f"--num_videos={args.num_videos}",
      f"--impl={args.impl}",
      "--config_overrides=" + json.dumps(overrides),
  ]
  subprocess.run(command, check=True)


if __name__ == "__main__":
  main()
