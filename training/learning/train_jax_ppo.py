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
"""Train a PPO agent using JAX on the specified environment."""

# ruff: noqa: E402

import datetime
import functools
import json
import os
import sys
import threading
import time
import warnings

# Configure CUDA allocation and headless rendering before importing JAX or
# MuJoCo.  Doing this after those imports can be too late for a console entry
# point and may leave too little free memory for a cuSolver handle.
xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_triton_gemm_any" not in xla_flags:
  xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jp
import mediapy as media
import mujoco
from absl import app, flags, logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from etils import epath
from ml_collections import config_dict

import training
from learning.ppo import train as ppo
from training import registry, wrapper
from training._src.locomotion.g1.multi_obstacle_navigation import (
    networks as navigation_networks,
)
from training.config import (
    dm_control_suite_params,
    locomotion_params,
    manipulation_params,
)

try:
  import tensorboardX
except ImportError:
  tensorboardX = None

try:
  import wandb
except ImportError:
  wandb = None


def _finish_wandb_with_timeout(timeout_seconds: float) -> None:
  """Finalizes W&B without allowing its service thread to block shutdown."""
  finished = threading.Event()
  errors = []

  def finish():
    try:
      wandb.finish()
    except Exception as exc:  # pylint: disable=broad-exception-caught
      errors.append(exc)
    finally:
      finished.set()

  # A daemon thread cannot keep Python alive.  More importantly, main waits
  # only for the explicit deadline below before the successful-exit path calls
  # os._exit, so a stuck W&B service cannot retain the training command.
  thread = threading.Thread(
      target=finish, name="wandb-finish", daemon=True
  )
  thread.start()
  completed = finished.wait(timeout_seconds)
  if not completed:
    print(
        "Warning: W&B did not finish within "
        f"{timeout_seconds:g} seconds; local run data is preserved and the "
        "completed training process will exit now."
    )
  elif errors:
    print(f"Warning: W&B finalization failed; exiting anyway: {errors[0]}")


def _start_successful_shutdown_watchdog(timeout_seconds: float) -> None:
  """Forces exit if successful-run cleanup blocks after artifacts are saved."""

  def force_exit():
    time.sleep(timeout_seconds)
    print(
        "Warning: post-training cleanup exceeded "
        f"{timeout_seconds:g} seconds; forcing successful process exit."
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

  threading.Thread(
      target=force_exit, name="training-shutdown-watchdog", daemon=True
  ).start()


# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "LeapCubeReorient",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_CONFIG_OVERRIDES = flags.DEFINE_string(
    "config_overrides",
    None,
    "Overrides for the training environment config.",
)
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input")
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_EXPERIMENT_NAME = flags.DEFINE_string(
    "experiment_name", None, "Exact experiment directory and run name."
)
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    True,
    "Use Weights & Biases for logging (ignored in play-only mode)",
)
_WANDB_FINISH_TIMEOUT_SECONDS = flags.DEFINE_float(
    "wandb_finish_timeout_seconds",
    60.0,
    "Maximum seconds to wait for W&B uploads when a run finishes.",
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "Use TensorBoard for logging (ignored in play-only mode)"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 1_000_000, "Number of timesteps"
)
_NUM_VIDEOS = flags.DEFINE_integer(
    "num_videos", 1, "Number of videos to record after training."
)
_TIMESTAMP_VIDEOS = flags.DEFINE_boolean(
    "timestamp_videos",
    False,
    "Append a timestamp to rollout filenames to avoid overwriting videos.",
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 10, "Unroll length")
_NUM_MINIBATCHES = flags.DEFINE_integer(
    "num_minibatches", 8, "Number of minibatches"
)
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 8, "Number of updates per batch"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 5e-4, "Learning rate")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "Entropy cost")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm")
_CLIPPING_EPSILON = flags.DEFINE_float(
    "clipping_epsilon", 0.3, "Clipping epsilon for PPO"
)
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [64, 64, 64],
    "Policy hidden layer sizes",
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [64, 64, 64],
    "Value hidden layer sizes",
)
_POLICY_OBS_KEY = flags.DEFINE_string(
    "policy_obs_key", "state", "Policy obs key"
)
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "Value obs key")
_RSCOPE_ENVS = flags.DEFINE_integer(
    "rscope_envs",
    None,
    "Number of parallel environment rollouts to save for the rscope viewer",
)
_DETERMINISTIC_RSCOPE = flags.DEFINE_boolean(
    "deterministic_rscope",
    True,
    "Run deterministic rollouts for the rscope viewer",
)
_RUN_EVALS = flags.DEFINE_boolean(
    "run_evals",
    True,
    "Run evaluation rollouts between policy updates.",
)
_LOG_TRAINING_METRICS = flags.DEFINE_boolean(
    "log_training_metrics",
    False,
    "Whether to log training metrics and callback to progress_fn. Significantly"
    " slows down training if too frequent.",
)
_TRAINING_METRICS_STEPS = flags.DEFINE_integer(
    "training_metrics_steps",
    1_000_000,
    "Number of steps between logging training metrics. Increase if training"
    " experiences slowdown.",
)
_WARP_KERNEL_CACHE_DIR = flags.DEFINE_string(
    "warp_kernel_cache_dir",
    None,
    "Directory for caching compiled Warp kernels.",
)
_LOGDIR = flags.DEFINE_string("logdir", None, "Directory for logging.")


class _EvaluationGate:
  """Enables eval after a fixed step or the final dynamic curriculum stage."""

  def __init__(self, start_step=None, target_stage=None):
    self.start_step = start_step
    self.target_stage = target_stage
    self.current_step = 0
    self.current_stage = 0.0

  def __bool__(self) -> bool:
    if self.start_step is not None:
      return self.current_step >= self.start_step
    if self.target_stage is not None:
      return self.current_stage >= self.target_stage
    return True


class _EarlyStopTraining(Exception):
  """Signals successful evaluation convergence without losing latest params."""


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  if env_name in training.manipulation._envs:
    if _VISION.value:
      return manipulation_params.brax_vision_ppo_config(env_name, _IMPL.value)
    return manipulation_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in training.locomotion._envs:
    return locomotion_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in training.dm_control_suite._envs:
    if _VISION.value:
      return dm_control_suite_params.brax_vision_ppo_config(
          env_name, _IMPL.value
      )
    return dm_control_suite_params.brax_ppo_config(env_name, _IMPL.value)

  raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def rscope_fn(full_states, obs, rew, done):
  """
  All arrays are of shape (unroll_length, rscope_envs, ...)
  full_states: dict with keys 'qpos', 'qvel', 'time', 'metrics'
  obs: nd.array or dict obs based on env configuration
  rew: nd.array rewards
  done: nd.array done flags
  """
  # Calculate cumulative rewards per episode, stopping at first done flag
  done_mask = jp.cumsum(done, axis=0)
  valid_rewards = rew * (done_mask == 0)
  episode_rewards = jp.sum(valid_rewards, axis=0)
  print(
      "Collected rscope rollouts with reward"
      f" {episode_rewards.mean():.3f} +- {episode_rewards.std():.3f}"
  )


def main(argv):
  """Run training and evaluation for the specified environment."""

  del argv

  if _WARP_KERNEL_CACHE_DIR.value is not None:
    import warp as wp  # pylint: disable=g-import-not-at-top  # noqa: PLC0415

    wp.config.kernel_cache_dir = _WARP_KERNEL_CACHE_DIR.value

  # Load environment configuration
  env_cfg = registry.get_default_config(_ENV_NAME.value)

  ppo_params = get_rl_config(_ENV_NAME.value)

  if _NUM_TIMESTEPS.present:
    ppo_params.num_timesteps = _NUM_TIMESTEPS.value
  if _PLAY_ONLY.present:
    ppo_params.num_timesteps = 0
  if _NUM_EVALS.present:
    ppo_params.num_evals = _NUM_EVALS.value
  if _REWARD_SCALING.present:
    ppo_params.reward_scaling = _REWARD_SCALING.value
  if _EPISODE_LENGTH.present:
    ppo_params.episode_length = _EPISODE_LENGTH.value
  if _NORMALIZE_OBSERVATIONS.present:
    ppo_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
  if _ACTION_REPEAT.present:
    ppo_params.action_repeat = _ACTION_REPEAT.value
  if _UNROLL_LENGTH.present:
    ppo_params.unroll_length = _UNROLL_LENGTH.value
  if _NUM_MINIBATCHES.present:
    ppo_params.num_minibatches = _NUM_MINIBATCHES.value
  if _NUM_UPDATES_PER_BATCH.present:
    ppo_params.num_updates_per_batch = _NUM_UPDATES_PER_BATCH.value
  if _DISCOUNTING.present:
    ppo_params.discounting = _DISCOUNTING.value
  if _LEARNING_RATE.present:
    ppo_params.learning_rate = _LEARNING_RATE.value
  if _ENTROPY_COST.present:
    ppo_params.entropy_cost = _ENTROPY_COST.value
  if _NUM_ENVS.present:
    ppo_params.num_envs = _NUM_ENVS.value
  if _NUM_EVAL_ENVS.present:
    ppo_params.num_eval_envs = _NUM_EVAL_ENVS.value
  if _BATCH_SIZE.present:
    ppo_params.batch_size = _BATCH_SIZE.value
  if _MAX_GRAD_NORM.present:
    ppo_params.max_grad_norm = _MAX_GRAD_NORM.value
  if _CLIPPING_EPSILON.present:
    ppo_params.clipping_epsilon = _CLIPPING_EPSILON.value
  if _POLICY_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.policy_hidden_layer_sizes = list(
        map(int, _POLICY_HIDDEN_LAYER_SIZES.value)
    )
  if _VALUE_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.value_hidden_layer_sizes = list(
        map(int, _VALUE_HIDDEN_LAYER_SIZES.value)
    )
  if _POLICY_OBS_KEY.present:
    ppo_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
  if _VALUE_OBS_KEY.present:
    ppo_params.network_factory.value_obs_key = _VALUE_OBS_KEY.value

  env_cfg_overrides = {"impl": _IMPL.value}
  if _VISION.value:
    env_cfg_overrides["vision"] = True
    env_cfg_overrides["vision_config.nworld"] = ppo_params.num_envs
  if _CONFIG_OVERRIDES.value is not None:
    env_cfg_overrides.update(json.loads(_CONFIG_OVERRIDES.value))

  env = registry.load(
      _ENV_NAME.value, config=env_cfg, config_overrides=env_cfg_overrides
  )
  use_split_value_critic = bool(getattr(env, "dodger_split_value_critic", False))
  ppo_params.use_split_value_critic = use_split_value_critic
  if _RUN_EVALS.present:
    ppo_params.run_evals = _RUN_EVALS.value
  if _LOG_TRAINING_METRICS.present:
    ppo_params.log_training_metrics = _LOG_TRAINING_METRICS.value
  if _TRAINING_METRICS_STEPS.present:
    ppo_params.training_metrics_steps = _TRAINING_METRICS_STEPS.value

  print(f"Environment Config:\n{env_cfg}")
  if env_cfg_overrides:
    print(f"Environment Config Overrides:\n{env_cfg_overrides}\n")
  print(f"PPO Training Parameters:\n{ppo_params}")

  # Generate unique experiment name
  now = datetime.datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  if _EXPERIMENT_NAME.value is not None:
    exp_name = _EXPERIMENT_NAME.value
  else:
    exp_name = f"{_ENV_NAME.value}-{timestamp}"
    if _SUFFIX.value is not None:
      exp_name += f"-{_SUFFIX.value}"
  print(f"Experiment name: {exp_name}")

  # Set up logging directory
  logdir = epath.Path(_LOGDIR.value or "logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  # Initialize Weights & Biases if required
  if _USE_WANDB.value and not _PLAY_ONLY.value:
    if wandb is None:
      raise ImportError(
          "wandb is required for --use_wandb. Install via: pip install wandb"
      )
    if _WANDB_FINISH_TIMEOUT_SECONDS.value <= 0.0:
      raise ValueError("wandb_finish_timeout_seconds must be positive")
    wandb.init(
        project="dodger",
        name=exp_name,
        settings=wandb.Settings(
            finish_timeout=_WANDB_FINISH_TIMEOUT_SECONDS.value,
            finish_timeout_raises=False,
        ),
    )
    wandb.config.update(env_cfg.to_dict())
    wandb.config.update({"env_name": _ENV_NAME.value})

  # Initialize TensorBoard if required
  writer = None
  if _USE_TB.value and not _PLAY_ONLY.value and tensorboardX is not None:
    writer = tensorboardX.SummaryWriter(logdir)

  # Handle checkpoint loading
  if _LOAD_CHECKPOINT_PATH.value is not None:
    # Convert to absolute path
    ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
    if ckpt_path.is_dir() and ckpt_path.name.isdigit():
      restore_checkpoint_path = ckpt_path
      print(f"Restoring from checkpoint: {restore_checkpoint_path}")
    elif ckpt_path.is_dir():
      latest_ckpts = list(ckpt_path.glob("*"))
      latest_ckpts = [
          ckpt for ckpt in latest_ckpts if ckpt.is_dir() and ckpt.name.isdigit()
      ]
      if not latest_ckpts:
        raise FileNotFoundError(f"No numbered checkpoints found in {ckpt_path}")
      latest_ckpts.sort(key=lambda x: int(x.name))
      latest_ckpt = latest_ckpts[-1]
      restore_checkpoint_path = latest_ckpt
      print(f"Restoring from: {restore_checkpoint_path}")
    else:
      restore_checkpoint_path = ckpt_path
      print(f"Restoring from checkpoint: {restore_checkpoint_path}")
  else:
    print("No checkpoint path provided, not restoring from checkpoint")
    restore_checkpoint_path = None

  # Set up checkpoint directory
  ckpt_path = logdir / "checkpoints"
  ckpt_path.mkdir(parents=True, exist_ok=True)
  print(f"Checkpoint path: {ckpt_path}")

  # Save environment configuration
  with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
    # ConfigDict.to_dict does not recursively unwrap ConfigDict objects stored
    # inside curriculum stage lists, whereas to_json does.
    json.dump(json.loads(env_cfg.to_json()), fp, indent=4)

  training_params = dict(ppo_params)
  custom_network_factory = training_params.pop("custom_network_factory", None)
  if "network_factory" in training_params:
    del training_params["network_factory"]

  if custom_network_factory == "g1_navigation_gat":
    network_fn = navigation_networks.make_gat_ppo_networks
  else:
    network_fn = (
        ppo_networks_vision.make_ppo_networks_vision
        if _VISION.value
        else ppo_networks.make_ppo_networks
    )
  if hasattr(ppo_params, "network_factory"):
    network_factory_kwargs = dict(ppo_params.network_factory)
    if custom_network_factory == "g1_navigation_gat":
      network_factory_kwargs["split_value_critic"] = use_split_value_critic
    network_factory = functools.partial(network_fn, **network_factory_kwargs)
  else:
    network_factory = network_fn

  if _DOMAIN_RANDOMIZATION.value:
    training_params["randomization_fn"] = registry.get_domain_randomizer(
        _ENV_NAME.value
    )

  num_eval_envs = ppo_params.get("num_eval_envs", 128)

  if "num_eval_envs" in training_params:
    del training_params["num_eval_envs"]

  if _ENV_NAME.value != "G1MultiObstacleNavigation":
    evaluation_gate = _EvaluationGate(start_step=0)
  else:
    evaluation_gate = _EvaluationGate(
        target_stage=len(env_cfg.curriculum.stages) - 1
    )
  if training_params.get("run_evals", True):
    training_params["run_evals"] = evaluation_gate

  train_fn = functools.partial(
      ppo.train,
      **training_params,
      network_factory=network_factory,
      seed=_SEED.value,
      restore_checkpoint_path=restore_checkpoint_path,
      save_checkpoint_path=ckpt_path,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      num_eval_envs=num_eval_envs,
      vision=_VISION.value,
  )

  times = [time.monotonic()]

  def wandb_metrics(metrics):
    """Groups Brax metrics into meaningful W&B sections without duplicates."""
    grouped = {}
    loss_names = {
        "total_loss": "total",
        "policy_loss": "policy",
        "v_loss": "value",
        "v_loss_positive": "value_positive",
        "v_loss_negative": "value_negative",
        "entropy_loss": "entropy",
        "kl_mean": "kl",
        "kl_candidate": "kl_candidate",
        "kl_candidate_max": "kl_candidate_max",
        "kl_limit_triggered": "kl_limit_triggered",
        "kl_update_skipped": "kl_update_skipped",
        "effective_minibatch_update": "effective_minibatch_update",
        "kl_policy_update_scale": "kl_policy_update_scale",
    }

    for key, value in metrics.items():
      if key == "episode/sum_reward":
        # EpisodeMetricsLogger already averages the returns of the most recent
        # completed training episodes (up to its 100-episode buffer).
        grouped["Training/mean_reward"] = value
      elif key.startswith("episode/reward/"):
        name = key.removeprefix("episode/reward/")
        grouped[f"Reward/{name}"] = value
      elif key.startswith("episode/navigation/termination_"):
        name = key.removeprefix("episode/navigation/termination_")
        grouped[f"Termination/{name}"] = value
      elif key == "episode/navigation/curriculum_stage_per_step":
        grouped["Curriculum/stage"] = value
      elif key == "episode/navigation/curriculum_success_rate_per_step":
        grouped["Curriculum/success_rate"] = value
      elif key.startswith("episode/navigation/dodger_"):
        name = key.removeprefix("episode/navigation/dodger_")
        name = name.removesuffix("_per_step")
        grouped[f"DODGER/{name}"] = value
      elif (
          key.startswith("episode/navigation/hard_cbf_")
          and not env_cfg.qp.hard_feasibility.enabled
      ):
        continue
      elif key.startswith("episode/navigation/"):
        name = key.removeprefix("episode/navigation/")
        name = name.removesuffix("_per_step")
        grouped[f"Navigation/{name}"] = value
      elif key == "episode/length":
        grouped["Training/episode_length"] = value
      elif key == "episode/sps":
        grouped["Performance/training_metrics_sps"] = value
      elif key.startswith("episode/"):
        name = key.removeprefix("episode/")
        if name in loss_names:
          grouped[f"Loss/{loss_names[name]}"] = value
        elif name == "learning_rate":
          grouped[f"Training/{name}"] = value
        elif name.startswith("policy_dist_"):
          continue
        else:
          grouped[key] = value
      elif key == "eval/episode_reward":
        grouped["Eval/reward/total"] = value
      elif key == "eval/episode_navigation/termination_goal_reached":
        grouped["Eval/success_rate"] = value
      elif key == "eval/episode_reward_std":
        pass
        # grouped["Eval/reward/total_std"] = value
      elif key.startswith("eval/episode_reward/"):
        pass
        # name = key.removeprefix("eval/episode_reward/")
        # grouped[f"Eval/reward/{name}"] = value
      elif key.startswith("eval/episode_navigation/termination_"):
        pass
        # name = key.removeprefix("eval/episode_navigation/termination_")
        # grouped[f"Eval/termination/{name}"] = value
      elif key.startswith("eval/episode_navigation/curriculum_stage_per_step"):
        pass
        # suffix = key.removeprefix(
        #     "eval/episode_navigation/curriculum_stage_per_step"
        # )
        # grouped[f"Eval/curriculum/stage{suffix}"] = value
      elif key.startswith("eval/episode_navigation/"):
        pass
        # name = key.removeprefix("eval/episode_navigation/")
        # grouped[f"Eval/navigation/{name}"] = value
      elif key == "eval/avg_episode_length":
        pass
        # grouped["Eval/episode/length"] = value
      elif key == "eval/std_episode_length":
        pass
        # grouped["Eval/episode/length_std"] = value
      elif key.startswith("training/"):
        name = key.removeprefix("training/")
        if name in loss_names:
          grouped[f"Loss/{loss_names[name]}"] = value
        elif name.startswith("value_decomposition/"):
          metric_name = name.removeprefix("value_decomposition/")
          grouped[f"ValueDecomposition/{metric_name}"] = value
        elif name == "sps":
          grouped["Performance/training_sps"] = value
        elif name == "walltime":
          grouped["Performance/training_walltime"] = value
        else:
          grouped[key] = value
      elif key == "eval/sps":
        pass
        # grouped["Performance/eval_sps"] = value
      elif key == "eval/walltime":
        pass
        # grouped["Performance/eval_walltime"] = value
      elif key == "eval/epoch_eval_time":
        pass
        # grouped["Performance/epoch_eval_time"] = value
      else:
        grouped[key] = value

    return grouped

  # Progress function for logging
  early_stop_success_rates = []

  def progress(num_steps, metrics):
    nonlocal early_stop_success_rates
    times.append(time.monotonic())
    training_stage_key = "episode/navigation/curriculum_stage_per_step"
    if training_stage_key in metrics:
      evaluation_gate.current_stage = float(metrics[training_stage_key])
      if (
          _ENV_NAME.value == "G1MultiObstacleNavigation"
          and evaluation_gate.current_stage < env_cfg.curriculum.eval_stage
      ):
        early_stop_success_rates = []

    # Log to Weights & Biases
    if _USE_WANDB.value and not _PLAY_ONLY.value:
      wandb.log(wandb_metrics(metrics), step=num_steps)

    # Log to TensorBoard
    if _USE_TB.value and not _PLAY_ONLY.value and writer is not None:
      for key, value in metrics.items():
        writer.add_scalar(key, value, num_steps)
      writer.flush()
    if "eval/episode_reward" in metrics:
      print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")
    success_key = "eval/episode_navigation/termination_goal_reached"
    early_termination = (
        _ENV_NAME.value == "G1MultiObstacleNavigation"
        and env_cfg.curriculum.early_termination
    )
    if early_termination and num_steps > 0 and success_key in metrics:
      success_rate = float(metrics[success_key])
      target = 1.0 - env_cfg.curriculum.early_stop_epsilon
      required_checks = env_cfg.curriculum.early_stop_required_checks
      early_stop_success_rates.append(success_rate)
      early_stop_success_rates = early_stop_success_rates[-required_checks:]
      collected_checks = len(early_stop_success_rates)
      if collected_checks < required_checks:
        print(
            f"Early-stop window: eval success rate {success_rate:.4f}, "
            f"collected {collected_checks}/{required_checks}; waiting for a "
            "full window."
        )
      else:
        mean_success_rate = sum(early_stop_success_rates) / required_checks
        print(
            f"Early-stop check: eval success rate {success_rate:.4f}, "
            f"last-{required_checks} mean {mean_success_rate:.4f} at threshold "
            f"{target:.4f}."
        )
        if mean_success_rate >= target:
          print(
              f"Early stopping at {num_steps}: eval success rate "
              f"mean {mean_success_rate:.4f} >= {target:.4f} over the latest "
              f"{required_checks} evaluations."
          )
          raise _EarlyStopTraining
    if _LOG_TRAINING_METRICS.value:
      if "episode/sum_reward" in metrics:
        print(
            f"{num_steps}: mean episode"
            f" reward={metrics['episode/sum_reward']:.3f}"
        )

  eval_env_overrides = dict(env_cfg_overrides)
  if _ENV_NAME.value == "G1MultiObstacleNavigation":
    eval_env_overrides["evaluation_mode"] = True
    # Evaluate the learned policy itself, without a runtime safety shield.
    # The QP result is still computed inside the environment for CBF metrics.
    eval_env_overrides["dpcbf.filter_actions"] = False
  if _VISION.value:
    eval_env_overrides["vision_config.nworld"] = num_eval_envs
  eval_env = registry.load(
      _ENV_NAME.value,
      config=registry.get_default_config(_ENV_NAME.value),
      config_overrides=eval_env_overrides,
  )

  latest_policy = {}

  def record_policy_params(current_step, make_policy, params):
    evaluation_gate.current_step = current_step
    latest_policy["make_policy"] = make_policy
    latest_policy["params"] = params

  policy_params_fn = record_policy_params

  if _RSCOPE_ENVS.value:
    # Interactive visualisation of policy checkpoints
    try:
      from learning import rscope_brax  # noqa: PLC0415
    except ImportError as exc:
      if exc.name == "rscope" or (
          exc.name is not None and exc.name.startswith("rscope.")
      ):
        raise ImportError(
            "rscope is required for --rscope_envs. Install the learning "
            "extras or run `python -m pip install rscope==0.0.8`."
        ) from exc
      raise

    if not _VISION.value:
      rscope_env = registry.load(
          _ENV_NAME.value, config=env_cfg, config_overrides=env_cfg_overrides
      )
      rscope_env = wrapper.wrap_for_brax_training(
          rscope_env,
          episode_length=ppo_params.episode_length,
          action_repeat=ppo_params.action_repeat,
          randomization_fn=training_params.get("randomization_fn"),
      )
    else:
      rscope_env = env

    rscope_handle = rscope_brax.BraxRolloutSaver(
        rscope_env,
        ppo_params,
        _VISION.value,
        _RSCOPE_ENVS.value,
        _DETERMINISTIC_RSCOPE.value,
        jax.random.PRNGKey(_SEED.value),
        rscope_fn,
    )
    print(
        "Rscope rollout capture enabled. Run `python -m rscope` in a "
        "second terminal to open the interactive viewer."
    )

    def policy_params_fn(current_step, make_policy, params):  # pylint: disable=unused-argument
      record_policy_params(current_step, make_policy, params)
      rscope_handle.set_make_policy(make_policy)
      rscope_handle.dump_rollout(params)

  # Train or load the model
  early_stopped = False
  try:
    make_inference_fn, params, _ = train_fn(  # pylint: disable=no-value-for-parameter
        environment=env,
        progress_fn=progress,
        policy_params_fn=policy_params_fn,
        eval_env=eval_env,
    )
  except _EarlyStopTraining:
    if not latest_policy:
      raise
    early_stopped = True
    make_inference_fn = latest_policy["make_policy"]
    params = latest_policy["params"]

  print("Done training." if not early_stopped else "Stopped training early.")
  if len(times) > 1:
    print(f"Time to JIT compile: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

  print("Starting inference...")

  # Create inference function.
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)

  infer_env_overrides = dict(env_cfg_overrides)
  if _ENV_NAME.value == "G1MultiObstacleNavigation":
    # Post-training deployment videos use the hardest curriculum distribution
    # and chain random goals without the training episode timeout.
    infer_env_overrides["play_mode"] = True
    if not _PLAY_ONLY.value:
      # Match filter-free hardware deployment in the automatic final video.
      # Training and evaluation environments keep their configured filter.
      infer_env_overrides["dpcbf.filter_actions"] = False
  if _VISION.value:
    infer_env_overrides["vision_config.nworld"] = _NUM_VIDEOS.value
  infer_env = registry.load(
      _ENV_NAME.value,
      config=registry.get_default_config(_ENV_NAME.value),
      config_overrides=infer_env_overrides,
  )
  inference_steps = ppo_params.episode_length
  if _ENV_NAME.value == "G1MultiObstacleNavigation" and not _PLAY_ONLY.value:
    inference_steps = round(90.0 / infer_env.dt)

  # Run evaluation rollouts matching how training handles batched environments.
  wrapped_infer_env = wrapper.wrap_for_brax_training(
      infer_env,
      episode_length=inference_steps,
      action_repeat=ppo_params.get("action_repeat", 1),
  )

  rng = jax.random.split(jax.random.PRNGKey(_SEED.value), _NUM_VIDEOS.value)
  reset_states = jax.jit(wrapped_infer_env.reset)(rng)

  empty_data = reset_states.data.__class__(
      **{k: None for k in reset_states.data.__annotations__}
  )  # pytype: disable=attribute-error
  empty_traj = reset_states.__class__(
      **{k: None for k in reset_states.__annotations__}
  )  # pytype: disable=attribute-error
  empty_traj = empty_traj.replace(data=empty_data)

  def step(carry, _):
    state, rng = carry
    rng, act_key = jax.random.split(rng)
    act_keys = jax.random.split(act_key, _NUM_VIDEOS.value)
    act = jax.vmap(jit_inference_fn)(state.obs, act_keys)[0]
    state = wrapped_infer_env.step(state, act)
    traj_data = empty_traj.tree_replace({
        "data.qpos": state.data.qpos,
        "data.qvel": state.data.qvel,
        "data.time": state.data.time,
        "data.ctrl": state.data.ctrl,
        "data.mocap_pos": state.data.mocap_pos,
        "data.mocap_quat": state.data.mocap_quat,
        "data.xfrc_applied": state.data.xfrc_applied,
    })
    return (state, rng), traj_data

  @jax.jit
  def do_rollout(state, rng):
    _, traj = jax.lax.scan(step, (state, rng), None, length=inference_steps)
    return traj

  traj_stacked = do_rollout(reset_states, jax.random.PRNGKey(_SEED.value + 1))
  # traj_stacked has shape (time, nworld, ...), swap to (nworld, time, ...).
  traj_stacked = jax.tree.map(lambda x: jp.moveaxis(x, 0, 1), traj_stacked)
  trajectories = [None] * _NUM_VIDEOS.value
  for i in range(_NUM_VIDEOS.value):
    t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
    trajectories[i] = [
        jax.tree.map(lambda x, j=j: x[j], t) for j in range(inference_steps)
    ]

  # Render and save the rollout.
  render_every = 2
  fps = 1.0 / infer_env.dt / render_every
  print(f"FPS for rendering: {fps}")
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
  video_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  for i, rollout in enumerate(trajectories):
    traj = rollout[::render_every]
    frames = infer_env.render(
        traj, height=480, width=640, scene_option=scene_option
    )
    if _TIMESTAMP_VIDEOS.value:
      stem = "rollout" if _NUM_VIDEOS.value == 1 else f"rollout{i}"
      stem += f"-{video_timestamp}"
    else:
      stem = f"rollout{i}"
    video_path = logdir / f"{stem}.mp4"
    media.write_video(video_path, frames, fps=fps)
    print(f"Rollout video saved as '{video_path}'.")

  # From this point onward all durable training artifacts have been written.
  # This independent watchdog still runs if the main thread blocks inside a
  # third-party cleanup routine, unlike a timeout checked by the main thread.
  cleanup_timeout = _WANDB_FINISH_TIMEOUT_SECONDS.value + 5.0
  _start_successful_shutdown_watchdog(cleanup_timeout)

  if writer is not None:
    writer.flush()
    writer.close()

  # Checkpoint and video writes have returned before reaching this point. Give
  # W&B a bounded interval to synchronize, then exit even if its service or a
  # JAX/XLA background thread is stuck during interpreter shutdown.
  if _USE_WANDB.value and not _PLAY_ONLY.value and wandb.run is not None:
    _finish_wandb_with_timeout(_WANDB_FINISH_TIMEOUT_SECONDS.value)

  print("Training artifacts finalized. Exiting.")
  sys.stdout.flush()
  sys.stderr.flush()
  os._exit(0)


def run():
  """Entry point for uv/pip script."""
  try:
    app.run(main)
  except SystemExit as exc:
    # Fallback for successful paths that return before main's explicit exit.
    # Never mask a nonzero absl exit caused by a training failure.
    if exc.code not in (None, 0, False):
      raise
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
  run()
