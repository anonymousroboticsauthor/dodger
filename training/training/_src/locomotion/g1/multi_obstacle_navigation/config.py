"""Single source of truth for G1 multi-obstacle navigation."""

from __future__ import annotations

from ml_collections import config_dict


def default_config() -> config_dict.ConfigDict:
  """Returns the environment configuration.

  The navigation actor and frozen locomotion policy run at 10 and 50 Hz.  The
  DPCBF QP is fixed at 50 Hz by the implementation.
  """
  return config_dict.create(
      sim_dt=0.002,  # 500 Hz deployment-compatible PD/simulation loop.
      ctrl_dt=0.1,   # 10 Hz navigation actor.
      policy_hz=10.0,
      low_level_hz=50.0,
      episode_length=300,
      action_repeat=1,
      impl="jax",
      evaluation_mode=False,
      algorithm="dodger",  # One of: dodger, cbf_rl, cbf_rl_min.
      naconmax=256,
      njmax=4096,
      obstacle=config_dict.create(
          count=20,
          max_count=20,  # Used for assigning memory in JIT-compiled functions.
          num_constraints=10,
          radius_range=[0.2, 0.3],
          speed_range=[0.0, 0.6],
          height=1.5,
          p_max=5.0,
          priority=1,  # 0: distance, 1: closing alignment then distance.
          initial_clearance=1.0,
          initial_h_margin=0.0,
          spawn_attempts=64,
      ),
      arena=config_dict.create(
          size=[10.0, 10.0],
          center=[0.0, 0.0],
          hard_boundary_padding=5.0,
          boundary_margin=0.15,
      ),
      robot=config_dict.create(
          radius=0.30,
          sagittal_velocity_range=[-1.0, 2.0],
          lateral_velocity_range=[-1.0, 1.0],
          yaw_rate_range=[-1.0, 1.0],
          sagittal_acceleration_range=[-4.0, 4.0],
          lateral_acceleration_range=[-2.0, 2.0],
          safety_factor=1.05,
          eps_v=0.05,
          eps_d=0.05,
          collision_grace_s=0.0,
          minimum_height=0.45,
      ),
      dpcbf=config_dict.create(
          k_mu=1.0,
          k_lambda=0.5,
          alpha=2.0,
          reward_sigma=0.5,
          reward_reduction="sum",
          violation_lower_bound=-0.40,
          filter_actions=True,
      ),
      dodger=config_dict.create(
          # Both soft constraints use the same training-step ramp.
          probability_ramp_end_steps=210_000_000,
          cmax_ema_tau=0.95,
          normalization_epsilon=1.0e-6,
          dpcbf_constraint=config_dict.create(
              probability_min=0.05,
              probability_max=0.25,
          ),
          # Keeps the high-level policy inside the command distribution used
          # to train the frozen low-level locomotion policy.  Violations are
          # measured before command clipping and normalized per command axis.
          low_level_command_constraint=config_dict.create(
              sagittal_range=[-1.0, 2.0],
              lateral_range=[-1.0, 1.0],
              yaw_rate_range=[-1.0, 1.0],
              probability_min=0.05,
              probability_max=0.25,
          ),
      ),
      qp=config_dict.create(
          slack_weight=1.0e6,
          enforce_input_bounds=True,
          max_iter=40,
          tolerance=1.0e-5,
          feasibility_tolerance=1.0e-3,
          failure_fallback="policy",
          regularization=1.0e-6,
          # Diagnostic only: minimum uniform relaxation required if DPCBF and
          # command rows were hard. It never changes the executed action.
          hard_feasibility=config_dict.create(
              enabled=True,
              tolerance=1.0e-3,
              solver_tolerance=1.0e-5,
              max_iter=40,
              relaxation_weight=1.0e6,
          ),
      ),
      goal=config_dict.create(
          radius=0.30,
          min_distance=2.0,
          max_distance=6.0,
          stationary_obstacle_speed_threshold=1.0e-2,
          safe_set_margin=0.0,
          sample_attempts=24,
          heading_progress_distance_scale=1.0,
          maximum_speed=0.2,
          maximum_yaw_rate=0.2,
      ),
      perception=config_dict.create(
          enabled=True,
          position_std=0.03,
          velocity_std=0.05,
          radius_std=0.015,
          per_episode_position_bias_std=0.02,
          dropout_probability=0.03,
          latency_steps=1,
      ),
      robot_state_randomization=config_dict.create(
          enabled=True,
          position_std=0.03,
          velocity_std=0.05,
          yaw_std=0.0174533,
          yaw_rate_std=0.02,
          per_episode_position_bias_std=0.05,
          per_episode_yaw_bias_std=0.0174533,
          position_random_walk_std=0.002,
          yaw_random_walk_std=0.000872665,
          latency_steps=1,
      ),
      curriculum=config_dict.create(
          enabled=True,
          # Dynamic curriculum promotion is allowed only after this many
          # environment transitions have been collected at the current stage.
          minimum_stage_steps_enabled=True,
          minimum_stage_steps=10_000_000,
          eval_stage=3,
          # Evaluation remains enabled, but high success only stops training
          # when this switch is explicitly enabled.
          early_termination=True,
          early_stop_epsilon=0.05,
          early_stop_required_checks=3,
          # Dynamic stage variables
          evaluation_episodes=512,
          promote_threshold=0.85,
          demote_threshold=-1.0,
          demote_required_checks=3,
          stages=[
              config_dict.create(
                  obstacle_fraction=0.40,
                  speed_fraction=0.0,
                  noise_fraction=0.0,
                  goal_heading_tolerance_deg=20.0,
              ),
              config_dict.create(
                  obstacle_fraction=0.60,
                  speed_fraction=0.35,
                  noise_fraction=0.25,
                  goal_heading_tolerance_deg=15.0,
              ),
              config_dict.create(
                  obstacle_fraction=0.80,
                  speed_fraction=0.65,
                  noise_fraction=0.60,
                  goal_heading_tolerance_deg=10.0,
              ),
              config_dict.create(
                  obstacle_fraction=1.0,
                  speed_fraction=1.0,
                  noise_fraction=1.0,
                  goal_heading_tolerance_deg=10.0,
              ),
          ],
      ),
      reward=config_dict.create(
          progress=60.0,
          heading_progress=10.0,
          goal=20.0,
          collision=-5.0,
          cbf=100.0,
          action_rate=-0.02,
          time=-0.01,
          timeout=-15.0,
          persistent_collision=-50.0,
          hard_outside=-100.0,
      ),
      low_level_policy=config_dict.create(
          weights_file="low_level_policy.npz",
          gait_period=0.6,
      ),
      play_mode=False,
  )


def validate_config(config: config_dict.ConfigDict) -> None:
  """Raises a useful error for shape/frequency configurations JIT cannot vary."""
  for name, hz in (
      ("policy_hz", config.policy_hz),
      ("low_level_hz", config.low_level_hz),
  ):
    ratio = 1.0 / (config.sim_dt * hz)
    if abs(ratio - round(ratio)) > 1.0e-6:
      raise ValueError(f"{name} must divide the simulation frequency")
  if abs(config.ctrl_dt * config.policy_hz - 1.0) > 1.0e-6:
    raise ValueError("ctrl_dt must equal 1 / policy_hz")
  qp_per_policy = 50.0 / config.policy_hz
  if abs(qp_per_policy - round(qp_per_policy)) > 1.0e-6:
    raise ValueError("the fixed 50 Hz QP rate must divide policy_hz")
  obs = config.obstacle
  if not 0 < obs.num_constraints <= obs.max_count:
    raise ValueError("num_constraints must be in [1, max_count]")
  if not 0 < obs.count <= obs.max_count:
    raise ValueError("obstacle count must be in [1, max_count]")
  if obs.priority not in (0, 1):
    raise ValueError("obstacle priority must be 0 or 1")
  if config.dpcbf.reward_reduction not in ("sum", "mean"):
    raise ValueError("DPCBF reward reduction must be 'sum' or 'mean'")
  if config.algorithm not in ("dodger", "cbf_rl", "cbf_rl_min"):
    raise ValueError("algorithm must be 'dodger', 'cbf_rl', or 'cbf_rl_min'")
  if config.dpcbf.violation_lower_bound > 0.0:
    raise ValueError("DPCBF violation_lower_bound must be non-positive")
  if config.dodger.probability_ramp_end_steps <= 0:
    raise ValueError("DODGER probability_ramp_end_steps must be positive")
  if not 0.0 <= config.dodger.cmax_ema_tau < 1.0:
    raise ValueError("DODGER cmax_ema_tau must be in [0, 1)")
  if config.dodger.normalization_epsilon <= 0.0:
    raise ValueError("DODGER normalization_epsilon must be positive")
  for name, constraint in (
      ("dpcbf_constraint", config.dodger.dpcbf_constraint),
      (
          "low_level_command_constraint",
          config.dodger.low_level_command_constraint,
      ),
  ):
    if not (
        0.0
        <= constraint.probability_min
        <= constraint.probability_max
        <= 1.0
    ):
      raise ValueError(
          f"DODGER {name} probabilities must satisfy "
          "0 <= probability_min <= probability_max <= 1"
      )
  command_constraint = config.dodger.low_level_command_constraint
  for name, limits in (
      ("sagittal_range", command_constraint.sagittal_range),
      ("lateral_range", command_constraint.lateral_range),
      ("yaw_rate_range", command_constraint.yaw_rate_range),
  ):
    if len(limits) != 2 or limits[0] >= limits[1]:
      raise ValueError(
          f"DODGER low_level_command_constraint.{name} must be [min, max]"
      )
  if config.qp.slack_weight <= 0.0:
    raise ValueError("QP slack_weight must be positive")
  if config.qp.max_iter <= 0:
    raise ValueError("QP max_iter must be positive")
  if config.qp.feasibility_tolerance <= 0.0:
    raise ValueError("QP feasibility_tolerance must be positive")
  hard_feasibility = config.qp.hard_feasibility
  if hard_feasibility.tolerance <= 0.0:
    raise ValueError("hard DPCBF feasibility tolerance must be positive")
  if hard_feasibility.solver_tolerance <= 0.0:
    raise ValueError("hard DPCBF solver tolerance must be positive")
  if hard_feasibility.max_iter <= 0:
    raise ValueError("hard DPCBF max_iter must be positive")
  if hard_feasibility.relaxation_weight <= 0.0:
    raise ValueError("hard DPCBF relaxation_weight must be positive")
  if config.qp.failure_fallback != "policy":
    raise ValueError("QP failure_fallback currently supports only 'policy'")
  if config.robot.collision_grace_s < 0.0:
    raise ValueError("collision_grace_s must be non-negative")
  if config.curriculum.demote_required_checks <= 0:
    raise ValueError("demote_required_checks must be positive")
  if config.curriculum.minimum_stage_steps < 0:
    raise ValueError("minimum_stage_steps must be non-negative")
  if not 0 <= config.curriculum.eval_stage < len(config.curriculum.stages):
    raise ValueError("curriculum eval_stage is outside the configured stages")
  if not 0.0 < config.curriculum.early_stop_epsilon < 1.0:
    raise ValueError("early_stop_epsilon must be in (0, 1)")
  if config.curriculum.early_stop_required_checks <= 0:
    raise ValueError("early_stop_required_checks must be positive")
  if config.goal.stationary_obstacle_speed_threshold < 0.0:
    raise ValueError("stationary obstacle speed threshold must be non-negative")
  if config.goal.sample_attempts <= 0:
    raise ValueError("goal sample_attempts must be positive")
  for name, latency in (
      ("perception.latency_steps", config.perception.latency_steps),
      (
          "robot_state_randomization.latency_steps",
          config.robot_state_randomization.latency_steps,
      ),
  ):
    if latency not in (0, 1):
      raise ValueError(f"{name} currently supports only 0 or 1")
