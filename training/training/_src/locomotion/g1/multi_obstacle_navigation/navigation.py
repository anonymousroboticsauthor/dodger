"""Hierarchical Unitree G1 multi-obstacle navigation environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math

from training._src import mjx_env
from training._src.locomotion.g1 import base as g1_base
from training._src.locomotion.g1.multi_obstacle_navigation import (
  config as nav_config,
)
from training._src.locomotion.g1.multi_obstacle_navigation import (
  dpcbf,
  low_level_policy,
)


def default_config() -> config_dict.ConfigDict:
  return nav_config.default_config()


def _scene_xml(config: config_dict.ConfigDict) -> str:
  """Builds the fixed-topology scene while retaining configurable dimensions."""
  center_x, center_y = config.arena.center
  size_x, size_y = config.arena.size
  outer_x = size_x * 0.5 + config.arena.hard_boundary_padding
  outer_y = size_y * 0.5 + config.arena.hard_boundary_padding
  obstacle_bodies = []
  for index in range(config.obstacle.max_count):
    obstacle_bodies.append(
        f'<body name="obstacle_{index:02d}" mocap="true" pos="0 0 -10">'
        f'<geom name="obstacle_geom_{index:02d}" type="cylinder" '
        f'size="0.25 {config.obstacle.height * 0.5}" '
        'rgba="0.1 0.4 1.0 0.65" contype="0" conaffinity="0"/>'
        "</body>"
    )
  return f"""
<mujoco model="g1 multi obstacle navigation">
  <include file="g1_mjx_feetonly.xml"/>
  <statistic center="0 0 0.7" extent="8" meansize="0.04"/>
  <visual>
    <headlight diffuse=".8 .8 .8" ambient=".2 .2 .2" specular="1 1 1"/>
    <global azimuth="140" elevation="-30"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="1 1 1" rgb2=".85 .85 .85" markrgb="0 0 0" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="10 10" reflectance="0"/>
  </asset>
  <worldbody>
    <geom name="floor" size="0 0 0.01" type="plane" material="groundplane"/>
    <body name="goal" mocap="true" pos="0 0 0.03">
      <geom name="goal_disk" type="cylinder" size="{config.goal.radius} .025" rgba=".15 .95 .25 .65" contype="0" conaffinity="0"/>
      <geom name="goal_heading" type="box" pos="{config.goal.radius * 0.35} 0 .055" size="{config.goal.radius * 0.55} .035 .025" rgba="1 .75 .1 .9" contype="0" conaffinity="0"/>
    </body>
    {''.join(obstacle_bodies)}
    <body name="arena_markers" pos="{center_x} {center_y} .02">
      <geom type="box" pos="0 {size_y * .5} 0" size="{size_x * .5} .025 .02" rgba=".2 .7 1 .35" contype="0" conaffinity="0"/>
      <geom type="box" pos="0 {-size_y * .5} 0" size="{size_x * .5} .025 .02" rgba=".2 .7 1 .35" contype="0" conaffinity="0"/>
      <geom type="box" pos="{size_x * .5} 0 0" size=".025 {size_y * .5} .02" rgba=".2 .7 1 .35" contype="0" conaffinity="0"/>
      <geom type="box" pos="{-size_x * .5} 0 0" size=".025 {size_y * .5} .02" rgba=".2 .7 1 .35" contype="0" conaffinity="0"/>
      <geom type="box" pos="0 {outer_y} .01" size="{outer_x} .04 .03" rgba="1 .15 .1 .4" contype="0" conaffinity="0"/>
      <geom type="box" pos="0 {-outer_y} .01" size="{outer_x} .04 .03" rgba="1 .15 .1 .4" contype="0" conaffinity="0"/>
      <geom type="box" pos="{outer_x} 0 .01" size=".04 {outer_y} .03" rgba="1 .15 .1 .4" contype="0" conaffinity="0"/>
      <geom type="box" pos="{-outer_x} 0 .01" size=".04 {outer_y} .03" rgba="1 .15 .1 .4" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <include file="sensor.xml"/>
</mujoco>
"""


def _yaw_from_quaternion(quaternion: jax.Array) -> jax.Array:
  w, x, y, z = quaternion
  return jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_angle(angle: jax.Array) -> jax.Array:
  return jp.arctan2(jp.sin(angle), jp.cos(angle))


class MultiObstacleNavigation(g1_base.G1Env):
  """10 Hz GAT navigation over a frozen 50 Hz G1 locomotion policy."""

  uses_fixed_window_curriculum = True
  requires_full_reset = True

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    # Build the configurable scene in-memory rather than fixing arena geometry
    # in an XML that disagrees with config overrides.
    mjx_env.MjxEnv.__init__(self, config, config_overrides)
    nav_config.validate_config(self._config)
    self.curriculum_config = (
        self._config.curriculum
        if self._config.curriculum.enabled
        and not self._config.play_mode
        and not self._config.evaluation_mode
        else None
    )
    self.dodger_config = (
        self._config.dodger if self._config.algorithm == "dodger" else None
    )
    self.dodger_apply_soft_discount = (
        self.dodger_config is not None
        and not self._config.evaluation_mode
        and not self._config.play_mode
    )
    self.dodger_split_value_critic = self.dodger_apply_soft_discount
    self._model_assets = g1_base.get_assets()
    scene = _scene_xml(self._config)
    self._mj_model = mujoco.MjModel.from_xml_string(
        scene, assets=self._model_assets
    )
    self._mj_model.opt.timestep = self.sim_dt
    low_level_policy.apply_deploy_pd_to_model(self._mj_model)
    self._mj_model.vis.global_.offwidth = 1920
    self._mj_model.vis.global_.offheight = 1080
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = "g1_multi_obstacle_navigation.xml"
    self._post_init()

  def _post_init(self) -> None:
    config = self._config
    self._default_pose = low_level_policy.DEFAULT_JOINT_POSITION
    self._action_scale = low_level_policy.ACTION_SCALE
    package_dir = Path(__file__).resolve().parent
    weights_path = package_dir / "assets" / config.low_level_policy.weights_file
    self._low_policy = low_level_policy.FrozenVelocityPolicy(weights_path)
    self._pelvis_imu_site_id = self._mj_model.site("imu_in_pelvis").id
    self._goal_mocap_id = self._mj_model.body("goal").mocapid[0]
    self._obstacle_mocap_ids = jp.asarray([
        self._mj_model.body(f"obstacle_{index:02d}").mocapid[0]
        for index in range(config.obstacle.max_count)
    ])
    self._sim_steps = round(config.ctrl_dt / config.sim_dt)
    self._qp_decimation = round(1.0 / (config.sim_dt * dpcbf.QP_HZ))
    self._low_decimation = round(1.0 / (config.sim_dt * config.low_level_hz))
    self._qp_updates_per_policy = round(dpcbf.QP_HZ / config.policy_hz)

  @property
  def action_size(self) -> int:
    return 3

  def _difficulty(self, stage: jax.Array) -> tuple[jax.Array, ...]:
    values = self._config.curriculum.stages
    obstacle_fraction = jp.asarray([x.obstacle_fraction for x in values])[stage]
    speed_fraction = jp.asarray([x.speed_fraction for x in values])[stage]
    noise_fraction = jp.asarray([x.noise_fraction for x in values])[stage]
    heading_tolerance = jp.asarray(
        [x.goal_heading_tolerance_deg for x in values]
    )[stage]
    return obstacle_fraction, speed_fraction, noise_fraction, heading_tolerance

  def reset(self, rng: jax.Array) -> mjx_env.State:
    if self._config.play_mode:
      stage = len(self._config.curriculum.stages) - 1
    elif self._config.evaluation_mode:
      stage = self._config.curriculum.eval_stage
    else:
      stage = 0
    return self.reset_at_stage(rng, jp.asarray(stage, dtype=jp.int32))

  def reset_at_stage(
      self, rng: jax.Array, curriculum_stage: jax.Array
  ) -> mjx_env.State:
    config = self._config
    qpos = jp.zeros((self.mjx_model.nq,))
    qpos = qpos.at[2].set(0.785)
    qpos = qpos.at[3].set(1.0)
    qpos = qpos.at[7:].set(self._default_pose)
    rng, yaw_rng = jax.random.split(rng)
    yaw = jax.random.uniform(yaw_rng, minval=-jp.pi, maxval=jp.pi)
    yaw_quat = math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)
    qpos = qpos.at[3:7].set(yaw_quat)
    qvel = jp.zeros((self.mjx_model.nv,))

    obstacle_fraction, speed_fraction, _, _ = self._difficulty(curriculum_stage)
    active_count = jp.maximum(
        1,
        jp.rint(config.obstacle.count * obstacle_fraction).astype(jp.int32),
    )
    active = jp.arange(config.obstacle.max_count) < active_count
    rng, radius_rng, speed_rng, angle_rng, position_rng = jax.random.split(
        rng, 5
    )
    radius = jax.random.uniform(
        radius_rng,
        (config.obstacle.max_count,),
        minval=config.obstacle.radius_range[0],
        maxval=config.obstacle.radius_range[1],
    )
    speed_max = (
        config.obstacle.speed_range[0]
        + (config.obstacle.speed_range[1] - config.obstacle.speed_range[0])
        * speed_fraction
    )
    speed = jax.random.uniform(
        speed_rng,
        (config.obstacle.max_count,),
        minval=config.obstacle.speed_range[0],
        maxval=speed_max,
    )
    angle = jax.random.uniform(
        angle_rng, (config.obstacle.max_count,), minval=-jp.pi, maxval=jp.pi
    )
    obstacle_velocity = (
        jp.stack((speed * jp.cos(angle), speed * jp.sin(angle)), axis=-1)
        * active[:, None]
    )
    half = jp.asarray(config.arena.size) * 0.5
    center = jp.asarray(config.arena.center)
    lower = center - half + radius[:, None]
    upper = center + half - radius[:, None]
    obstacle_position = jax.random.uniform(
        position_rng,
        (config.obstacle.max_count, 2),
        minval=lower,
        maxval=upper,
    )

    def reject_spawn(iteration, carry):
      positions, key = carry
      surface_clearance = (
          jp.linalg.norm(positions, axis=-1) - config.robot.radius - radius
      )
      selected_all = dpcbf.SelectedObstacles(
          relative_position=positions,
          relative_velocity=obstacle_velocity,
          radius=radius,
          valid=active,
          indices=jp.arange(config.obstacle.max_count),
      )
      zero_action = jp.zeros((3,))
      zero_velocity = jp.zeros((2,))
      barrier = dpcbf.evaluate_dpcbf(
          selected_all, yaw, zero_velocity, zero_action, config
      ).h
      invalid = active & (
          (surface_clearance < config.obstacle.initial_clearance)
          | (barrier < config.obstacle.initial_h_margin)
      )
      key, replacement_rng = jax.random.split(key)
      replacement = jax.random.uniform(
          jax.random.fold_in(replacement_rng, iteration),
          positions.shape,
          minval=lower,
          maxval=upper,
      )
      return jp.where(invalid[:, None], replacement, positions), key

    obstacle_position, rng = jax.lax.fori_loop(
        0,
        config.obstacle.spawn_attempts,
        reject_spawn,
        (obstacle_position, rng),
    )
    rng, goal_rng, heading_rng = jax.random.split(rng, 3)
    goal, rng = self._sample_goal(
        goal_rng,
        rng,
        jp.zeros((2,)),
        obstacle_position,
        obstacle_velocity,
        radius,
        active,
    )
    goal_heading = jax.random.uniform(heading_rng, minval=-jp.pi, maxval=jp.pi)

    mocap_pos = jp.zeros((self.mjx_model.nmocap, 3))
    mocap_quat = jp.zeros((self.mjx_model.nmocap, 4)).at[:, 0].set(1.0)
    mocap_pos = mocap_pos.at[self._goal_mocap_id].set(
        jp.asarray((goal[0], goal[1], 0.03))
    )
    mocap_quat = mocap_quat.at[self._goal_mocap_id].set(
        math.axis_angle_to_quat(jp.asarray((0.0, 0.0, 1.0)), goal_heading)
    )
    obstacle_xyz = jp.concatenate(
        (
            obstacle_position,
            jp.where(active, config.obstacle.height * 0.5, -10.0)[:, None],
        ),
        axis=-1,
    )
    mocap_pos = mocap_pos.at[self._obstacle_mocap_ids].set(obstacle_xyz)
    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=self._default_pose,
        mocap_pos=mocap_pos,
        mocap_quat=mocap_quat,
        impl=self.mjx_model.impl.value,
        naconmax=config.naconmax,
        njmax=config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)
    (
        rng,
        perception_bias_rng,
        robot_position_bias_rng,
        robot_yaw_bias_rng,
    ) = jax.random.split(rng, 4)
    goal_distance = jp.linalg.norm(goal)
    info = {
        "rng": rng,
        "step": jp.asarray(0, dtype=jp.int32),
        "curriculum_stage": curriculum_stage,
        "obstacle_position": obstacle_position,
        "obstacle_velocity": obstacle_velocity,
        "obstacle_radius": radius,
        "obstacle_active": active,
        "goal": goal,
        "goal_heading": goal_heading,
        "previous_goal_distance": goal_distance,
        "previous_heading_error": jp.abs(_wrap_angle(goal_heading - yaw)),
        "velocity_command": jp.zeros((3,)),
        "policy_action": jp.zeros((3,)),
        "safe_action": jp.zeros((3,)),
        "previous_raw_action": jp.zeros((3,)),
        "last_low_action": jp.zeros((29,)),
        "motor_targets": self._default_pose,
        "phase_time": jp.asarray(0.0),
        "perception_position_bias": (
            jax.random.normal(perception_bias_rng, obstacle_position.shape)
            * config.perception.per_episode_position_bias_std
        ),
        "robot_position_bias": (
            jax.random.normal(robot_position_bias_rng, (2,))
            * config.robot_state_randomization.per_episode_position_bias_std
        ),
        "robot_yaw_bias": (
            jax.random.normal(robot_yaw_bias_rng)
            * config.robot_state_randomization.per_episode_yaw_bias_std
        ),
        "robot_position_walk": jp.zeros((2,)),
        "robot_yaw_walk": jp.asarray(0.0),
        "previous_actor_position": jp.zeros((2,)),
        "previous_actor_yaw": yaw,
        "previous_actor_velocity_body": jp.zeros((2,)),
        "previous_actor_yaw_rate": jp.asarray(0.0),
        "previous_perceived_obstacle_position": obstacle_position,
        "previous_perceived_obstacle_velocity": obstacle_velocity,
        "previous_perceived_obstacle_radius": radius,
        "previous_perceived_obstacle_active": active,
        "collision_time": jp.asarray(0.0),
        "goal_reached": jp.asarray(False),
        "qp_slack": jp.zeros((config.obstacle.num_constraints,)),
        "cbf_condition": jp.zeros((config.obstacle.num_constraints,)),
        "qp_residual": jp.asarray(0.0),
        "qp_iterations": jp.asarray(0, dtype=jp.int32),
        "qp_failed": jp.asarray(False),
        "hard_cbf_minimum_relaxation": jp.asarray(0.0),
        "hard_cbf_feasible": jp.asarray(True),
        "hard_cbf_numerical_failure": jp.asarray(False),
        "dodger_constraint_violation": jp.asarray(0.0),
        "dodger_low_level_command_violations": jp.zeros((3,)),
        # DPCBF followed by sagittal, lateral, and yaw command constraints.
        "dodger_constraint_violations": jp.zeros((4,)),
    }
    metrics = self._zero_metrics()
    obs, info = self._get_observation(data, info)
    return mjx_env.State(
        data, obs, jp.asarray(0.0), jp.asarray(0.0), metrics, info
    )

  def _sample_goal(  # noqa: PLR0917
      self,
      sample_rng,
      next_rng,
      robot_position,
      obstacle_position,
      obstacle_velocity,
      obstacle_radius,
      obstacle_active,
  ):
    config = self._config
    half = jp.asarray(config.arena.size) * 0.5
    center = jp.asarray(config.arena.center)
    margin = config.goal.radius + config.arena.boundary_margin
    angle_rng, distance_rng = jax.random.split(sample_rng)
    angle = jax.random.uniform(angle_rng, minval=-jp.pi, maxval=jp.pi)
    distance = jax.random.uniform(
        distance_rng,
        minval=config.goal.min_distance,
        maxval=config.goal.max_distance,
    )
    goal = robot_position + distance * jp.asarray(
        (jp.cos(angle), jp.sin(angle))
    )
    goal = jp.clip(goal, center - half + margin, center + half - margin)

    stationary = obstacle_active & (
        jp.linalg.norm(obstacle_velocity, axis=-1)
        <= config.goal.stationary_obstacle_speed_threshold
    )
    # For a stationary robot and obstacle, DPCBF h >= margin reduces exactly
    # to a minimum center distance.  Use that equivalent form here instead of
    # repeatedly evaluating all DPCBF derivatives during every reset.
    safe_radius = (
        config.robot.radius + obstacle_radius
    ) * config.robot.safety_factor
    adaptive = jp.sqrt(config.robot.safety_factor**2 - 1.0) / safe_radius
    required_safe_distance = config.goal.safe_set_margin / jp.maximum(
        adaptive * config.dpcbf.k_mu, 1.0e-8
    )
    stationary_safe_distance = jp.sqrt(
        safe_radius**2
        + 2.0 * config.robot.eps_d * required_safe_distance
        + required_safe_distance**2
    )

    def is_invalid(candidate):
      clearance = jp.linalg.norm(obstacle_position - candidate, axis=-1)
      overlap_distance = obstacle_radius + config.goal.radius + 0.2
      required_distance = jp.where(
          stationary,
          jp.maximum(overlap_distance, stationary_safe_distance),
          overlap_distance,
      )
      overlap = jp.any(obstacle_active & (clearance < required_distance))
      candidate_distance = jp.linalg.norm(candidate - robot_position)
      return overlap | (candidate_distance < config.goal.min_distance)

    def improve_goal(iteration, carry):
      current, key, found = carry
      key, a_rng, d_rng = jax.random.split(key, 3)
      candidate_angle = jax.random.uniform(
          jax.random.fold_in(a_rng, iteration), minval=-jp.pi, maxval=jp.pi
      )
      candidate_distance = jax.random.uniform(
          jax.random.fold_in(d_rng, iteration),
          minval=config.goal.min_distance,
          maxval=config.goal.max_distance,
      )
      candidate = robot_position + candidate_distance * jp.asarray(
          (jp.cos(candidate_angle), jp.sin(candidate_angle))
      )
      candidate = jp.clip(
          candidate, center - half + margin, center + half - margin
      )
      candidate_valid = ~is_invalid(candidate)
      accept_candidate = ~found & candidate_valid
      return (
          jp.where(accept_candidate, candidate, current),
          key,
          found | candidate_valid,
      )

    goal, next_rng, _ = jax.lax.fori_loop(
        0,
        config.goal.sample_attempts,
        improve_goal,
        (goal, next_rng, ~is_invalid(goal)),
    )
    return goal, next_rng

  def _robot_state(self, data: mjx.Data):
    position = data.qpos[:2]
    yaw = _yaw_from_quaternion(data.qpos[3:7])
    velocity_body = self.get_local_linvel(data, "pelvis")[:2]
    velocity_world = self.get_global_linvel(data, "pelvis")[:2]
    yaw_rate = self.get_gyro(data, "pelvis")[2]
    return position, yaw, velocity_body, velocity_world, yaw_rate

  def _physical_policy_action(self, raw_action, info):
    config = self._config
    raw = jp.clip(raw_action, -1.0, 1.0)
    ranges = jp.asarray((
        config.robot.sagittal_acceleration_range,
        config.robot.lateral_acceleration_range,
        config.robot.yaw_rate_range,
    ))
    physical = (
        0.5 * (ranges[:, 0] + ranges[:, 1])
        + 0.5 * (ranges[:, 1] - ranges[:, 0]) * raw
    )
    return physical, raw, info

  def _low_level_observation(self, data, info):
    gyro = self.get_gyro(data, "pelvis")
    gravity = data.site_xmat[self._pelvis_imu_site_id].T @ jp.asarray(
        (0.0, 0.0, -1.0)
    )
    phase = (
        2.0
        * jp.pi
        * info["phase_time"]
        / self._config.low_level_policy.gait_period
    )
    phase = jp.asarray((jp.sin(phase), jp.cos(phase)))
    phase = jp.where(jp.linalg.norm(info["velocity_command"]) < 0.1, 0.0, phase)
    return jp.concatenate((
        gyro,
        gravity,
        info["velocity_command"],
        phase,
        data.qpos[7:] - self._default_pose,
        data.qvel[6:],
        info["last_low_action"],
    ))

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    config = self._config
    policy_action, raw_action, info = self._physical_policy_action(
        action, state.info
    )
    info["policy_action"] = policy_action
    # Report whether any QP update failed during this 10 Hz policy step.
    info["qp_failed"] = jp.asarray(False)
    initial = (
        state.data,
        info,
        jp.asarray(False),
        jp.asarray(0.0),
        jp.asarray(0.0),
        jp.asarray(0.0),
        jp.asarray(jp.inf),
        jp.zeros((3,)),
    )

    def physics_step(carry, sim_index):
      (
          data,
          inner_info,
          collision_latched,
          cbf_sum,
          intervention_sum,
          slack_sum,
          minimum_condition,
          maximum_command_violation,
      ) = carry
      position, yaw, velocity_body, velocity_world, _ = self._robot_state(data)
      is_qp_step = jp.mod(sim_index, self._qp_decimation) == 0

      def run_qp(qp_carry):
        (
            inner_info,
            cbf_sum,
            intervention_sum,
            slack_sum,
            minimum_condition,
            maximum_command_violation,
        ) = qp_carry
        selected = dpcbf.select_obstacles(
            inner_info["obstacle_position"],
            inner_info["obstacle_velocity"],
            inner_info["obstacle_radius"],
            inner_info["obstacle_active"],
            position,
            velocity_world,
            p_max=config.obstacle.p_max,
            num_constraints=config.obstacle.num_constraints,
            priority=config.obstacle.priority,
        )
        filtered = dpcbf.filter_action(
            selected,
            yaw,
            velocity_body,
            inner_info["velocity_command"],
            policy_action,
            config,
        )
        if config.qp.hard_feasibility.enabled:
          hard_diagnostic = jax.lax.cond(
              sim_index == 0,
              lambda _: dpcbf.diagnose_hard_feasibility(
                  filtered.values,
                  inner_info["velocity_command"],
                  config,
              ),
              lambda _: dpcbf.HardFeasibilityResult(
                  minimum_relaxation=inner_info[
                      "hard_cbf_minimum_relaxation"
                  ],
                  feasible=inner_info["hard_cbf_feasible"],
                  numerical_failure=inner_info[
                      "hard_cbf_numerical_failure"
                  ],
              ),
              operand=None,
          )
          inner_info["hard_cbf_minimum_relaxation"] = jp.where(
              hard_diagnostic.numerical_failure,
              0.0,
              hard_diagnostic.minimum_relaxation,
          )
          inner_info["hard_cbf_feasible"] = hard_diagnostic.feasible
          inner_info["hard_cbf_numerical_failure"] = (
              hard_diagnostic.numerical_failure
          )
        safe = filtered.safe_action
        executed_action = jp.where(
            config.dpcbf.filter_actions & (config.algorithm != "dodger"),
            safe,
            policy_action,
        )
        command = inner_info["velocity_command"]
        command = command.at[:2].add(executed_action[:2] / dpcbf.QP_HZ)
        command = command.at[2].set(executed_action[2])
        if config.algorithm == "dodger":
          command_config = config.dodger.low_level_command_constraint
          lower = jp.asarray((
              command_config.sagittal_range[0],
              command_config.lateral_range[0],
              command_config.yaw_rate_range[0],
          ))
          upper = jp.asarray((
              command_config.sagittal_range[1],
              command_config.lateral_range[1],
              command_config.yaw_rate_range[1],
          ))
          command_violation = jp.maximum(
              jp.maximum(lower - command, command - upper), 0.0
          )
          maximum_command_violation = jp.maximum(
              maximum_command_violation, command_violation
          )
        if config.qp.enforce_input_bounds:
          command = command.at[0].set(
              jp.clip(command[0], *config.robot.sagittal_velocity_range)
          )
          command = command.at[1].set(
              jp.clip(command[1], *config.robot.lateral_velocity_range)
          )
          command = command.at[2].set(
              jp.clip(command[2], *config.robot.yaw_rate_range)
          )
        violation = jp.minimum(filtered.values.condition, 0.0)
        violation = jp.where(filtered.values.valid, violation, 0.0)
        if config.algorithm == "cbf_rl_min":
          worst_violation = jp.min(
              jp.where(filtered.values.valid, violation, jp.inf)
          )
          worst_violation = jp.where(
              jp.any(filtered.values.valid), worst_violation, 0.0
          )
          violation_reward = jp.maximum(
              worst_violation, config.dpcbf.violation_lower_bound
          )
        elif config.dpcbf.reward_reduction == "mean":
          violation_reward = jp.sum(violation) / jp.maximum(
              jp.sum(filtered.values.valid), 1
          )
        else:
          violation_reward = jp.sum(violation)
        intervention = (
            jp.exp(
                -jp.sum((policy_action - safe) ** 2)
                / config.dpcbf.reward_sigma**2
            )
            - 1.0
        )
        inner_info["velocity_command"] = command
        inner_info["safe_action"] = safe
        inner_info["qp_slack"] = filtered.slack
        inner_info["cbf_condition"] = filtered.values.condition
        inner_info["qp_residual"] = filtered.qp_residual
        inner_info["qp_iterations"] = filtered.qp_iterations
        inner_info["qp_failed"] = inner_info["qp_failed"] | filtered.qp_failed
        current_condition = jp.min(
            jp.where(filtered.values.valid, filtered.values.condition, jp.inf)
        )
        return (
            inner_info,
            cbf_sum + violation_reward,
            intervention_sum + intervention,
            slack_sum + jp.sum(filtered.slack),
            jp.minimum(minimum_condition, current_condition),
            maximum_command_violation,
        )

      (
          inner_info,
          cbf_sum,
          intervention_sum,
          slack_sum,
          minimum_condition,
          maximum_command_violation,
      ) = jax.lax.cond(
          is_qp_step,
          run_qp,
          lambda x: x,
          (
              inner_info,
              cbf_sum,
              intervention_sum,
              slack_sum,
              minimum_condition,
              maximum_command_violation,
          ),
      )
      is_low_step = jp.mod(sim_index, self._low_decimation) == 0

      def run_low_level(inner_info):
        low_observation = self._low_level_observation(data, inner_info)
        low_action = self._low_policy(low_observation)
        inner_info["last_low_action"] = low_action
        inner_info["motor_targets"] = (
            self._default_pose + self._action_scale * low_action
        )
        inner_info["phase_time"] += 1.0 / config.low_level_hz
        return inner_info

      inner_info = jax.lax.cond(
          is_low_step, run_low_level, lambda x: x, inner_info
      )
      obstacle_position = (
          inner_info["obstacle_position"]
          + inner_info["obstacle_velocity"] * config.sim_dt
      )
      radius = inner_info["obstacle_radius"]
      half = jp.asarray(config.arena.size) * 0.5
      center = jp.asarray(config.arena.center)
      lower = center - half + radius[:, None]
      upper = center + half - radius[:, None]
      hit = (obstacle_position < lower) | (obstacle_position > upper)
      obstacle_position = jp.clip(obstacle_position, lower, upper)
      obstacle_velocity = jp.where(
          hit, -inner_info["obstacle_velocity"], inner_info["obstacle_velocity"]
      )
      obstacle_velocity *= inner_info["obstacle_active"][:, None]
      inner_info["obstacle_position"] = obstacle_position
      inner_info["obstacle_velocity"] = obstacle_velocity
      obstacle_xyz = jp.concatenate(
          (
              obstacle_position,
              jp.where(
                  inner_info["obstacle_active"],
                  config.obstacle.height * 0.5,
                  -10.0,
              )[:, None],
          ),
          axis=-1,
      )
      mocap_pos = data.mocap_pos.at[self._obstacle_mocap_ids].set(obstacle_xyz)
      data = data.replace(mocap_pos=mocap_pos)
      distance = jp.linalg.norm(obstacle_position - position, axis=-1)
      contact = jp.any(
          inner_info["obstacle_active"]
          & (distance <= radius + config.robot.radius)
      )
      collision_latched |= contact
      inner_info["collision_time"] = jp.where(
          contact,
          inner_info["collision_time"] + config.sim_dt,
          0.0,
      )
      data = mjx_env.step(self.mjx_model, data, inner_info["motor_targets"], 1)
      return (
          data,
          inner_info,
          collision_latched,
          cbf_sum,
          intervention_sum,
          slack_sum,
          minimum_condition,
          maximum_command_violation,
      ), None

    ###################################################
    # Reward Term Computation and Episode Termination #
    ###################################################
    (
        (
            data,
            info,
            collision_latched,
            cbf_sum,
            intervention_sum,
            slack_sum,
            minimum_condition,
            maximum_command_violation,
        ),
        _,
    ) = jax.lax.scan(physics_step, initial, jp.arange(self._sim_steps))
    position, yaw, velocity_body, _, yaw_rate = self._robot_state(data)
    goal_distance = jp.linalg.norm(info["goal"] - position)
    heading_error = jp.abs(_wrap_angle(info["goal_heading"] - yaw))
    _, _, _, tolerance_deg = self._difficulty(info["curriculum_stage"])

    # Goal termination
    goal_reached = (
        (goal_distance <= config.robot.radius + config.goal.radius)
        & (heading_error <= jp.deg2rad(tolerance_deg))
        & (jp.linalg.norm(velocity_body) <= config.goal.maximum_speed)
        & (jp.abs(yaw_rate) <= config.goal.maximum_yaw_rate)
    )

    # Collision termination
    grace = config.robot.collision_grace_s
    collision_done = jp.where(
        grace == 0.0,
        collision_latched,
        info["collision_time"] >= grace,
    )

    # Outside termination
    outer_half = (
        jp.asarray(config.arena.size) * 0.5 + config.arena.hard_boundary_padding
    )
    outside = jp.any(
        jp.abs(position - jp.asarray(config.arena.center)) + config.robot.radius
        >= outer_half
    )

    # Fallen termination
    fallen = data.qpos[2] < config.robot.minimum_height

    # Timeout termination
    timed_out = (info["step"] + 1 >= config.episode_length) & ~config.play_mode

    # Termination is true if any of the above conditions are met, or if the goal is reached and we are not in play mode.
    done = collision_done | outside | fallen | timed_out
    done |= goal_reached & ~config.play_mode

    # Progress and Heading Progress Terms
    # progress: The change in distance to the goal since the last step. Positive if moving towards the goal, negative if moving away.
    # heading_progress: The change in heading error since the last step. Positive if the robot is aligning better with the goal heading, negative if it's diverging.
    progress = info["previous_goal_distance"] - goal_distance
    heading_progress = info["previous_heading_error"] - heading_error
    heading_gate = jp.exp(
        -((goal_distance / config.goal.heading_progress_distance_scale) ** 2)
    )

    # CBF Reward Term:
    # cbf_sum: sum of CBF constraint violations (min(h_dot+ alpha*h, 0)).
    # intervention_sum: sum of intervention penalties (exp(-||u_safe - u_policy||^2 / sigma^2) - 1).
    if config.algorithm == "dodger":
      cbf_reward = intervention_sum / self._qp_updates_per_policy
    else:
      cbf_reward = (cbf_sum + intervention_sum) / self._qp_updates_per_policy

    minimum_condition = jp.where(
        jp.isfinite(minimum_condition), minimum_condition, 0.0
    )
    # DODGER always constrains the complete DPCBF condition evaluated at the
    # policy action: -(Lf h + Lg h u_policy + alpha h).
    dpcbf_violation = jp.maximum(-minimum_condition, 0.0)
    info["dodger_constraint_violation"] = dpcbf_violation
    info["dodger_low_level_command_violations"] = maximum_command_violation
    dodger_constraint_violations = jp.concatenate(
        (dpcbf_violation[None], maximum_command_violation)
    )
    info["dodger_constraint_violations"] = dodger_constraint_violations

    # Keep all metric keys shape-stable while selecting the reward formulation.
    reward_terms = {name: jp.asarray(0.0) for name in config.reward.keys()}
    reward_terms.update({
        "progress": progress,
        "heading_progress": heading_progress * heading_gate,
        "goal": goal_reached.astype(jp.float32),
        "collision": collision_latched.astype(jp.float32),
        "cbf": cbf_reward,
        "action_rate": jp.sum(
            (raw_action - info["previous_raw_action"]) ** 2
        ),
        "time": jp.asarray(1.0),
        "timeout": timed_out.astype(jp.float32),
        "persistent_collision": collision_done.astype(jp.float32),
        "hard_outside": outside.astype(jp.float32),
    })
    reward = sum(
        reward_terms[name] * config.reward[name] for name in reward_terms
    )
    info["step"] += 1
    info["previous_goal_distance"] = goal_distance
    info["previous_heading_error"] = heading_error
    info["previous_raw_action"] = raw_action
    info["goal_reached"] = goal_reached

    # Play chains random goals while preserving robot/command state.
    def resample_play_goal(operand):
      info, current_data = operand
      info["rng"], sample_rng, next_rng, heading_rng = jax.random.split(
          info["rng"], 4
      )
      goal, _ = self._sample_goal(
          sample_rng,
          next_rng,
          position,
          info["obstacle_position"],
          info["obstacle_velocity"],
          info["obstacle_radius"],
          info["obstacle_active"],
      )
      heading = jax.random.uniform(heading_rng, minval=-jp.pi, maxval=jp.pi)
      info["goal"] = goal
      info["goal_heading"] = heading
      info["previous_goal_distance"] = jp.linalg.norm(goal - position)
      info["previous_heading_error"] = jp.abs(_wrap_angle(heading - yaw))
      goal_pos = jp.asarray((goal[0], goal[1], 0.03))
      goal_quat = math.axis_angle_to_quat(jp.asarray((0.0, 0.0, 1.0)), heading)
      return info, current_data.replace(
          mocap_pos=current_data.mocap_pos.at[self._goal_mocap_id].set(
              goal_pos
          ),
          mocap_quat=current_data.mocap_quat.at[self._goal_mocap_id].set(
              goal_quat
          ),
      )

    if config.play_mode:
      info, data = jax.lax.cond(
          goal_reached,
          resample_play_goal,
          lambda x: x,
          (info, data),
      )
    obs, info = self._get_observation(data, info)
    metrics = state.metrics
    for name, value in reward_terms.items():
      metrics[f"reward/{name}"] = value * config.reward[name]
    metrics["navigation/goal_distance"] = goal_distance
    metrics["navigation/collision"] = collision_latched.astype(jp.float32)
    metrics["navigation/outside"] = outside.astype(jp.float32)
    metrics["navigation/qp_slack"] = slack_sum / self._qp_updates_per_policy
    metrics["navigation/qp_failed"] = info["qp_failed"].astype(jp.float32)
    metrics["navigation/hard_cbf_feasible_per_step"] = info[
        "hard_cbf_feasible"
    ].astype(jp.float32)
    metrics["navigation/hard_cbf_minimum_relaxation_per_step"] = info[
        "hard_cbf_minimum_relaxation"
    ]
    metrics["navigation/hard_cbf_numerical_failure_per_step"] = info[
        "hard_cbf_numerical_failure"
    ].astype(jp.float32)
    metrics["navigation/dodger_min_dhdt_alphah_per_step"] = minimum_condition
    metrics["navigation/dodger_constraint_violation_per_step"] = info[
        "dodger_constraint_violation"
    ]
    metrics["navigation/curriculum_stage_per_step"] = info[
        "curriculum_stage"
    ].astype(jp.float32)
    # These are mutually exclusive only in the usual case (for example, a
    # robot can reach a goal exactly as its episode times out).  Keeping each
    # terminal condition separate makes their episode sums directly useful as
    # W&B termination rates.
    metrics["navigation/termination_goal_reached"] = goal_reached.astype(
        jp.float32
    )
    metrics["navigation/termination_collision"] = collision_done.astype(
        jp.float32
    )
    metrics["navigation/termination_outside"] = outside.astype(jp.float32)
    metrics["navigation/termination_fallen"] = fallen.astype(jp.float32)
    metrics["navigation/termination_timeout"] = timed_out.astype(jp.float32)
    return state.replace(
        data=data,
        obs=obs,
        reward=reward,
        done=done.astype(reward.dtype),
        metrics=metrics,
        info=info,
    )

  def _get_observation(self, data, info):
    config = self._config
    position, yaw, velocity_body, velocity_world, yaw_rate = self._robot_state(
        data
    )
    _, _, noise_fraction, _ = self._difficulty(info["curriculum_stage"])
    noisy_position, noisy_yaw = position, yaw
    noisy_velocity_body, noisy_yaw_rate = velocity_body, yaw_rate
    obstacle_position = info["obstacle_position"]
    obstacle_velocity = info["obstacle_velocity"]
    obstacle_radius = info["obstacle_radius"]
    obstacle_active = info["obstacle_active"]
    if config.perception.enabled or config.robot_state_randomization.enabled:
      info["rng"], *keys = jax.random.split(info["rng"], 12)
      robot_noise = config.robot_state_randomization
      if robot_noise.enabled:
        info["robot_position_walk"] += (
            jax.random.normal(keys[0], (2,))
            * robot_noise.position_random_walk_std
            * noise_fraction
        )
        info["robot_yaw_walk"] += (
            jax.random.normal(keys[1])
            * robot_noise.yaw_random_walk_std
            * noise_fraction
        )
        noisy_position += (
            info["robot_position_bias"] + info["robot_position_walk"]
        ) * noise_fraction + jax.random.normal(
            keys[2], (2,)
        ) * robot_noise.position_std * noise_fraction
        noisy_yaw += (
            info["robot_yaw_bias"] + info["robot_yaw_walk"]
        ) * noise_fraction + jax.random.normal(
            keys[3]
        ) * robot_noise.yaw_std * noise_fraction
        noisy_velocity_body += (
            jax.random.normal(keys[4], (2,))
            * robot_noise.velocity_std
            * noise_fraction
        )
        noisy_yaw_rate += (
            jax.random.normal(keys[5])
            * robot_noise.yaw_rate_std
            * noise_fraction
        )
      if config.perception.enabled:
        obstacle_position += (
            info["perception_position_bias"] * noise_fraction
            + jax.random.normal(keys[6], obstacle_position.shape)
            * config.perception.position_std
            * noise_fraction
        )
        obstacle_velocity += (
            jax.random.normal(keys[7], obstacle_velocity.shape)
            * config.perception.velocity_std
            * noise_fraction
        )
        obstacle_radius = jp.maximum(
            0.02,
            obstacle_radius
            + jax.random.normal(keys[8], obstacle_radius.shape)
            * config.perception.radius_std
            * noise_fraction,
        )
        dropout = (
            jax.random.uniform(keys[9], obstacle_radius.shape)
            < config.perception.dropout_probability * noise_fraction
        )
        obstacle_active &= ~dropout
    noisy_yaw = _wrap_angle(noisy_yaw)

    robot_delayed = (
        config.robot_state_randomization.enabled
        & (config.robot_state_randomization.latency_steps == 1)
        & (noise_fraction >= 0.5)
    )
    current_robot = (
        noisy_position,
        noisy_yaw,
        noisy_velocity_body,
        noisy_yaw_rate,
    )
    delayed_robot = (
        info["previous_actor_position"],
        info["previous_actor_yaw"],
        info["previous_actor_velocity_body"],
        info["previous_actor_yaw_rate"],
    )
    noisy_position, noisy_yaw, noisy_velocity_body, noisy_yaw_rate = tuple(
        jp.where(robot_delayed, old, new)
        for old, new in zip(delayed_robot, current_robot)
    )
    (
        info["previous_actor_position"],
        info["previous_actor_yaw"],
        info["previous_actor_velocity_body"],
        info["previous_actor_yaw_rate"],
    ) = current_robot

    perception_delayed = (
        config.perception.enabled
        & (config.perception.latency_steps == 1)
        & (noise_fraction >= 0.5)
    )
    current_perception = (
        obstacle_position,
        obstacle_velocity,
        obstacle_radius,
        obstacle_active,
    )
    delayed_perception = (
        info["previous_perceived_obstacle_position"],
        info["previous_perceived_obstacle_velocity"],
        info["previous_perceived_obstacle_radius"],
        info["previous_perceived_obstacle_active"],
    )
    obstacle_position, obstacle_velocity, obstacle_radius, obstacle_active = (
        tuple(
            jp.where(perception_delayed, old, new)
            for old, new in zip(delayed_perception, current_perception)
        )
    )
    (
        info["previous_perceived_obstacle_position"],
        info["previous_perceived_obstacle_velocity"],
        info["previous_perceived_obstacle_radius"],
        info["previous_perceived_obstacle_active"],
    ) = current_perception
    noisy_velocity_world = jp.asarray((
        jp.cos(noisy_yaw) * noisy_velocity_body[0]
        - jp.sin(noisy_yaw) * noisy_velocity_body[1],
        jp.sin(noisy_yaw) * noisy_velocity_body[0]
        + jp.cos(noisy_yaw) * noisy_velocity_body[1],
    ))
    actor_selected = dpcbf.select_obstacles(
        obstacle_position,
        obstacle_velocity,
        obstacle_radius,
        obstacle_active,
        noisy_position,
        noisy_velocity_world,
        p_max=config.obstacle.p_max,
        num_constraints=config.obstacle.num_constraints,
        priority=config.obstacle.priority,
    )
    critic_selected = dpcbf.select_obstacles(
        info["obstacle_position"],
        info["obstacle_velocity"],
        info["obstacle_radius"],
        info["obstacle_active"],
        position,
        velocity_world,
        p_max=config.obstacle.p_max,
        num_constraints=config.obstacle.num_constraints,
        priority=config.obstacle.priority,
    )
    actor_obs = self._compose_local_observation(
        actor_selected,
        noisy_position,
        noisy_yaw,
        noisy_velocity_body,
        noisy_yaw_rate,
        noisy_velocity_world,
        info,
    )
    critic_obs = self._compose_local_observation(
        critic_selected,
        position,
        yaw,
        velocity_body,
        yaw_rate,
        velocity_world,
        info,
    )
    return {"state": actor_obs, "privileged_state": critic_obs}, info

  def _compose_local_observation(  # noqa: PLR0917
      self,
      selected,
      robot_position,
      yaw,
      velocity_body,
      yaw_rate,
      velocity_world,
      info,
  ):
    count = self._config.obstacle.num_constraints
    nodes = jp.zeros((count + 2, 9))
    nodes = nodes.at[0, 0].set(1.0)
    nodes = nodes.at[0, 5].set(self._config.robot.radius)
    nodes = nodes.at[0, 8].set(1.0)
    rel_pos = dpcbf.world_to_body(selected.relative_position, yaw)
    rel_vel = dpcbf.world_to_body(selected.relative_velocity, yaw)
    nodes = nodes.at[1 : count + 1, 1].set(1.0)
    nodes = nodes.at[1 : count + 1, 3:5].set(rel_pos)
    nodes = nodes.at[1 : count + 1, 5].set(selected.radius)
    nodes = nodes.at[1 : count + 1, 6:8].set(rel_vel)
    nodes = nodes.at[1 : count + 1, 8].set(selected.valid.astype(jp.float32))
    goal_index = count + 1
    goal_relative = dpcbf.world_to_body(info["goal"] - robot_position, yaw)
    goal_velocity = dpcbf.world_to_body(-velocity_world, yaw)
    nodes = nodes.at[goal_index, 2].set(1.0)
    nodes = nodes.at[goal_index, 3:5].set(goal_relative)
    nodes = nodes.at[goal_index, 5].set(self._config.goal.radius)
    nodes = nodes.at[goal_index, 6:8].set(goal_velocity)
    nodes = nodes.at[goal_index, 8].set(1.0)
    heading_error = _wrap_angle(info["goal_heading"] - yaw)
    local = jp.concatenate((
        goal_relative,
        jp.asarray((jp.sin(heading_error), jp.cos(heading_error))),
        velocity_body,
        jp.asarray((yaw_rate,)),
        info["velocity_command"],
        info["previous_raw_action"],
    ))
    return jp.concatenate((nodes.reshape(-1), local))

  def _zero_metrics(self):
    names = tuple(self._config.reward.keys())
    metrics = {f"reward/{name}": jp.asarray(0.0) for name in names}
    metrics.update({
        "navigation/goal_distance": jp.asarray(0.0),
        "navigation/collision": jp.asarray(0.0),
        "navigation/outside": jp.asarray(0.0),
        "navigation/qp_slack": jp.asarray(0.0),
        "navigation/qp_failed": jp.asarray(0.0),
        "navigation/hard_cbf_feasible_per_step": jp.asarray(0.0),
        "navigation/hard_cbf_minimum_relaxation_per_step": jp.asarray(0.0),
        "navigation/hard_cbf_numerical_failure_per_step": jp.asarray(0.0),
        "navigation/dodger_min_dhdt_alphah_per_step": jp.asarray(0.0),
        "navigation/dodger_constraint_violation_per_step": jp.asarray(0.0),
        "navigation/dodger_delta_per_step": jp.asarray(0.0),
        "navigation/dodger_cmax_per_step": jp.asarray(0.0),
        "navigation/dodger_probability_per_step": jp.asarray(0.0),
        "navigation/curriculum_stage_per_step": jp.asarray(0.0),
        "navigation/curriculum_success_rate_per_step": jp.asarray(0.0),
        "navigation/termination_goal_reached": jp.asarray(0.0),
        "navigation/termination_collision": jp.asarray(0.0),
        "navigation/termination_outside": jp.asarray(0.0),
        "navigation/termination_fallen": jp.asarray(0.0),
        "navigation/termination_timeout": jp.asarray(0.0),
    })
    return metrics
