# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def feet_clearance(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    #print("sensor_cfg.name ", sensor_cfg.name) #sensor_cfg.name  feet_position
    frame_transformer: FrameTransformer = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    #print("sensor_cfg.body_ids ", sensor_cfg.body_ids) #slice(None, None, None
    #feet_positon = frame_transformer.data.target_pos_w[:, sensor_cfg.body_ids]
    #print("feet_positon ", feet_positon) #torch.Size([num_envs, 4, 3])

    # feet_x = frame_transformer.data.target_pos_w[:, :, 0]
    # feet_y = frame_transformer.data.target_pos_w[:, :, 1]
    # print("feet_x ", feet_x)
    # print("feet_y ", feet_y)

    feet_height = frame_transformer.data.target_pos_w[:, :, 2]
    #print("feet_height ", feet_height)
    #body_lin_vel = asset.data.body_lin_vel_w #torch.Size([2048, 17, 3])
    feet_lin_vel = asset.data.body_lin_vel_w[:, 13:17, :] #torch.Size([2048, 4, 3])
   
    # Compute the height deviation
    height_deviation = threshold - feet_height  # Shape: [batch_size, num_feet]
    #print("height_deviation ", height_deviation)
    # Compute the squared deviation
    squared_deviation = height_deviation**2  # Shape: [batch_size, num_feet]
    #abs_deviation = abs(height_deviation)
    #ten_times_abs_deviation = abs(height_deviation) * 10

    # Compute the magnitude of the linear velocity in the xy-plane
    # Here we use the sum of squares in the xy-plane, ignoring the z-component
    lin_vel_xy = torch.norm(feet_lin_vel[:, :, :2], dim=2)  # Shape: [batch_size, num_feet]
    # lin_vel_xy = torch.clamp(lin_vel_xy, min=0, max=1)  # Shape: [batch_size, num_feet]
    #print("lin_vel_xy ", lin_vel_xy)
    #lin_vel_xy = torch.norm(feet_lin_vel[:, :, :2], dim=2)**0.5

    # height_deviation = torch.sum(feet_height -threshold)
    # foot_clearance_reward = torch.sum(height_deviation * lin_vel_xy, dim=1)

    # Compute the foot clearance reward term
    # I want deviation -> small -> foot_clerance_reward is negative but close to 0
    # clearance = torch.clamp(feet_height - threshold, min=0.0, max = 0.04)

    # foot_clearance_reward = torch.sum(clearance * lin_vel_xy, dim=1)  # Shape: [batch_size]
    foot_clearance_reward = -torch.sum(squared_deviation * lin_vel_xy, dim=1)  # Shape: [batch_size]
    #foot_clearance_reward = -torch.sum(abs_deviation * lin_vel_xy, dim=1)
    #foot_clearance_reward = -torch.sum(ten_times_abs_deviation * lin_vel_xy, dim=1)

    #print("foot_clearance_reward ", foot_clearance_reward)

    return foot_clearance_reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)
