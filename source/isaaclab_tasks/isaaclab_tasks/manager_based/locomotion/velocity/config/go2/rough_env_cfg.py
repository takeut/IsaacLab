# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        # self.events.push_robot = None  # コメントアウト（velocity_env_cfgでpush_robotがコメントアウトされているため）
        # self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards for feet - 足を上げるための報酬を調整
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 1.0e-5
        self.rewards.feet_clearance.weight = 1.0e-5
        self.rewards.feet_clearance.params["threshold"] = 0.08  # ユーザー設定に合わせて調整

        # self.rewards.undesired_contacts = None

        # reward for move
        self.rewards.dof_torques_l2.weight = -0.0002  # ユーザー設定に合わせて調整
        self.rewards.track_lin_vel_xy_exp.weight = 1.5  # 線形速度の追跡報酬を強化（前後左右の歩行性能向上）
        self.rewards.track_ang_vel_z_exp.weight = 0.75  # 角速度の追跡報酬を強化（回転性能向上）
        self.rewards.dof_acc_l2.weight = -2.5e-7  # ユーザー設定に合わせて調整

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

        # reward for stability
        self.rewards.base_height_l2.weight = -30.0  # ユーザー設定に合わせて調整
        self.rewards.flat_orientation_l2.weight = -0.5  # 平らな姿勢を維持するためのペナルティを強化
        # self.rewards.ang_vel_xy_l2.weight = -0.05  # XY平面での角速度に対するペナルティを強化

        self.commands.base_velocity.rel_standing_envs = 0.1

        # self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        # self.events.physics_material.static_friction_range = (0.6, 1.0)
        # self.events.physics_material.dynamic_friction_range = (0.6, 1.25)

        self.actions.joint_pos.scale= 0.45

@configclass
class UnitreeGo2RoughEnvCfg_PLAY(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        # self.events.push_robot = None  # コメントアウト（velocity_env_cfgでpush_robotがコメントアウトされているため）
