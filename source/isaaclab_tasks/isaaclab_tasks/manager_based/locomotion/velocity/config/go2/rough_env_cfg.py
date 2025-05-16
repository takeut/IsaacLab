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
        
        # add by kobayashi
        # self.scene.robot.init_state.pos = (0.0, 0.0, 0.2)  # 初期位置を変更
        # self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)  # 横倒れ（x軸に90度回転）

        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        self.events.push_robot = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
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

        # rewards for feet
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.feet_clearance.weight = 25.0
        self.rewards.feet_clearance.params["threshold"] = 0.1

        # self.rewards.undesired_contacts = None

        # reward for move
        self.rewards.dof_torques_l2.weight = -0.0001 # 関節トルクの使用制限を緩める, より強い支持力の使用を許容
        self.rewards.track_lin_vel_xy_exp.weight = 0.7 # より穏やかな速度指令への追従を促す,急激な動作を抑制し、安定性を向上
        self.rewards.track_ang_vel_z_exp.weight = 0.3 # 回転動作を抑制,より直線的な動作を促進
        self.rewards.dof_acc_l2.weight = -2.8e-7 # より滑らかな加速を可能に, 急激な動作変化を抑制しつつ、必要な動作は許容

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

        # reward for stability
        self.rewards.base_height_l2.weight = -30.0
        self.rewards.flat_orientation_l2.weight = -1.8
        self.rewards.ang_vel_xy_l2.weight = -0.05

        self.rewards.joint_pos_limits.weight = -10.0

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.1

        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.physics_material.static_friction_range = (0.6, 1.0)
        self.events.physics_material.dynamic_friction_range = (0.6, 1.25)

        self.actions.joint_pos.scale= 0.4

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
        self.events.push_robot = None

        #add by kobayashi 
        if hasattr(self.observations.policy, "height_scan"):
            del self.observations.policy.height_scan
        if hasattr(self.observations.policy, "base_lin_vel"):
            del self.observations.policy.base_lin_vel
