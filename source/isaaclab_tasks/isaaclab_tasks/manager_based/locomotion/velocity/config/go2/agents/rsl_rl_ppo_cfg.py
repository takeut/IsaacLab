# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 50
    experiment_name = "unitree_go2_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    runner = RslRlOnPolicyRunnerCfg(
        obs_delay_steps = 0
    )


@configclass
class UnitreeGo2FlatPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 100000
        self.experiment_name = "unitree_go2_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]

        # さらに安定した学習のための設定
        # self.policy.init_noise_std = 0.8  # 探索ノイズを少し減らす
        self.algorithm.value_loss_coef = 0.5  # 値関数の損失係数を小さくする
        # self.algorithm.learning_rate = 0.0003  # 学習率をさらに小さくする
        # self.algorithm.max_grad_norm = 0.8  # 勾配のノルム制限を強化
        # self.algorithm.clip_param = 0.1  # クリッピングパラメータを小さくして更新を安定化
        # self.algorithm.desired_kl = 0.007  # KLダイバージェンスの目標値を小さくする
        self.algorithm.entropy_coef = 0.005  # エントロピー係数を減らす
        # self.empirical_normalization = True  # 経験的正規化を有効にして入力の分布を安定化
        
        # #normal
        self.policy.init_noise_std = 1.0
        # self.algorithm.value_loss_coef = 1.0
        # self.algorithm.learning_rate = 0.001
        self.empirical_normalization = False

        # モーターの遅延をシミュレートするためのアクション遅延ステップの数
        # self.runner.obs_delay_steps = 10
