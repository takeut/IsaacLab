# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom OnPolicyRunner with value function loss monitoring."""

import os
import time
import torch
from collections import deque

# from on_policy_runner import OnPolicyRunner
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.env import VecEnv
from rsl_rl.utils import store_code_state


class CustomOnPolicyRunner(OnPolicyRunner):
    """Custom OnPolicyRunner that monitors value function loss and reloads from checkpoint if it exceeds a threshold."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        """Initialize the custom runner.

        Args:
            env: The environment to train on.
            train_cfg: The training configuration.
            log_dir: The directory to save logs to.
            device: The device to use for training.
        """
        super().__init__(env, train_cfg, log_dir, device)
        
        # 追加の設定
        self.value_loss_threshold = 1000.0  # value_function lossのしきい値
        self.checkpoint_history = deque(maxlen=100)  # 最新100個のチェックポイントを保存
        self.original_learning_rate = self.alg.learning_rate  # 元の学習率を保存

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Override the learn method to monitor value function loss.

        Args:
            num_learning_iterations: The number of iterations to train for.
            init_at_random_ep_len: Whether to initialize at random episode lengths.
        """
        # 元のメソッドと同様の初期化
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # check if teacher is loaded
        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs, privileged_obs)
                    # Step the environment
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # perform normalization
                    obs = self.obs_normalizer(obs)
                    if self.privileged_obs_type is not None:
                        privileged_obs = self.privileged_obs_normalizer(
                            infos["observations"][self.privileged_obs_type].to(self.device)
                        )
                    else:
                        privileged_obs = obs

                    # process the step
                    self.alg.process_env_step(rewards, dones, infos)

                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards  # type: ignore
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        # -- common
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        # -- intrinsic and extrinsic rewards
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            # update policy
            loss_dict = self.alg.update()

            # チェックポイントの保存と監視
            if self.log_dir is not None and not self.disable_logs:
                # 定期的なチェックポイントの保存
                if it % self.save_interval == 0:
                    checkpoint_path = os.path.join(self.log_dir, f"model_{it}.pt")
                    self.save(checkpoint_path)
                    self.checkpoint_history.append((it, checkpoint_path))
                
                # value_function lossの監視
                if "value_function" in loss_dict and loss_dict["value_function"] > self.value_loss_threshold:
                    print(f"\n[WARNING] Value function loss ({loss_dict['value_function']:.2f}) exceeded threshold ({self.value_loss_threshold:.2f})")
                    
                    # 100回前のチェックポイントがあれば、そこからロード
                    if len(self.checkpoint_history) > 0:
                        # 最も古いチェックポイントを取得
                        oldest_it, oldest_checkpoint = self.checkpoint_history[0]
                        print(f"Loading checkpoint from iteration {oldest_it}")
                        
                        # チェックポイントをロード
                        self.load(oldest_checkpoint, load_optimizer=True)
                        
                        # 学習率を0.1倍に設定
                        new_learning_rate = self.original_learning_rate * 0.1
                        print(f"Reducing learning rate from {self.alg.learning_rate:.6f} to {new_learning_rate:.6f}")
                        self.alg.learning_rate = new_learning_rate
                        
                        # 観測を再取得
                        obs, extras = self.env.get_observations()
                        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
                        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
                        
                        # 現在のイテレーションを更新
                        self.current_learning_iteration = oldest_it
                        it = oldest_it
                        
                        # チェックポイント履歴をクリア
                        self.checkpoint_history.clear()
                        
                        print(f"Resumed training from iteration {oldest_it} with reduced learning rate")
                        continue

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
