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
        # self.value_loss_threshold = 50.0  # value_function lossのしきい値
        # self.checkpoint_history = deque(maxlen=6)  # 最新10個のチェックポイントを保存
        # self.used_checkpoints = set()  # 使用済みのチェックポイントを記録
        # self.original_learning_rate = self.alg.learning_rate  # 元の学習率を保存
        # self.learning_rate_decay_factor = 0.1  # 学習率の減衰係数
        # self.min_learning_rate = 1e-10  # 学習率の下限
        # self.recovery_attempts = 0  # 回復試行回数
        # self.max_recovery_attempts = 100  # 最大回復試行回数
        
    # def load(self, path: str, load_optimizer: bool = True):
    #     """チェックポイントからモデルをロードする（PyTorch推論モードの問題に対応）

    #     Args:
    #         path: チェックポイントのパス
    #         load_optimizer: オプティマイザーをロードするかどうか

    #     Returns:
    #         ロードされたチェックポイントの追加情報
    #     """
    #     print(f"Loading checkpoint from: {path} with safe tensor handling")
    #     loaded_dict = torch.load(path, weights_only=False)
        
    #     # モデルの状態をロード
    #     resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        
    #     # RNDモデルがある場合はロード
    #     if self.alg.rnd:
    #         self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
            
    #     # 経験的正規化を使用している場合
    #     if self.empirical_normalization:
    #         if resumed_training:
    #             # 安全にテンソルをロード（クローンを作成してからロード）
    #             try:
    #                 # 観測の正規化器の状態をロード
    #                 obs_norm_state = loaded_dict["obs_norm_state_dict"]
    #                 safe_obs_norm_state = {}
    #                 for key, value in obs_norm_state.items():
    #                     if isinstance(value, torch.Tensor):
    #                         safe_obs_norm_state[key] = value.clone().detach()
    #                     else:
    #                         safe_obs_norm_state[key] = value
    #                 self.obs_normalizer.load_state_dict(safe_obs_norm_state)
                    
    #                 # 特権観測の正規化器の状態をロード
    #                 priv_obs_norm_state = loaded_dict["privileged_obs_norm_state_dict"]
    #                 safe_priv_obs_norm_state = {}
    #                 for key, value in priv_obs_norm_state.items():
    #                     if isinstance(value, torch.Tensor):
    #                         safe_priv_obs_norm_state[key] = value.clone().detach()
    #                     else:
    #                         safe_priv_obs_norm_state[key] = value
    #                 self.privileged_obs_normalizer.load_state_dict(safe_priv_obs_norm_state)
                    
    #                 print("Successfully loaded normalizer states with safe tensor handling")
    #             except Exception as e:
    #                 print(f"Warning: Failed to load normalizer states: {e}")
    #                 print("Continuing with default normalizer states")
    #         else:
    #             # トレーニングが再開されない場合（蒸留トレーニングの場合など）
    #             try:
    #                 self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
    #                 print("Loaded teacher normalizer state for distillation")
    #             except Exception as e:
    #                 print(f"Warning: Failed to load teacher normalizer state: {e}")
                    
    #     # オプティマイザーをロード
    #     if load_optimizer and resumed_training:
    #         try:
    #             # アルゴリズムのオプティマイザー
    #             self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                
    #             # RNDオプティマイザーがある場合
    #             if self.alg.rnd:
    #                 self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
                    
    #             print("Successfully loaded optimizer states")
    #         except Exception as e:
    #             print(f"Warning: Failed to load optimizer states: {e}")
    #             print("Continuing with default optimizer states")
                
    #     # 現在の学習イテレーションをロード
    #     if resumed_training:
    #         self.current_learning_iteration = loaded_dict["iter"]
            
    #     return loaded_dict.get("infos", None)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
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
            # TODO: Do we need to synchronize empirical normalizers?
            #   Right now: No, because they all should converge to the same values "asymptotically".

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

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

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