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
        self.value_loss_threshold = 50.0  # value_function lossのしきい値
        self.checkpoint_history = deque(maxlen=3)  # 最新3つのチェックポイントを保存
        self.original_learning_rate = self.alg.learning_rate  # 元の学習率を保存
        self.learning_rate_decay_factor = 0.1  # 学習率の減衰係数
        self.recovery_attempts = 0  # 回復試行回数
        self.max_recovery_attempts = 100  # 最大回復試行回数
        
    def load(self, path: str, load_optimizer: bool = True):
        """チェックポイントからモデルをロードする（PyTorch推論モードの問題に対応）

        Args:
            path: チェックポイントのパス
            load_optimizer: オプティマイザーをロードするかどうか

        Returns:
            ロードされたチェックポイントの追加情報
        """
        print(f"Loading checkpoint from: {path} with safe tensor handling")
        loaded_dict = torch.load(path, weights_only=False)
        
        # モデルの状態をロード
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        
        # RNDモデルがある場合はロード
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
            
        # 経験的正規化を使用している場合
        if self.empirical_normalization:
            if resumed_training:
                # 安全にテンソルをロード（クローンを作成してからロード）
                try:
                    # 観測の正規化器の状態をロード
                    obs_norm_state = loaded_dict["obs_norm_state_dict"]
                    safe_obs_norm_state = {}
                    for key, value in obs_norm_state.items():
                        if isinstance(value, torch.Tensor):
                            safe_obs_norm_state[key] = value.clone().detach()
                        else:
                            safe_obs_norm_state[key] = value
                    self.obs_normalizer.load_state_dict(safe_obs_norm_state)
                    
                    # 特権観測の正規化器の状態をロード
                    priv_obs_norm_state = loaded_dict["privileged_obs_norm_state_dict"]
                    safe_priv_obs_norm_state = {}
                    for key, value in priv_obs_norm_state.items():
                        if isinstance(value, torch.Tensor):
                            safe_priv_obs_norm_state[key] = value.clone().detach()
                        else:
                            safe_priv_obs_norm_state[key] = value
                    self.privileged_obs_normalizer.load_state_dict(safe_priv_obs_norm_state)
                    
                    print("Successfully loaded normalizer states with safe tensor handling")
                except Exception as e:
                    print(f"Warning: Failed to load normalizer states: {e}")
                    print("Continuing with default normalizer states")
            else:
                # トレーニングが再開されない場合（蒸留トレーニングの場合など）
                try:
                    self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                    print("Loaded teacher normalizer state for distillation")
                except Exception as e:
                    print(f"Warning: Failed to load teacher normalizer state: {e}")
                    
        # オプティマイザーをロード
        if load_optimizer and resumed_training:
            try:
                # アルゴリズムのオプティマイザー
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                
                # RNDオプティマイザーがある場合
                if self.alg.rnd:
                    self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
                    
                print("Successfully loaded optimizer states")
            except Exception as e:
                print(f"Warning: Failed to load optimizer states: {e}")
                print("Continuing with default optimizer states")
                
        # 現在の学習イテレーションをロード
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
            
        return loaded_dict.get("infos", None)

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
                    # 標準偏差をチェックして安全に保つ
                    self._ensure_valid_action_std()
                    
                    # Sample actions
                    try:
                        actions = self.alg.act(obs, privileged_obs)
                    except RuntimeError as e:
                        if "normal expects all elements of std >= 0.0" in str(e):
                            print("[ERROR] Caught std < 0 error during action sampling.")
                            # 標準偏差を修正して再試行
                            self._ensure_valid_action_std(force_reset=True)
                            actions = self.alg.act(obs, privileged_obs)
                        else:
                            raise
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

            # ポリシー更新前に標準偏差を確認
            try:
                # 標準偏差が負の値や異常値になっていないか確認
                if hasattr(self.alg.policy, 'action_std'):
                    action_std = self.alg.policy.action_std
                    if torch.any(action_std < 0) or torch.any(torch.isnan(action_std)) or torch.any(torch.isinf(action_std)):
                        print(f"[WARNING] Detected invalid action_std values: min={action_std.min().item()}, max={action_std.max().item()}")
                        # チェックポイントからロードして回復
                        success, new_it, new_obs, new_privileged_obs = self._load_checkpoint_and_restart(it, "Invalid action_std detected", decay_factor=self.learning_rate_decay_factor)
                        if success:
                            # 状態を更新して学習を継続
                            it = new_it
                            obs = new_obs
                            privileged_obs = new_privileged_obs
                            # forループの次のイテレーションに進む
                            continue
                        else:
                            # 回復に失敗した場合は学習を終了
                            return False
            except Exception as e:
                print(f"Error checking action_std: {e}")

            # update policy
            try:
                loss_dict = self.alg.update()
            except RuntimeError as e:
                if "normal expects all elements of std >= 0.0" in str(e):
                    print("[ERROR] Caught std < 0 error during policy update.")
                    success, new_it, new_obs, new_privileged_obs = self._load_checkpoint_and_restart(it, "std < 0 error", decay_factor=self.learning_rate_decay_factor)
                    if success:
                        # 状態を更新して学習を継続
                        it = new_it
                        obs = new_obs
                        privileged_obs = new_privileged_obs
                        # forループの次のイテレーションに進む
                        continue
                    else:
                        # 回復に失敗した場合は学習を終了
                        return False
                else:
                    # その他のエラーは再発生
                    raise

            # チェックポイントの保存と監視
            if self.log_dir is not None and not self.disable_logs:
                # 定期的なチェックポイントの保存
                if it % self.save_interval == 0:
                    checkpoint_path = os.path.join(self.log_dir, f"model_{it}.pt")
                    self.save(checkpoint_path)
                    self.checkpoint_history.append((it, checkpoint_path))
                
                # value_function lossの監視
                if "value_function" in loss_dict and (loss_dict["value_function"] > self.value_loss_threshold or 
                                                     torch.isinf(torch.tensor(loss_dict["value_function"])) or 
                                                     torch.isnan(torch.tensor(loss_dict["value_function"]))):
                    print(f"\n[WARNING] Value function loss ({loss_dict['value_function']}) exceeded threshold ({self.value_loss_threshold:.2f}) or is inf/nan")
                    success, new_it, new_obs, new_privileged_obs = self._load_checkpoint_and_restart(it, "Value function loss exceeded threshold", decay_factor=self.learning_rate_decay_factor)
                    if success:
                        # 状態を更新して学習を継続
                        it = new_it
                        obs = new_obs
                        privileged_obs = new_privileged_obs
                        # forループの次のイテレーションに進む
                        continue
                    else:
                        # 回復に失敗した場合は学習を終了
                        return False

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
            
    def _load_checkpoint_and_restart(self, current_it, reason, decay_factor=None):
        """チェックポイントからロードして学習を再開するヘルパーメソッド

        Args:
            current_it: 現在のイテレーション
            reason: 再開の理由
            decay_factor: 学習率の減衰係数（Noneの場合はself.learning_rate_decay_factorを使用）

        Returns:
            tuple: (成功したかどうか, 新しいイテレーション, 新しい観測, 新しい特権観測)
        """
        self.recovery_attempts += 1
        if self.recovery_attempts > self.max_recovery_attempts:
            print(f"[ERROR] Maximum recovery attempts ({self.max_recovery_attempts}) reached. Stopping training.")
            return False
            
        print(f"[RECOVERY ATTEMPT {self.recovery_attempts}/{self.max_recovery_attempts}] Reason: {reason}")
        
        # チェックポイントがあるか確認
        if len(self.checkpoint_history) == 0:
            print("[ERROR] No checkpoints available for recovery.")
            # チェックポイントがない場合でも、適切なタプルを返す
            return False, None, None, None
            
        # 最も古いチェックポイントを取得
        oldest_it, oldest_checkpoint = self.checkpoint_history[0]
        print(f"Loading checkpoint from iteration {oldest_it}")
        
        # チェックポイントをロード
        self.load(oldest_checkpoint, load_optimizer=True)
        
        # 学習率を減衰
        if decay_factor is None:
            decay_factor = self.learning_rate_decay_factor
        
        # チェックポイントロード前の学習率を保存
        pre_load_learning_rate = self.alg.learning_rate
        
        # チェックポイントをロードした後、学習率を適切に設定
        # チェックポイントの学習率ではなく、現在の減衰した学習率を使用
        current_learning_rate = pre_load_learning_rate * decay_factor
        print(f"Reducing learning rate from {pre_load_learning_rate:.8f} to {current_learning_rate:.8f}")
        self.alg.learning_rate = current_learning_rate
        
        # 学習率の履歴を保存（デバッグ用）
        if not hasattr(self, 'learning_rate_history'):
            self.learning_rate_history = []
        self.learning_rate_history.append((self.recovery_attempts, current_learning_rate))
        print(f"Learning rate history: {self.learning_rate_history}")
        
        # 環境のリソースを適切に管理して再初期化
        try:
            print("Preparing environment for checkpoint reload...")
            
            # ガベージコレクションを明示的に実行してメモリリークを防止
            import gc
            gc.collect()
            
            # 環境をリセットする前に長めの一時停止を入れる（リソースの解放のため）
            print("Waiting for resources to be released...")
            time.sleep(5.0)
            
            # トーチのキャッシュをクリア
            if hasattr(torch, 'cuda'):
                torch.cuda.empty_cache()
            
            # 環境をリセット
            print("Attempting to reset environment...")
            obs, extras = self.env.reset()
            privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
            obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
            
            # エピソード長バッファをリセット
            self.env.episode_length_buf = torch.zeros_like(self.env.episode_length_buf)
            
            # 追加のガベージコレクション
            gc.collect()
            if hasattr(torch, 'cuda'):
                torch.cuda.empty_cache()
                
            print("Environment successfully reset")
        except Exception as e:
            print(f"Error resetting environment: {e}")
            print("Trying to get observations without reset...")
            try:
                # ガベージコレクションを実行
                import gc
                gc.collect()
                if hasattr(torch, 'cuda'):
                    torch.cuda.empty_cache()
                    
                obs, extras = self.env.get_observations()
                privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
                obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
                print("Successfully got observations without reset")
            except Exception as e2:
                print(f"Error getting observations: {e2}")
                print("Attempting to continue with previous observations...")
                return False
        
        # 現在のイテレーションを更新
        self.current_learning_iteration = oldest_it
        
        # チェックポイント履歴から使用したチェックポイントのみを削除
        # 全てクリアするのではなく、使用したチェックポイントだけを削除
        if len(self.checkpoint_history) > 0:
            self.checkpoint_history.popleft()  # 最も古いチェックポイントを削除
        
        print(f"Resumed training from iteration {oldest_it} with reduced learning rate")
        
        # 学習を再開するための状態を返す
        print(f"Returning state for iteration {oldest_it}")
        
        # 成功したかどうか、新しいイテレーション、新しい観測、新しい特権観測を返す
        return True, oldest_it, obs, privileged_obs
        
    def _ensure_valid_action_std(self, force_reset=False, min_std=1e-8, max_std=1e+8):
        """標準偏差が有効な値であることを確認する

        Args:
            force_reset: 強制的に標準偏差をリセットするかどうか
            min_std: 最小の標準偏差値（0/nanの場合に使用）
            max_std: 最大の標準偏差値（infの場合に使用）
        """
        try:
            if hasattr(self.alg.policy, 'action_std'):
                action_std = self.alg.policy.action_std
                
                # 標準偏差の問題をチェック
                has_small_values = torch.any(action_std < min_std)
                has_nan_values = torch.any(torch.isnan(action_std))
                has_inf_values = torch.any(torch.isinf(action_std))
                
                # 問題がある場合または強制リセットが要求されている場合
                if force_reset or has_small_values or has_nan_values or has_inf_values:
                    if not force_reset:
                        print(f"[WARNING] Detected problematic action_std values: min={action_std.min().item() if not has_nan_values else 'NaN'}, max={action_std.max().item() if not has_inf_values else 'Inf'}")
                    else:
                        print("[INFO] Forcing action_std reset")
                    
                    # 要素ごとに適切な値に修正
                    with torch.no_grad():
                        # 新しい標準偏差テンソルを作成
                        new_std = action_std.clone()
                        
                        # 小さすぎる値またはNaNを最小値に置き換え
                        small_or_nan_mask = (new_std < min_std) | torch.isnan(new_std)
                        if torch.any(small_or_nan_mask):
                            new_std[small_or_nan_mask] = min_std
                            print(f"Reset {torch.sum(small_or_nan_mask).item()} small/NaN values to {min_std}")
                        
                        # 無限大の値を最大値に置き換え
                        inf_mask = torch.isinf(new_std)
                        if torch.any(inf_mask):
                            new_std[inf_mask] = max_std
                            print(f"Reset {torch.sum(inf_mask).item()} infinite values to {max_std}")
                        
                        # 更新された標準偏差を設定
                        self.alg.policy.action_std.copy_(new_std)
                        
                    print(f"Action std updated: min={new_std.min().item()}, max={new_std.max().item()}, mean={new_std.mean().item()}")
        except Exception as e:
            print(f"Error in _ensure_valid_action_std: {e}")
            
    # 終了時のリソースクリーンアップ
    def _cleanup_resources(self):
        """終了時のリソースクリーンアップ"""
        print("Cleaning up resources before exit...")
        try:
            # ガベージコレクションを実行
            import gc
            gc.collect()
            
            # PyTorchのCUDAキャッシュをクリア
            if hasattr(torch, 'cuda'):
                torch.cuda.empty_cache()
                
            # 明示的にバッファをクリア
            if hasattr(self, 'checkpoint_history'):
                self.checkpoint_history.clear()
                
            print("Resource cleanup completed")
        except Exception as e:
            print(f"Warning: Error during resource cleanup: {e}")
