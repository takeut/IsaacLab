# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import math
import time
import glob
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.algorithms import PPO

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # チェックポイントの履歴を保存する変数
    checkpoint_history = []
    # 最大チェックポイント履歴数
    max_checkpoint_history = 10
    # 回復試行回数
    recovery_attempts = 0
    # 最大回復試行回数
    max_recovery_attempts = 10
    # 学習率削減係数
    lr_reduction_factor = 0.1
    
    # logger_type属性を追加
    if not hasattr(runner, 'logger_type'):
        runner.logger_type = agent_cfg.logger
    
    # disable_logs属性を追加
    if not hasattr(runner, 'disable_logs'):
        runner.disable_logs = False
    
    # オリジナルのsaveメソッドを保存
    original_save = runner.save
    
    # saveメソッドをオーバーライド
    def custom_save(path, infos=None):
        # チェックポイントを保存
        try:
            # オリジナルのsaveメソッドを呼び出す
            original_save(path, infos)
        except Exception as e:
            print(f"[WARNING] Error saving checkpoint: {str(e)}")
            # 代替の保存方法
            checkpoint = {
                "model_state_dict": runner.alg.actor_critic.state_dict(),
                "optimizer_state_dict": runner.alg.optimizer.state_dict(),
                "iter": runner.current_learning_iteration,
                "infos": infos
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(checkpoint, path)
            print(f"[INFO] Checkpoint saved using alternative method: {path}")
        
        # チェックポイントの履歴に追加
        checkpoint_history.append(path)
        
        # 履歴が最大数を超えた場合、古いものから削除
        if len(checkpoint_history) > max_checkpoint_history:
            checkpoint_history.pop(0)
    
    # saveメソッドを置き換え
    runner.save = custom_save
    
    # オリジナルのupdateメソッドを保存
    original_update = runner.alg.update
    
    # updateメソッドをオーバーライド
    def custom_update(storage):
        nonlocal recovery_attempts
        
        try:
            # オリジナルのupdateメソッドを呼び出す
            mean_value_loss = original_update(storage)
            
            # 値関数の損失が1000を超えた場合
            if mean_value_loss > 1000 or math.isinf(mean_value_loss) or math.isnan(mean_value_loss):
                raise ValueError(f"Value function loss is {mean_value_loss}")
            
            return mean_value_loss
        except ValueError as e:
            if "Value function loss" in str(e) and recovery_attempts < max_recovery_attempts:
                recovery_attempts += 1
                print(f"[WARNING] {str(e)}. Attempting recovery ({recovery_attempts}/{max_recovery_attempts})...")
                
                # チェックポイントの履歴から100回前のチェックポイントを取得
                if len(checkpoint_history) > 0:
                    # 利用可能な最も古いチェックポイントを取得
                    recovery_path = checkpoint_history[0]
                    print(f"[INFO] Loading checkpoint from {recovery_path}")
                    
                    # チェックポイントを読み込む
                    runner.load(recovery_path)
                    
                    # 学習率を0.1倍に減らす
                    if isinstance(runner.alg, PPO):
                        current_lr = runner.alg.learning_rate
                        new_lr = current_lr * lr_reduction_factor
                        print(f"[INFO] Reducing learning rate from {current_lr} to {new_lr}")
                        runner.alg.learning_rate = new_lr
                        
                        # オプティマイザーの学習率も更新
                        for param_group in runner.alg.optimizer.param_groups:
                            param_group['lr'] = new_lr
                    
                    print("[INFO] Resuming training...")
                    
                    # ダミーの値を返す
                    return 0.0
                else:
                    print("[ERROR] No checkpoint available for recovery. Exiting...")
                    raise e
            else:
                print(f"[ERROR] Fatal error or maximum recovery attempts reached: {str(e)}")
                raise e
    
    # updateメソッドを置き換え
    runner.alg.update = custom_update
    
    # 初期チェックポイントを保存
    runner.save(os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt"))
    
    # run training
    current_iteration = runner.current_learning_iteration
    max_iterations = agent_cfg.max_iterations
    
    while current_iteration < max_iterations:
        try:
            # 一定回数の学習を実行
            next_save_iteration = current_iteration + agent_cfg.save_interval
            next_target = min(next_save_iteration, max_iterations)
            
            print(f"[INFO] Training from iteration {current_iteration} to {next_target}")
            runner.learn(num_learning_iterations=next_target, init_at_random_ep_len=(current_iteration == 0))
            
            # 現在の反復回数を更新
            current_iteration = runner.current_learning_iteration
            
        except Exception as e:
            print(f"[WARNING] Training error at iteration {current_iteration}: {str(e)}")
            
            # 最終チェックポイントを保存
            error_path = os.path.join(log_dir, f"model_error_{current_iteration}.pt")
            runner.save(error_path)
            
            # 回復処理
            if recovery_attempts < max_recovery_attempts:
                recovery_attempts += 1
                print(f"[INFO] Attempting recovery ({recovery_attempts}/{max_recovery_attempts})...")
                
                # チェックポイントの履歴から最も古いチェックポイントを取得
                if len(checkpoint_history) > 0:
                    recovery_path = checkpoint_history[0]
                    print(f"[INFO] Loading checkpoint from {recovery_path}")
                    
                    # チェックポイントを読み込む
                    runner.load(recovery_path)
                    current_iteration = runner.current_learning_iteration
                    
                    # 学習率を0.1倍に減らす
                    if isinstance(runner.alg, PPO):
                        current_lr = runner.alg.learning_rate
                        new_lr = current_lr * lr_reduction_factor
                        print(f"[INFO] Reducing learning rate from {current_lr} to {new_lr}")
                        runner.alg.learning_rate = new_lr
                        
                        # オプティマイザーの学習率も更新
                        for param_group in runner.alg.optimizer.param_groups:
                            param_group['lr'] = new_lr
                    
                    print(f"[INFO] Resuming training from iteration {current_iteration}")
                    continue
                else:
                    print("[ERROR] No checkpoint available for recovery. Exiting...")
                    break
            else:
                print(f"[ERROR] Maximum recovery attempts ({max_recovery_attempts}) reached. Exiting...")
                break
    
    # 最終チェックポイントを保存
    runner.save(os.path.join(log_dir, f"model_final_{runner.current_learning_iteration}.pt"))

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
