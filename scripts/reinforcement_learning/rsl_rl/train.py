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
from datetime import datetime
import time
import glob
import re

from rsl_rl.runners import OnPolicyRunner

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

def createRslRlEnv(env_cfg, agent_cfg, log_dir):
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print("[INFO] Created isaac environment.")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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
    print("[INFO] Created RslRlVecEnvWrapper.")

    return env

def find_checkpoints(log_dir):
    """ログディレクトリ内のチェックポイントを見つける"""
    try:
        # チェックポイントファイルの一覧を取得
        checkpoint_files = glob.glob(os.path.join(log_dir, "model_*.pt"))
        
        if not checkpoint_files:
            print("[WARNING] No checkpoint files found in {log_dir}")
            return []
            
        # チェックポイントファイルをイテレーション番号でソート
        def get_iteration(filename):
            match = re.search(r'model_(\d+)\.pt', os.path.basename(filename))
            if match:
                return int(match.group(1))
            return 0
            
        sorted_checkpoints = sorted(checkpoint_files, key=get_iteration)
        print("[INFO] Found {len(sorted_checkpoints)} checkpoints in {log_dir}")
        return sorted_checkpoints
    except Exception as e:
        print("[WARNING] finding checkpoints: {e}")
        return []

def get_checkpoint_for_recovery(log_dir):
    """回復用のチェックポイントを取得する（最新から2つ前）"""
    checkpoints = find_checkpoints(log_dir)
    
    if len(checkpoints) < 3:
        print("[WARNING] Not enough checkpoints for recovery. Found only {len(checkpoints)} checkpoints.")
        if checkpoints:
            # 少なくとも1つのチェックポイントがある場合は最も古いものを使用
            print("[INFO] Using oldest available checkpoint: {checkpoints[0]}")
            return checkpoints[0]
        return None
    
    # 最新から2つ前のチェックポイントを取得
    recovery_checkpoint = checkpoints[-3]
    print("[INFO] Selected recovery checkpoint: {recovery_checkpoint} (3rd newest)")
    return recovery_checkpoint

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

    print("****** createRslRlEnv *******")
    # env = createRslRlEnv(env_cfg, agent_cfg, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print("[INFO] Created isaac environment.")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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
    print("[INFO] Created RslRlVecEnvWrapper.")

    print("****** finish createRslRlEnv *******")

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

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

    # run training
    recovery_attempts = 0
    max_recovery_attempts = 100
    while recovery_attempts > max_recovery_attempts:
        try:
            print("[INFO] runner.learn.")
            runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
            print("[INFO] finish runner.learn.")
            break
        except RuntimeError as e:
            loss_dict = runner.alg.update()
            value_loss_threshold = 50
            learning_rate_decay_factor = 0.5

            env.close()
            time.sleep(5)

            # recovery checkpoint
            recovery_checkpoint = get_checkpoint_for_recovery(log_dir)
            agent_cfg.load_checkpoint = recovery_checkpoint
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            
            # env = createRslRlEnv(env_cfg, agent_cfg, log_root_path, log_dir)
            # create isaac environment
            env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
            print("[INFO] Created isaac environment.")

            # convert to single-agent instance if required by the RL algorithm
            if isinstance(env.unwrapped, DirectMARLEnv):
                env = multi_agent_to_single_agent(env)

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
            print("[INFO] Created RslRlVecEnvWrapper.")
            
            if "normal expects all elements of std >= 0.0" in str(e) or (
                "value_function" in loss_dict and (loss_dict["value_function"] > value_loss_threshold or 
                                                    torch.isinf(torch.tensor(loss_dict["value_function"])) or 
                                                    torch.isnan(torch.tensor(loss_dict["value_function"])))):
                print("[ERROR] Caught std < 0 error during action sampling.")
                # 学習率を修正して再試行
                runner.load(resume_path)
                learning_rate = runner.alg.learning_rate
                new_learning_rate = learning_rate * learning_rate_decay_factor
                time.sleep(5)
                runner.alg.learning_rate = new_learning_rate
            else:
                raise

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
