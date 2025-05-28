#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL with automatic recovery from errors."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import time
import glob
import re
import logging
import traceback
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mytrain.log')
    ]
)
logger = logging.getLogger(__name__)

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL with automatic recovery.")
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

# mytrain.py固有の引数を追加
parser.add_argument("--recovery_attempts", type=int, default=10, help="Maximum number of recovery attempts.")
parser.add_argument("--learning_rate_scale", type=float, default=0.5, help="Learning rate scale factor for recovery.")

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

"""Rest everything follows."""

import gymnasium as gym
import torch
import importlib.metadata as metadata
from packaging import version

# 必要なモジュールをインポート（AppLauncher初期化後）
import isaaclab_tasks
from isaaclab_tasks.utils import get_checkpoint_path


def find_latest_log_dir(base_path="logs/rsl_rl", experiment_name=None):
    """最新のログディレクトリを見つける"""
    try:
        # 実験名が指定されている場合は、その実験のディレクトリを使用
        if experiment_name:
            experiment_path = os.path.join(base_path, experiment_name)
            if not os.path.exists(experiment_path):
                logger.warning(f"Experiment directory {experiment_path} does not exist")
                return None
            base_path = experiment_path
        
        # ログディレクトリの一覧を取得
        log_dirs = glob.glob(os.path.join(base_path, "*"))
        
        # ディレクトリのみをフィルタリング
        log_dirs = [d for d in log_dirs if os.path.isdir(d)]
        
        if not log_dirs:
            logger.warning(f"No log directories found in {base_path}")
            return None
            
        # 最新のログディレクトリを返す
        latest_log_dir = max(log_dirs, key=os.path.getmtime)
        logger.info(f"Found latest log directory: {latest_log_dir}")
        return latest_log_dir
    except Exception as e:
        logger.error(f"Error finding latest log directory: {e}")
        return None


def find_checkpoints(log_dir):
    """ログディレクトリ内のチェックポイントを見つける"""
    try:
        # チェックポイントファイルの一覧を取得
        checkpoint_files = glob.glob(os.path.join(log_dir, "model_*.pt"))
        
        if not checkpoint_files:
            logger.warning(f"No checkpoint files found in {log_dir}")
            return []
            
        # チェックポイントファイルをイテレーション番号でソート
        def get_iteration(filename):
            match = re.search(r'model_(\d+)\.pt', os.path.basename(filename))
            if match:
                return int(match.group(1))
            return 0
            
        sorted_checkpoints = sorted(checkpoint_files, key=get_iteration)
        logger.info(f"Found {len(sorted_checkpoints)} checkpoints in {log_dir}")
        return sorted_checkpoints
    except Exception as e:
        logger.error(f"Error finding checkpoints: {e}")
        return []


def get_checkpoint_for_recovery(log_dir):
    """回復用のチェックポイントを取得する（最新から2つ前）"""
    checkpoints = find_checkpoints(log_dir)
    
    if len(checkpoints) < 3:
        logger.warning(f"Not enough checkpoints for recovery. Found only {len(checkpoints)} checkpoints.")
        if checkpoints:
            # 少なくとも1つのチェックポイントがある場合は最も古いものを使用
            logger.info(f"Using oldest available checkpoint: {checkpoints[0]}")
            return checkpoints[0]
        return None
    
    # 最新から2つ前のチェックポイントを取得
    recovery_checkpoint = checkpoints[-3]
    logger.info(f"Selected recovery checkpoint: {recovery_checkpoint} (3rd newest)")
    return recovery_checkpoint


def extract_run_name_from_path(path):
    """パスからrun名を抽出する（例：logs/rsl_rl/experiment_name/2025-04-23_18-32-29 -> 2025-04-23_18-32-29）"""
    if path is None:
        return None
    return os.path.basename(path)


def extract_checkpoint_name_from_path(path):
    """パスからチェックポイント名を抽出する（例：logs/rsl_rl/experiment_name/2025-04-23_18-32-29/model_4200.pt -> model_4200.pt）"""
    if path is None:
        return None
    return os.path.basename(path)


def run_train(args, recovery_mode=False, load_run=None, checkpoint=None, learning_rate_scale=1.0):
    """train_wrapper.pyをサブプロセスとして実行する"""
    try:
        # isaaclab.shのパスを取得
        isaaclab_script = "./isaaclab.sh" if os.path.exists("./isaaclab.sh") else "isaaclab.sh"
        
        # コマンドを構築
        cmd = [isaaclab_script, "-p", "scripts/reinforcement_learning/rsl_rl/train_wrapper.py"]
        
        # train.pyが受け付ける引数のみを渡す
        train_args = [
            "video", "video_length", "video_interval", "num_envs", "task", "seed", 
            "max_iterations", "distributed", "load_run", "checkpoint", "resume", "device",
            "headless", "enable_cameras"
        ]
        
        # argsの内容をコマンドライン引数に変換
        for arg_name, arg_value in vars(args).items():
            # train.pyが受け付ける引数のみを処理
            if arg_name not in train_args:
                continue
                
            if arg_value is not None:
                if isinstance(arg_value, bool):
                    if arg_value:
                        cmd.append(f"--{arg_name}")
                else:
                    cmd.append(f"--{arg_name}={arg_value}")
        
        # 回復モードの場合、チェックポイントからの再開に必要な引数を追加
        if recovery_mode and load_run and checkpoint:
            cmd.append(f"--load_run={load_run}")
            cmd.append(f"--checkpoint={checkpoint}")
            cmd.append("--resume=True")
            
            # 学習率を調整するための環境変数を設定
            if learning_rate_scale != 1.0:
                # 環境変数を通じて学習率のスケールを伝える
                # train.pyでこの環境変数を読み取る処理が必要
                os.environ["LEARNING_RATE_SCALE"] = str(learning_rate_scale)
                logger.info(f"Setting learning rate scale to {learning_rate_scale} via environment variable")
        
        # コマンドをログに記録
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # サブプロセスを実行
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1  # ラインバッファリング
        )
        
        # 出力を読み取り、表示
        if process.stdout:
            for line in process.stdout:
                print(line, end='')
                sys.stdout.flush()
        else:
            logger.warning("No stdout available from subprocess")
        
        # プロセスの終了を待つ
        return_code = process.wait()
        logger.info(f"Process exited with code {return_code}")
        return return_code
        
    except Exception as e:
        logger.error(f"Error in run_train: {e}")
        logger.error(traceback.format_exc())
        return 1


def main():
    # 必須引数のチェック
    if args_cli.task is None:
        logger.error("Task name is required. Please specify with --task.")
        return 1
    
    # experiment_nameが指定されていない場合のデフォルト値を設定
    if not hasattr(args_cli, 'experiment_name') or args_cli.experiment_name is None:
        # タスク名から実験名を推定
        if 'Go2' in args_cli.task and 'Flat' in args_cli.task:
            args_cli.experiment_name = 'unitree_go2_flat'
        else:
            args_cli.experiment_name = None
    
    # 回復試行回数のカウンタ
    recovery_attempt = 0
    
    while recovery_attempt <= args_cli.recovery_attempts:
        # 回復モードかどうか
        recovery_mode = recovery_attempt > 0
        
        if recovery_mode:
            logger.info(f"Starting recovery attempt {recovery_attempt}/{args_cli.recovery_attempts}")
            
            # 最新のログディレクトリを見つける
            log_dir = find_latest_log_dir(experiment_name=args_cli.experiment_name)
            if log_dir is None:
                logger.error("Could not find log directory for recovery.")
                return 1
                
            # 回復用のチェックポイントを取得
            checkpoint_path = get_checkpoint_for_recovery(log_dir)
            if checkpoint_path is None:
                logger.error("Could not find suitable checkpoint for recovery.")
                return 1
                
            # ログディレクトリとチェックポイントの名前を抽出
            load_run = extract_run_name_from_path(log_dir)
            checkpoint = extract_checkpoint_name_from_path(checkpoint_path)
            
            # 学習率のスケール係数を計算（回復試行ごとに0.5倍）
            learning_rate_scale = args_cli.learning_rate_scale ** recovery_attempt
            
            # train.pyを実行（回復モード）
            return_code = run_train(args_cli, recovery_mode=True, load_run=load_run, checkpoint=checkpoint, learning_rate_scale=learning_rate_scale)
        else:
            # 通常モードでtrain.pyを実行
            logger.info("Starting training in normal mode")
            return_code = run_train(args_cli)
        
        # 正常終了した場合
        if return_code == 0:
            logger.info("Training completed successfully.")
            return 0
            
        # エラーが発生した場合
        logger.warning(f"Training failed with return code {return_code}. Attempting recovery.")
        recovery_attempt += 1
        
        # 回復試行の間に少し待機
        time.sleep(5)
    
    logger.error(f"Maximum recovery attempts ({args_cli.recovery_attempts}) reached. Giving up.")
    return 1


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        sys.exit(130)  # 130はSIGINTの標準的な終了コード
    finally:
        # close sim app
        print("Closing simulation app...")
        simulation_app.close()
        print("Simulation app closed")
