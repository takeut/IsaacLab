#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper script for train.py that applies learning rate scaling."""

import os
import sys
import importlib.util
import logging

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train_wrapper.log')
    ]
)
logger = logging.getLogger(__name__)

# train.pyのパス
TRAIN_PY_PATH = os.path.join(os.path.dirname(__file__), "train.py")

def load_module_from_path(module_name, file_path):
    """ファイルパスからモジュールをロードする"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        logger.error(f"Could not load module from {file_path}")
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def apply_learning_rate_scale():
    """環境変数から学習率のスケールを取得し、適用する"""
    # 環境変数から学習率のスケールを取得
    learning_rate_scale_str = os.environ.get("LEARNING_RATE_SCALE")
    
    if not learning_rate_scale_str:
        return
        
    try:
        learning_rate_scale = float(learning_rate_scale_str)
        logger.info(f"Applying learning rate scale: {learning_rate_scale}")
        
        # モンキーパッチを適用
        # isaaclab_tasks.utils.hydra.hydra_task_configデコレータをオーバーライド
        from isaaclab_tasks.utils.hydra import hydra_task_config as original_hydra_task_config
        
        def patched_hydra_task_config(task_name, cfg_entry_point=None):
            """学習率をスケールするhydra_task_configデコレータ"""
            original_decorator = original_hydra_task_config(task_name, cfg_entry_point)
            
            def wrapper(func):
                original_func = original_decorator(func)
                
                def scaled_func(*args, **kwargs):
                    # 引数を取得
                    if len(args) >= 2:
                        env_cfg, agent_cfg = args[0], args[1]
                        
                        # 学習率をスケール
                        if hasattr(agent_cfg.algorithm, "learning_rate"):
                            original_lr = agent_cfg.algorithm.learning_rate
                            agent_cfg.algorithm.learning_rate *= learning_rate_scale
                            logger.info(f"Scaled learning rate from {original_lr} to {agent_cfg.algorithm.learning_rate}")
                        else:
                            logger.warning("Could not find learning_rate attribute in agent_cfg.algorithm")
                    
                    # 元の関数を呼び出す
                    return original_func(*args, **kwargs)
                
                return scaled_func
            
            return wrapper
        
        # モンキーパッチを適用
        import isaaclab_tasks.utils.hydra
        isaaclab_tasks.utils.hydra.hydra_task_config = patched_hydra_task_config
        
    except ValueError:
        logger.error(f"Invalid learning rate scale: {learning_rate_scale_str}")

if __name__ == "__main__":
    # 学習率のスケールを適用
    apply_learning_rate_scale()
    
    # train.pyを実行
    logger.info("Executing train.py")
    
    # train.pyのパスを取得
    train_py_path = os.path.abspath(TRAIN_PY_PATH)
    
    # train.pyを実行
    with open(train_py_path, 'r') as f:
        train_code = f.read()
    
    # train.pyのコードを実行
    exec(train_code, {'__name__': '__main__', '__file__': train_py_path})
