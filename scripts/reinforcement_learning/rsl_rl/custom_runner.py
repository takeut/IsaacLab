# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom runner for RSL-RL with error handling."""

import os
import time
import math
import torch
import numpy as np
from typing import Optional

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.algorithms import PPO


class CustomOnPolicyRunner(OnPolicyRunner):
    """Custom runner for RSL-RL with error handling for value function explosion.
    
    This runner extends the OnPolicyRunner from RSL-RL to add error handling for
    the case when the value function loss becomes infinity. When this happens,
    the runner will load a checkpoint from 100 iterations ago and continue training
    with a reduced learning rate.
    """

    def __init__(self, env, cfg, log_dir, device="cpu"):
        """Initialize the custom runner."""
        super().__init__(env, cfg, log_dir, device)
        # Keep track of checkpoints for recovery
        self.checkpoint_history = []
        # Maximum number of checkpoints to keep in history
        self.max_checkpoint_history = 5
        # Counter for recovery attempts
        self.recovery_attempts = 0
        # Maximum number of recovery attempts
        self.max_recovery_attempts = 10
        # Learning rate reduction factor on recovery
        self.lr_reduction_factor = 0.1
        # Make sure logger_type is initialized
        if not hasattr(self, 'logger_type'):
            self.logger_type = cfg.get('logger', 'tensorboard')
        # Make sure disable_logs is initialized
        if not hasattr(self, 'disable_logs'):
            self.disable_logs = False

    def save(self, path, infos=None):
        """Save the current state of the runner.
        
        Overrides the save method to keep track of checkpoint paths.
        """
        # Call the parent save method
        super().save(path, infos)
        
        # Add the checkpoint path to history
        self.checkpoint_history.append(path)
        
        # Keep only the most recent checkpoints
        if len(self.checkpoint_history) > self.max_checkpoint_history:
            self.checkpoint_history.pop(0)

    def get_recovery_checkpoint(self):
        """Get a checkpoint path from history for recovery.
            
        Returns:
            Path to the checkpoint for recovery, or None if no suitable checkpoint is found.
        """
        if not self.checkpoint_history:
            return None
        
        # Get the most recent checkpoint
        return self.checkpoint_history[-1]

    def reduce_learning_rate(self):
        """Reduce the learning rate of the algorithm."""
        if isinstance(self.alg, PPO):
            current_lr = self.alg.learning_rate
            new_lr = current_lr * self.lr_reduction_factor
            print(f"[INFO] Reducing learning rate from {current_lr} to {new_lr}")
            self.alg.learning_rate = new_lr
            
            # Update the optimizer's learning rate
            for param_group in self.alg.optimizer.param_groups:
                param_group['lr'] = new_lr

    def learn(self, num_learning_iterations=1e6, init_at_random_ep_len=False):
        """Train the policy.
        
        Overrides the learn method to add error handling for value function explosion.
        
        Args:
            num_learning_iterations: Number of iterations to train for.
            init_at_random_ep_len: Whether to initialize at a random episode length.
        """
        # Initialize the runner
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        
        # Main training loop
        current_iteration = self.current_learning_iteration
        while current_iteration < num_learning_iterations:
            try:
                # Collect rollout using parent class method
                start = time.time()
                super().collect_rollout()
                end = time.time()
                collection_time = end - start
                
                # Update policy
                start = time.time()
                self.alg.device = self.device
                
                # This is where the value function loss might become infinity
                mean_value_loss = self.alg.update(self.storage)
                
                # Check if value loss is infinity or NaN
                if math.isinf(mean_value_loss) or math.isnan(mean_value_loss):
                    raise ValueError(f"Value function loss is {mean_value_loss}")
                
                end = time.time()
                update_time = end - start
                
                # Log data using parent class method
                super().log(collection_time, update_time)
                
                # Save checkpoint periodically
                if self.current_learning_iteration % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
                
                current_iteration = self.current_learning_iteration
                self.current_learning_iteration += 1
                
            except ValueError as e:
                if "Value function loss" in str(e) and self.recovery_attempts < self.max_recovery_attempts:
                    self.recovery_attempts += 1
                    print(f"[WARNING] Value function loss became {str(e)}. Attempting recovery ({self.recovery_attempts}/{self.max_recovery_attempts})...")
                    
                    # Get recovery checkpoint
                    recovery_path = self.get_recovery_checkpoint()
                    if recovery_path is None:
                        print("[ERROR] No checkpoint available for recovery. Exiting...")
                        break
                    
                    print(f"[INFO] Loading checkpoint from {recovery_path}")
                    self.load(recovery_path)
                    
                    # Reduce learning rate
                    self.reduce_learning_rate()
                    
                    print("[INFO] Resuming training...")
                    continue
                else:
                    print(f"[ERROR] Fatal error or maximum recovery attempts reached: {str(e)}")
                    break
        
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
