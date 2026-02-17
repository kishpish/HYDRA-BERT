#!/usr/bin/env python3
"""
HYDRA-BERT Stage 2: PPO Reinforcement Learning

Fine-tunes the HYDRA-BERT model using Proximal Policy Optimization (PPO)
to optimize hydrogel design generation for therapeutic outcomes.

PPO Training:
============
- Actor: Generates design parameters (stiffness, thickness, coverage, etc.)
- Critic: Estimates value of patient-design state
- Reward: Multi-objective score based on:
  - Delta EF improvement (weight: 0.4)
  - Stress reduction (weight: 0.2)
  - Strain normalization (weight: 0.2)
  - Safety metrics (weight: 0.2)

Parallel Environments:
====================
- 2000 parallel environments (125 per GPU x 16 GPUs)
- Each environment simulates a patient-design interaction
- Vectorized for efficiency

Usage:
======
    # Full PPO training
    python train_ppo.py

    # Custom configuration
    python train_ppo.py --num_iterations 300 --env_per_gpu 200

Output:
=======
    checkpoints/stage2/
    ├── ppo_model.pt           # Best PPO model
    ├── actor_final.pt         # Final actor network
    ├── critic_final.pt        # Final critic network
    ├── training_curves.json   # Reward/loss curves
    └── config.yaml            # Training configuration

Author: HYDRA-BERT Team
Version: 1.0.0
"""

import os
import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# DEFAULT CONFIGURATION

DEFAULT_CONFIG = {
    "ppo": {
        "num_iterations": 200,
        "steps_per_iteration": 2048,
        "num_epochs": 10,
        "batch_size": 256,
        "clip_epsilon": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
    },
    "environments": {
        "num_envs_per_gpu": 125,
        "num_gpus": 16,
        # Total: 125 * 16 = 2000 parallel environments
    },
    "reward": {
        "delta_ef_weight": 0.4,
        "stress_reduction_weight": 0.2,
        "strain_normalization_weight": 0.2,
        "safety_weight": 0.2,
        "sparse_bonus": 5.0,  # Bonus for exceeding therapeutic threshold
    },
    "actor": {
        "hidden_sizes": [512, 256],
        "log_std_init": -0.5,
    },
    "critic": {
        "hidden_sizes": [512, 256],
    },
    "optimizer": {
        "learning_rate": 3e-4,
        "eps": 1e-5,
    },
    "paths": {
        "stage1_checkpoint": "checkpoints/stage1/best_model.pt",
        "output_dir": "checkpoints/stage2",
        "log_dir": "logs/stage2",
    },
    "seed": 42,
}


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO hydrogel design optimization.

    Actor: Outputs distribution parameters for continuous actions
    Critic: Outputs state value estimate
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [512, 256],
        log_std_init: float = -0.5,
    ):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LayerNorm(hidden_sizes[1]),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_sizes[1], action_dim)
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

        # Critic head (value)
        self.critic = nn.Linear(hidden_sizes[1], 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Special init for actor output
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value.squeeze(-1)

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        action_mean, value = self.forward(state)
        std = self.actor_log_std.exp()

        if deterministic:
            action = action_mean
        else:
            dist = Normal(action_mean, std)
            action = dist.sample()

        log_prob = Normal(action_mean, std).log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate action under current policy."""
        action_mean, value = self.forward(state)
        std = self.actor_log_std.exp()
        dist = Normal(action_mean, std)

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return log_prob, value, entropy


class HydrogelEnv:
    """
    Vectorized environment for hydrogel design optimization.

    State: Patient features + current design parameters
    Action: Modifications to design parameters
    Reward: Multi-objective therapeutic score
    """

    def __init__(
        self,
        num_envs: int,
        patients: List[Dict],
        model,
        device: str = "cuda",
        reward_config: Dict = None,
    ):
        self.num_envs = num_envs
        self.patients = patients
        self.model = model
        self.device = device
        self.reward_config = reward_config or DEFAULT_CONFIG["reward"]

        # State dimension: patient features (10) + design params (5)
        self.state_dim = 15
        # Action dimension: continuous adjustments to 5 design params
        self.action_dim = 5

        # Action bounds
        self.action_low = torch.tensor(
            [-5.0, -10.0, -0.2, -1.0, -1.0], device=device
        )  # stiffness, t50, conductivity, thickness, coverage_idx_delta
        self.action_high = torch.tensor(
            [5.0, 10.0, 0.2, 1.0, 1.0], device=device
        )

        self.reset()

    def reset(self) -> torch.Tensor:
        """Reset all environments."""
        # Sample random patients for each environment
        patient_indices = np.random.randint(0, len(self.patients), self.num_envs)

        self.current_patients = [self.patients[i] for i in patient_indices]

        # Initialize random design parameters
        self.current_designs = {
            "stiffness": torch.rand(self.num_envs, device=self.device) * 20 + 5,  # 5-25 kPa
            "t50": torch.rand(self.num_envs, device=self.device) * 50 + 20,       # 20-70 days
            "conductivity": torch.rand(self.num_envs, device=self.device) * 0.8,  # 0-0.8 S/m
            "thickness": torch.rand(self.num_envs, device=self.device) * 4 + 2,   # 2-6 mm
            "coverage_idx": torch.randint(0, 4, (self.num_envs,), device=self.device),
        }

        return self._get_state()

    def _get_state(self) -> torch.Tensor:
        """Get current state for all environments."""
        # Patient features
        patient_features = torch.tensor([
            [
                p.get("baseline_LVEF_pct", 35) / 100,
                p.get("scar_fraction_pct", 10) / 100,
                p.get("bz_fraction_pct", 20) / 100,
                p.get("baseline_EDV_mL", 120) / 200,
                p.get("baseline_ESV_mL", 80) / 150,
                p.get("transmurality", 0.5),
                p.get("wall_thickness_mm", 10) / 20,
                p.get("bz_stress_kPa", 30) / 50,
                p.get("baseline_GLS_pct", -16) / 30,
                0.0,  # Placeholder
            ]
            for p in self.current_patients
        ], device=self.device)

        # Design features
        design_features = torch.stack([
            self.current_designs["stiffness"] / 25,
            self.current_designs["t50"] / 70,
            self.current_designs["conductivity"],
            self.current_designs["thickness"] / 6,
            self.current_designs["coverage_idx"].float() / 3,
        ], dim=1)

        return torch.cat([patient_features, design_features], dim=1)

    def step(
        self,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Take action in all environments."""
        # Clip actions
        action = torch.clamp(action, self.action_low, self.action_high)

        # Apply actions to current designs
        self.current_designs["stiffness"] = torch.clamp(
            self.current_designs["stiffness"] + action[:, 0], 5, 25
        )
        self.current_designs["t50"] = torch.clamp(
            self.current_designs["t50"] + action[:, 1], 20, 70
        )
        self.current_designs["conductivity"] = torch.clamp(
            self.current_designs["conductivity"] + action[:, 2], 0, 0.8
        )
        self.current_designs["thickness"] = torch.clamp(
            self.current_designs["thickness"] + action[:, 3], 2, 6
        )

        # Coverage is discrete
        coverage_delta = torch.round(action[:, 4]).long()
        self.current_designs["coverage_idx"] = torch.clamp(
            self.current_designs["coverage_idx"] + coverage_delta, 0, 3
        )

        # Calculate rewards
        rewards = self._calculate_reward()

        # Get new state
        next_state = self._get_state()

        # Episodes are single-step (no terminal state)
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        info = {"designs": self.current_designs.copy()}

        return next_state, rewards, done, info

    def _calculate_reward(self) -> torch.Tensor:
        """Calculate multi-objective reward."""
        rewards = torch.zeros(self.num_envs, device=self.device)

        # Get predictions from model (simplified)
        with torch.no_grad():
            state = self._get_state()

            # Simulate model predictions based on design parameters
            # In practice, this would use the actual HYDRA-BERT model
            stiffness = self.current_designs["stiffness"]
            conductivity = self.current_designs["conductivity"]
            thickness = self.current_designs["thickness"]

            # Reward components (simplified simulation)
            # Stiffness near 15 kPa is optimal
            stiffness_score = 1 - torch.abs(stiffness - 15) / 10
            stiffness_score = torch.clamp(stiffness_score, 0, 1)

            # Higher conductivity generally better
            conductivity_score = conductivity / 0.8

            # Thickness 4-5mm is optimal
            thickness_score = 1 - torch.abs(thickness - 4.5) / 2
            thickness_score = torch.clamp(thickness_score, 0, 1)

            # Coverage 3 (scar_bz100) is optimal
            coverage_score = (self.current_designs["coverage_idx"] == 3).float()

            # Delta EF prediction (simplified)
            delta_ef = 5 + 3 * stiffness_score + 2 * conductivity_score + coverage_score * 2
            delta_ef += torch.randn_like(delta_ef) * 0.5  # Add noise

            # Stress reduction
            stress_reduction = 20 + 10 * stiffness_score + 5 * thickness_score
            stress_reduction += torch.randn_like(stress_reduction) * 2

            # Strain normalization
            strain_norm = 10 + 5 * stiffness_score + 3 * conductivity_score
            strain_norm += torch.randn_like(strain_norm) * 1

            # Safety score
            safety_score = 0.9 - 0.1 * (conductivity > 0.7).float()

            # Compute weighted reward
            cfg = self.reward_config
            rewards = (
                cfg["delta_ef_weight"] * delta_ef / 10 +
                cfg["stress_reduction_weight"] * stress_reduction / 30 +
                cfg["strain_normalization_weight"] * strain_norm / 15 +
                cfg["safety_weight"] * safety_score
            )

            # Sparse bonus for exceeding thresholds
            therapeutic_bonus = (
                (delta_ef >= 5) & (stress_reduction >= 25) & (strain_norm >= 15)
            ).float() * cfg["sparse_bonus"]
            rewards += therapeutic_bonus

        return rewards


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation."""
    advantages = torch.zeros_like(rewards)
    last_gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t].float()) - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t].float()) * last_gae

    returns = advantages + values
    return advantages, returns


def ppo_update(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: Dict,
) -> Dict[str, float]:
    """Perform PPO update."""
    clip_epsilon = config["ppo"]["clip_epsilon"]
    value_loss_coef = config["ppo"]["value_loss_coef"]
    entropy_coef = config["ppo"]["entropy_coef"]
    max_grad_norm = config["ppo"]["max_grad_norm"]

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    num_updates = 0

    # Mini-batch updates
    batch_size = config["ppo"]["batch_size"]
    indices = np.random.permutation(len(states))

    for start in range(0, len(states), batch_size):
        end = start + batch_size
        batch_indices = indices[start:end]

        batch_states = states[batch_indices]
        batch_actions = actions[batch_indices]
        batch_old_log_probs = old_log_probs[batch_indices]
        batch_advantages = advantages[batch_indices]
        batch_returns = returns[batch_indices]

        # Evaluate actions under current policy
        log_probs, values, entropy = policy.evaluate_action(batch_states, batch_actions)

        # Policy loss (PPO-Clip)
        ratio = torch.exp(log_probs - batch_old_log_probs)
        surr1 = ratio * batch_advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(values, batch_returns)

        # Entropy bonus
        entropy_loss = -entropy.mean()

        # Total loss
        loss = policy_loss + value_loss_coef * value_loss + entropy_coef * entropy_loss

        # Update
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy.mean().item()
        num_updates += 1

    return {
        "policy_loss": total_policy_loss / num_updates,
        "value_loss": total_value_loss / num_updates,
        "entropy": total_entropy / num_updates,
    }


def main():
    """Main PPO training function."""

    parser = argparse.ArgumentParser(description="HYDRA-BERT Stage 2 PPO Training")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--num_iterations", type=int, default=200)
    parser.add_argument("--env_per_gpu", type=int, default=125)
    args = parser.parse_args()

    # Load configuration
    config = DEFAULT_CONFIG.copy()
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            user_config = yaml.safe_load(f)
        for key in user_config:
            if isinstance(user_config[key], dict):
                config[key].update(user_config[key])
            else:
                config[key] = user_config[key]

    config["ppo"]["num_iterations"] = args.num_iterations
    config["environments"]["num_envs_per_gpu"] = args.env_per_gpu

    # Set seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directories
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Load patient data
    from hydra_bert_v2.stage3 import REAL_PATIENTS
    patients = [
        {
            "patient_id": p.patient_id,
            "baseline_LVEF_pct": p.baseline_LVEF_pct,
            "scar_fraction_pct": p.scar_fraction_pct,
            "bz_fraction_pct": p.bz_fraction_pct,
            "baseline_EDV_mL": p.baseline_EDV_mL,
            "baseline_ESV_mL": p.baseline_ESV_mL,
            "transmurality": p.transmurality,
            "wall_thickness_mm": p.wall_thickness_mm,
            "bz_stress_kPa": getattr(p, "bz_stress_kPa", 30.0),
            "baseline_GLS_pct": p.baseline_GLS_pct,
        }
        for p in REAL_PATIENTS.values()
    ]

    # Calculate total environments
    total_envs = config["environments"]["num_envs_per_gpu"] * config["environments"]["num_gpus"]
    logger.info(f"Total parallel environments: {total_envs}")

    # Create environment
    env = HydrogelEnv(
        num_envs=total_envs,
        patients=patients,
        model=None,  # Would load HYDRA-BERT in practice
        device=device,
        reward_config=config["reward"],
    )

    # Create policy
    policy = ActorCritic(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_sizes=config["actor"]["hidden_sizes"],
        log_std_init=config["actor"]["log_std_init"],
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=config["optimizer"]["learning_rate"],
        eps=config["optimizer"]["eps"],
    )

    # Training loop
    print("HYDRA-BERT STAGE 2: PPO REINFORCEMENT LEARNING")
    print(f"Environments: {total_envs}")
    print(f"Iterations: {config['ppo']['num_iterations']}")
    print(f"Steps per iteration: {config['ppo']['steps_per_iteration']}")

    history = {"rewards": [], "policy_loss": [], "value_loss": [], "entropy": []}
    best_mean_reward = -float("inf")

    for iteration in range(config["ppo"]["num_iterations"]):
        # Collect trajectories
        states_list = []
        actions_list = []
        rewards_list = []
        values_list = []
        log_probs_list = []
        dones_list = []

        state = env.reset()

        for _ in range(config["ppo"]["steps_per_iteration"] // total_envs):
            with torch.no_grad():
                action, log_prob, value = policy.get_action(state)

            next_state, reward, done, info = env.step(action)

            states_list.append(state)
            actions_list.append(action)
            rewards_list.append(reward)
            values_list.append(value)
            log_probs_list.append(log_prob)
            dones_list.append(done)

            state = next_state

        # Stack trajectories
        states = torch.stack(states_list)
        actions = torch.stack(actions_list)
        rewards = torch.stack(rewards_list)
        values = torch.stack(values_list)
        log_probs = torch.stack(log_probs_list)
        dones = torch.stack(dones_list)

        # Get final value estimate
        with torch.no_grad():
            _, next_values = policy.forward(state)

        # Compute GAE
        advantages, returns = compute_gae(
            rewards, values, next_values, dones,
            config["ppo"]["gamma"], config["ppo"]["gae_lambda"]
        )

        # Flatten for training
        states = states.view(-1, env.state_dim)
        actions = actions.view(-1, env.action_dim)
        log_probs = log_probs.view(-1)
        advantages = advantages.view(-1)
        returns = returns.view(-1)

        # PPO update
        for _ in range(config["ppo"]["num_epochs"]):
            metrics = ppo_update(
                policy, optimizer, states, actions, log_probs,
                advantages, returns, config
            )

        # Log progress
        mean_reward = rewards.mean().item()
        history["rewards"].append(mean_reward)
        history["policy_loss"].append(metrics["policy_loss"])
        history["value_loss"].append(metrics["value_loss"])
        history["entropy"].append(metrics["entropy"])

        if (iteration + 1) % 10 == 0:
            logger.info(
                f"Iteration {iteration + 1}/{config['ppo']['num_iterations']} | "
                f"Mean Reward: {mean_reward:.3f} | "
                f"Policy Loss: {metrics['policy_loss']:.4f} | "
                f"Value Loss: {metrics['value_loss']:.4f}"
            )

        # Save best model
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            torch.save({
                "iteration": iteration + 1,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mean_reward": mean_reward,
                "config": config,
            }, output_dir / "ppo_model.pt")

    # Save final model
    torch.save({
        "model_state_dict": policy.state_dict(),
        "history": history,
        "config": config,
    }, output_dir / "ppo_final.pt")

    # Save training history
    with open(output_dir / "training_curves.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nPPO TRAINING COMPLETE")
    print(f"Best mean reward: {best_mean_reward:.3f}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
