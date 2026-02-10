#!/usr/bin/env python3
"""
Prediction Heads for HYDRA-BERT

Three prediction heads for multi-task learning:
1. GCS Regression: Predicts circumferential strain improvement
2. Outcome Classification: 3-class (harmful / beneficial / optimal)
3. Safety Head: Multi-output safety scores

Uses uncertainty-weighted multi-task loss with learned homoscedastic σ per task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class GCSRegressionHead(nn.Module):
    """
    Predicts Global Circumferential Strain (GCS) improvement.
    
    Output: Single scalar representing predicted strain improvement (%).
    Loss: Smooth L1 (Huber) loss for robustness to outliers.
    
    Architecture:
        fusion_dim → 256 → 128 → 64 → 1
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Fused representation (batch, input_dim)
        Returns:
            GCS prediction (batch, 1)
        """
        return self.network(x)


class OutcomeClassificationHead(nn.Module):
    """
    Classifies treatment outcome into three categories.
    
    Classes:
        0: Harmful (negative outcome)
        1: Beneficial (positive but suboptimal)
        2: Optimal (best therapeutic outcome)
    
    Class distribution (from data): ~22% / ~59% / ~18%
    Uses 2× upweighting of optimal class to address imbalance.
    
    Architecture:
        fusion_dim → 256 → 128 → 3
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: Tuple[int, ...] = (256, 128),
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Fused representation (batch, input_dim)
        Returns:
            Class logits (batch, num_classes)
        """
        return self.network(x)
    
    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities via softmax."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def get_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)


class SafetyHead(nn.Module):
    """
    Predicts multiple safety scores.
    
    Outputs:
        - Toxicity score (0-1, lower is better)
        - Fibrosis risk (0-1, lower is better)
        - Structural integrity (0-1, higher is better)
    
    Safety constraints for RL:
        - Toxicity < 0.15
        - Structural integrity > 0.80
    
    Architecture:
        fusion_dim → 256 → 128 → 3
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: Tuple[int, ...] = (256, 128),
        num_outputs: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_outputs = num_outputs
        self.output_names = ["toxicity", "fibrosis_risk", "structural_integrity"]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_outputs))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Fused representation (batch, input_dim)
        Returns:
            Safety scores (batch, 3) with sigmoid activation
        """
        raw_output = self.network(x)
        return torch.sigmoid(raw_output)  # Bound to [0, 1]
    
    def get_safety_dict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get named safety scores."""
        scores = self.forward(x)
        return {
            name: scores[:, i] for i, name in enumerate(self.output_names)
        }
    
    def check_safety_constraints(
        self,
        x: torch.Tensor,
        toxicity_threshold: float = 0.15,
        integrity_threshold: float = 0.80
    ) -> torch.Tensor:
        """
        Check if designs pass safety constraints.
        
        Returns:
            Boolean tensor (batch,) - True if safe
        """
        scores = self.forward(x)
        toxicity = scores[:, 0]
        integrity = scores[:, 2]
        
        return (toxicity < toxicity_threshold) & (integrity > integrity_threshold)


class MultiTaskHead(nn.Module):
    """
    Combined multi-task prediction head with uncertainty weighting.
    
    Implements uncertainty-weighted multi-task loss (Kendall et al., 2018):
        L_total = Σ (1/(2σ²_i)) * L_i + log(σ_i)
    
    The σ parameters are learned during training, allowing the model
    to automatically balance task difficulty.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Individual task heads
        self.gcs_head = GCSRegressionHead(
            input_dim=input_dim,
            hidden_dims=(256, 128, 64),
            dropout=dropout
        )
        
        self.outcome_head = OutcomeClassificationHead(
            input_dim=input_dim,
            hidden_dims=(256, 128),
            num_classes=3,
            dropout=dropout
        )
        
        self.safety_head = SafetyHead(
            input_dim=input_dim,
            hidden_dims=(256, 128),
            num_outputs=3,
            dropout=dropout
        )
        
        # Learnable log-variance parameters for uncertainty weighting
        # Initialize to 0 (σ² = 1)
        self.log_var_gcs = nn.Parameter(torch.zeros(1))
        self.log_var_outcome = nn.Parameter(torch.zeros(1))
        self.log_var_safety = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all heads.
        
        Args:
            x: Fused representation (batch, input_dim)
        
        Returns:
            Dict containing:
                - gcs_prediction: (batch, 1)
                - outcome_logits: (batch, 3)
                - outcome_probs: (batch, 3)
                - safety_scores: (batch, 3)
                - toxicity: (batch,)
                - fibrosis_risk: (batch,)
                - structural_integrity: (batch,)
        """
        gcs = self.gcs_head(x)
        outcome_logits = self.outcome_head(x)
        outcome_probs = F.softmax(outcome_logits, dim=-1)
        safety = self.safety_head(x)
        
        return {
            "gcs_prediction": gcs.squeeze(-1),
            "outcome_logits": outcome_logits,
            "outcome_probs": outcome_probs,
            "safety_scores": safety,
            "toxicity": safety[:, 0],
            "fibrosis_risk": safety[:, 1],
            "structural_integrity": safety[:, 2],
        }
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        outcome_class_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainty-weighted multi-task loss.
        
        Args:
            predictions: Dict from forward()
            targets: Dict containing:
                - gcs_target: (batch,)
                - outcome_target: (batch,) class indices
                - safety_target: (batch, 3)
            outcome_class_weights: Optional class weights for imbalance
        
        Returns:
            Dict containing:
                - total_loss: Combined loss
                - gcs_loss: GCS regression loss
                - outcome_loss: Classification loss
                - safety_loss: Safety prediction loss
                - uncertainties: Dict of learned σ values
        """
        # GCS Loss: Smooth L1 (Huber)
        gcs_loss = F.smooth_l1_loss(
            predictions["gcs_prediction"],
            targets["gcs_target"],
            beta=1.0
        )
        
        # Outcome Loss: Weighted Cross-Entropy
        if outcome_class_weights is None:
            # Default weights: 1.0, 1.0, 2.0 (upweight optimal class)
            outcome_class_weights = torch.tensor([1.0, 1.0, 2.0])
        outcome_class_weights = outcome_class_weights.to(predictions["outcome_logits"].device)
        
        outcome_loss = F.cross_entropy(
            predictions["outcome_logits"],
            targets["outcome_target"],
            weight=outcome_class_weights
        )
        
        # Safety Loss: MSE for each score
        safety_loss = F.mse_loss(
            predictions["safety_scores"],
            targets["safety_target"]
        )
        
        # Uncertainty-weighted combination
        precision_gcs = torch.exp(-self.log_var_gcs)
        precision_outcome = torch.exp(-self.log_var_outcome)
        precision_safety = torch.exp(-self.log_var_safety)
        
        total_loss = (
            precision_gcs * gcs_loss + self.log_var_gcs +
            precision_outcome * outcome_loss + self.log_var_outcome +
            precision_safety * safety_loss + self.log_var_safety
        )
        
        return {
            "total_loss": total_loss.squeeze(),
            "gcs_loss": gcs_loss,
            "outcome_loss": outcome_loss,
            "safety_loss": safety_loss,
            "uncertainties": {
                "sigma_gcs": torch.exp(0.5 * self.log_var_gcs).item(),
                "sigma_outcome": torch.exp(0.5 * self.log_var_outcome).item(),
                "sigma_safety": torch.exp(0.5 * self.log_var_safety).item(),
            }
        }
    
    def get_reward_for_rl(
        self,
        x: torch.Tensor,
        safety_penalty_weight: float = 10.0
    ) -> torch.Tensor:
        """
        Compute reward signal for RL (Stage 2).
        
        Reward = GCS_improvement + P(optimal) - safety_penalties
        
        Safety penalties:
            - If toxicity > 0.15: -safety_penalty_weight
            - If integrity < 0.80: -safety_penalty_weight
        
        Args:
            x: Fused representation (batch, input_dim)
            safety_penalty_weight: Penalty for safety violations
        
        Returns:
            Reward tensor (batch,)
        """
        with torch.no_grad():
            outputs = self.forward(x)
            
            # Base reward: GCS improvement + optimal probability
            gcs_reward = outputs["gcs_prediction"]
            optimal_prob = outputs["outcome_probs"][:, 2]  # Class 2 = optimal
            
            # Safety penalties
            toxicity = outputs["toxicity"]
            integrity = outputs["structural_integrity"]
            
            toxicity_penalty = (toxicity > 0.15).float() * safety_penalty_weight
            integrity_penalty = (integrity < 0.80).float() * safety_penalty_weight
            
            reward = gcs_reward + optimal_prob - toxicity_penalty - integrity_penalty
            
            return reward
