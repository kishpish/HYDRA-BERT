#!/usr/bin/env python3
"""
HYDRA-BERT: HYdrogel Domain-Refined Adaptation of BERT

Complete model architecture combining:
- polyBERT with LoRA (~86M frozen + 1.2M trainable)
- Cross-modal fusion transformer (4 layers, 8 heads)
- Multi-task prediction heads

Total: ~115M parameters

Usage:
    model = HydraBERT()
    
    # Stage 1: Supervised training
    outputs = model(smiles, formulation_params, patient_features)
    loss = model.compute_loss(outputs, targets)
    
    # Stage 2: RL reward function
    reward = model.get_reward(smiles, formulation_params, patient_features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .polybert_lora import PolyBERTWithLoRA
from .fusion_transformer import FullFusionModule
from .prediction_heads import MultiTaskHead


@dataclass
class HydraBERTConfig:
    """Configuration for HYDRA-BERT model."""
    
    # polyBERT configuration
    polybert_model: str = "kuelumbus/polyBERT"
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    freeze_polybert: bool = True
    
    # Formulation encoder
    formulation_input_dim: int = 6
    formulation_hidden_dim: int = 128
    formulation_output_dim: int = 256
    
    # Patient encoder
    patient_input_dim: int = 10
    patient_hidden_dim: int = 64
    patient_output_dim: int = 128
    
    # Fusion transformer
    fusion_dim: int = 512
    num_fusion_layers: int = 4
    num_fusion_heads: int = 8
    fusion_mlp_ratio: float = 4.0
    
    # General
    dropout: float = 0.1
    max_seq_length: int = 128
    
    # Training
    outcome_class_weights: Tuple[float, ...] = (1.0, 1.0, 2.0)


class HydraBERT(nn.Module):
    """
    HYDRA-BERT: Complete model for hydrogel design optimization.
    
    Architecture:
        1. polyBERT (frozen) + LoRA: SMILES → 600-dim embedding
        2. Formulation MLP: 6 params → 256-dim encoding
        3. Patient MLP: features → 128-dim encoding
        4. Cross-Modal Fusion: 4-layer transformer → 512-dim fused
        5. Multi-Task Heads:
           - GCS regression
           - Outcome classification (3 classes)
           - Safety scores (3 values)
    
    Parameters:
        - polyBERT base: ~86M (frozen)
        - LoRA adapters: ~1.2M (trainable)
        - Fusion + Heads: ~28M (trainable)
        - Total: ~115M
    
    Args:
        config: HydraBERTConfig with all hyperparameters
    """
    
    def __init__(self, config: Optional[HydraBERTConfig] = None):
        super().__init__()
        
        if config is None:
            config = HydraBERTConfig()
        self.config = config
        
        # 1. polyBERT with LoRA
        self.polybert = PolyBERTWithLoRA(
            model_name=config.polybert_model,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            freeze_base=config.freeze_polybert
        )
        
        # Get actual hidden size from polyBERT
        polymer_dim = self.polybert.hidden_size
        
        # 2 & 3. Fusion module (includes formulation and patient encoders)
        self.fusion_module = FullFusionModule(
            polymer_dim=polymer_dim,
            formulation_input_dim=config.formulation_input_dim,
            formulation_hidden_dim=config.formulation_hidden_dim,
            formulation_output_dim=config.formulation_output_dim,
            patient_input_dim=config.patient_input_dim,
            patient_hidden_dim=config.patient_hidden_dim,
            patient_output_dim=config.patient_output_dim,
            fusion_dim=config.fusion_dim,
            num_fusion_layers=config.num_fusion_layers,
            num_fusion_heads=config.num_fusion_heads,
            dropout=config.dropout
        )
        
        # 4. Multi-task prediction heads
        self.task_heads = MultiTaskHead(
            input_dim=config.fusion_dim,
            dropout=config.dropout
        )
        
        # Store class weights for loss computation
        self.register_buffer(
            "outcome_class_weights",
            torch.tensor(config.outcome_class_weights)
        )
        
        # Count parameters
        self._count_parameters()
    
    def _count_parameters(self):
        """Count and store parameter statistics."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        self.param_counts = {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "polybert_total": sum(p.numel() for p in self.polybert.parameters()),
            "polybert_trainable": sum(
                p.numel() for p in self.polybert.parameters() if p.requires_grad
            ),
            "fusion_trainable": sum(
                p.numel() for p in self.fusion_module.parameters() if p.requires_grad
            ),
            "heads_trainable": sum(
                p.numel() for p in self.task_heads.parameters() if p.requires_grad
            ),
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        formulation_params: torch.Tensor,
        patient_features: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            input_ids: Tokenized SMILES (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            formulation_params: 6 formulation parameters (batch, 6)
            patient_features: Patient features (batch, patient_dim)
            return_attention: Whether to return attention maps
        
        Returns:
            Dict containing:
                - gcs_prediction: GCS improvement prediction
                - outcome_logits: 3-class logits
                - outcome_probs: 3-class probabilities
                - safety_scores: 3 safety scores
                - toxicity, fibrosis_risk, structural_integrity
                - polymer_embedding: Raw polyBERT embedding
                - fused_representation: Final fused representation
        """
        # 1. Encode SMILES with polyBERT
        polybert_output = self.polybert(input_ids, attention_mask)
        polymer_embedding = polybert_output["embedding"]
        
        # 2. Fuse all modalities
        fusion_output = self.fusion_module(
            polymer_embedding=polymer_embedding,
            formulation_params=formulation_params,
            patient_features=patient_features,
            return_attention=return_attention
        )
        
        # 3. Get predictions from all heads
        fused_repr = fusion_output["fused_representation"]
        predictions = self.task_heads(fused_repr)
        
        # Add embeddings to output
        predictions["polymer_embedding"] = polymer_embedding
        predictions["fused_representation"] = fused_repr
        predictions["formulation_encoding"] = fusion_output["formulation_encoding"]
        predictions["patient_encoding"] = fusion_output["patient_encoding"]
        
        if return_attention:
            predictions["attention_weights"] = fusion_output.get("attention_weights")
        
        return predictions
    
    def forward_from_smiles(
        self,
        smiles_list: List[str],
        formulation_params: torch.Tensor,
        patient_features: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass directly from SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            formulation_params: (batch, 6)
            patient_features: (batch, patient_dim)
        
        Returns:
            Same as forward()
        """
        device = formulation_params.device
        
        # Tokenize SMILES
        encoded = self.polybert.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            formulation_params=formulation_params,
            patient_features=patient_features,
            return_attention=return_attention
        )
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss with uncertainty weighting.
        
        Args:
            predictions: Output from forward()
            targets: Dict containing:
                - gcs_target: (batch,)
                - outcome_target: (batch,) class indices
                - safety_target: (batch, 3)
        
        Returns:
            Dict with total_loss and individual losses
        """
        return self.task_heads.compute_loss(
            predictions=predictions,
            targets=targets,
            outcome_class_weights=self.outcome_class_weights
        )
    
    def get_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        formulation_params: torch.Tensor,
        patient_features: torch.Tensor,
        safety_penalty_weight: float = 10.0
    ) -> torch.Tensor:
        """
        Compute reward for RL (Stage 2).
        
        This function is called by the PPO agent with the Stage 1 model frozen.
        
        Reward = GCS_improvement + P(optimal) - safety_penalties
        
        Args:
            input_ids, attention_mask: Tokenized SMILES
            formulation_params: (batch, 6)
            patient_features: (batch, patient_dim)
            safety_penalty_weight: Penalty for safety violations
        
        Returns:
            Reward tensor (batch,)
        """
        with torch.no_grad():
            predictions = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                formulation_params=formulation_params,
                patient_features=patient_features
            )
            
            # Compute reward
            gcs_reward = predictions["gcs_prediction"]
            optimal_prob = predictions["outcome_probs"][:, 2]
            
            # Safety penalties
            toxicity_penalty = (predictions["toxicity"] > 0.15).float() * safety_penalty_weight
            integrity_penalty = (predictions["structural_integrity"] < 0.80).float() * safety_penalty_weight
            
            reward = gcs_reward + optimal_prob - toxicity_penalty - integrity_penalty
            
            return reward
    
    def freeze_for_rl(self):
        """Freeze all parameters for use as RL reward function."""
        for param in self.parameters():
            param.requires_grad = False
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get only trainable parameters (for optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def save_checkpoint(self, path: str, optimizer=None, epoch=None, metrics=None):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "param_counts": self.param_counts,
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            checkpoint["epoch"] = epoch
        if metrics is not None:
            checkpoint["metrics"] = metrics
        
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cuda"):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        config = checkpoint.get("config", HydraBERTConfig())
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        
        return model, checkpoint
    
    def __repr__(self) -> str:
        return (
            f"HydraBERT(\n"
            f"  polybert={self.config.polybert_model},\n"
            f"  lora_rank={self.config.lora_rank},\n"
            f"  fusion_layers={self.config.num_fusion_layers},\n"
            f"  fusion_heads={self.config.num_fusion_heads},\n"
            f"  fusion_dim={self.config.fusion_dim},\n"
            f"  total_params={self.param_counts['total']:,},\n"
            f"  trainable_params={self.param_counts['trainable']:,},\n"
            f"  frozen_params={self.param_counts['frozen']:,}\n"
            f")"
        )
