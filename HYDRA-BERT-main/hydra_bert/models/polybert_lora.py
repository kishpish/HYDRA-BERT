#!/usr/bin/env python3
"""
PolyBERT with Low-Rank Adaptation (LoRA)

Foundation model: polyBERT (DeBERTa-v2, 86M parameters)
- Pre-trained on 100 million polymer SMILES strings
- Produces 600-dimensional chemical fingerprints
- 12 encoder layers, 12 attention heads

LoRA Configuration:
- Rank: 16
- Applied to: attention query and value projections
- Trainable parameters: ~1.2M
- Base encoder: FROZEN (preserves chemistry knowledge)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from transformers import AutoModel, AutoTokenizer, AutoConfig


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer.
    
    Implements: W' = W + BA where B ∈ R^{d×r}, A ∈ R^{r×k}
    Only B and A are trainable, original W is frozen.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank (default: 16)
        alpha: LoRA scaling factor (default: 32)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank decomposition matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize A with Kaiming, B with zeros (start at pretrained weights)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute LoRA delta: scaling * (x @ A^T @ B^T)
        """
        # x: (batch, seq, in_features)
        # lora_A: (rank, in_features) -> A^T: (in_features, rank)
        # lora_B: (out_features, rank) -> B^T: (rank, out_features)
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return lora_output * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    Freezes the original linear weights and adds trainable low-rank matrices.
    """
    
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.original_linear = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # Freeze original weights
        for param in self.original_linear.parameters():
            param.requires_grad = False
        
        # Add LoRA adaptation
        self.lora = LoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: original output + LoRA delta."""
        return self.original_linear(x) + self.lora(x)
    
    @property
    def weight(self) -> torch.Tensor:
        """Return effective weight matrix (for compatibility)."""
        return self.original_linear.weight + (
            self.lora.lora_B @ self.lora.lora_A * self.lora.scaling
        )


class PolyBERTWithLoRA(nn.Module):
    """
    polyBERT encoder with Low-Rank Adaptation.
    
    Architecture:
        - Base: DeBERTa-v2 (86M parameters, 12 layers, 12 heads)
        - Pre-training: 100M polymer SMILES strings
        - Output: 600-dimensional chemical fingerprints
        - LoRA: Applied to Q and V projections in attention (rank=16)
        - Trainable: ~1.2M parameters (base encoder frozen)
    
    Args:
        model_name: HuggingFace model name (default: "kuelumbus/polyBERT")
        lora_rank: LoRA rank (default: 16)
        lora_alpha: LoRA scaling factor (default: 32)
        lora_dropout: LoRA dropout rate (default: 0.1)
        freeze_base: Whether to freeze base encoder (default: True)
    """
    
    def __init__(
        self,
        model_name: str = "kuelumbus/polyBERT",
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.1,
        freeze_base: bool = True
    ):
        super().__init__()
        
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.freeze_base = freeze_base
        
        # Load polyBERT (DeBERTa-v2 architecture)
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get hidden size (should be 600 for polyBERT, but check)
        self.hidden_size = self.config.hidden_size
        
        # Freeze base encoder if specified
        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Apply LoRA to attention layers (Q and V projections)
        self._apply_lora_to_attention(lora_rank, lora_alpha, lora_dropout)
        
        # Count parameters
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def _apply_lora_to_attention(
        self,
        rank: int,
        alpha: float,
        dropout: float
    ):
        """
        Apply LoRA to query and value projections in all attention layers.
        """
        self.lora_layers = nn.ModuleDict()
        
        # DeBERTa-v2 has encoder.layer[i].attention.self.{query,key,value}_proj
        for layer_idx, layer in enumerate(self.encoder.encoder.layer):
            attention = layer.attention.self
            
            # Apply LoRA to query projection
            if hasattr(attention, 'query_proj'):
                original_q = attention.query_proj
                lora_q = LoRALinear(original_q, rank=rank, alpha=alpha, dropout=dropout)
                attention.query_proj = lora_q
                self.lora_layers[f"layer_{layer_idx}_query"] = lora_q.lora
            
            # Apply LoRA to value projection
            if hasattr(attention, 'value_proj'):
                original_v = attention.value_proj
                lora_v = LoRALinear(original_v, rank=rank, alpha=alpha, dropout=dropout)
                attention.value_proj = lora_v
                self.lora_layers[f"layer_{layer_idx}_value"] = lora_v.lora
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through polyBERT with LoRA.
        
        Args:
            input_ids: Tokenized SMILES (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            return_all_hidden_states: Whether to return all layer outputs
        
        Returns:
            Dict containing:
                - embedding: 600-dim polymer fingerprint (batch, hidden_size)
                - last_hidden_state: Full sequence output (batch, seq_len, hidden_size)
                - all_hidden_states: Optional list of all layer outputs
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_all_hidden_states
        )
        
        last_hidden_state = outputs.last_hidden_state
        
        # Use [CLS] token embedding as polymer fingerprint
        # For DeBERTa, this is the first token
        cls_embedding = last_hidden_state[:, 0, :]  # (batch, hidden_size)
        
        result = {
            "embedding": cls_embedding,
            "last_hidden_state": last_hidden_state,
        }
        
        if return_all_hidden_states and hasattr(outputs, 'hidden_states'):
            result["all_hidden_states"] = outputs.hidden_states
        
        return result
    
    def encode_smiles(
        self,
        smiles_list: list,
        max_length: int = 128,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Encode a list of SMILES strings to embeddings.
        
        Args:
            smiles_list: List of SMILES strings
            max_length: Maximum sequence length
            device: Target device
        
        Returns:
            Tensor of shape (batch, hidden_size)
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Tokenize
        encoded = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        # Forward pass
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.forward(input_ids, attention_mask)
        
        return outputs["embedding"]
    
    def get_lora_parameters(self) -> list:
        """Get only LoRA parameters for optimizer."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def save_lora_weights(self, path: str):
        """Save only LoRA weights."""
        lora_state_dict = {
            name: param for name, param in self.state_dict().items()
            if "lora" in name.lower()
        }
        torch.save(lora_state_dict, path)
    
    def load_lora_weights(self, path: str):
        """Load LoRA weights."""
        lora_state_dict = torch.load(path, map_location="cpu")
        self.load_state_dict(lora_state_dict, strict=False)
    
    def __repr__(self) -> str:
        return (
            f"PolyBERTWithLoRA(\n"
            f"  base_model={self.model_name},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  lora_rank={self.lora_rank},\n"
            f"  lora_alpha={self.lora_alpha},\n"
            f"  total_params={self.total_params:,},\n"
            f"  trainable_params={self.trainable_params:,},\n"
            f"  frozen_base={self.freeze_base}\n"
            f")"
        )
