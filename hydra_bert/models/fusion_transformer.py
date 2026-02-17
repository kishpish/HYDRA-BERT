#!/usr/bin/env python3
"""
Cross-Modal Fusion Transformer

Fuses three input streams into a unified representation:
1. Polymer stream: SMILES → polyBERT → 600-dim embedding
2. Formulation stream: 6 parameters → MLP → 256-dim encoding
3. Patient stream: Patient features → encoded representation

Architecture:
- Concatenation: 600 + 256 + variable = 856+ dimensional fused input
- 4 transformer layers with 8 attention heads
- Learned aggregation token for final representation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class FormulationEncoder(nn.Module):
    """
    Encodes 6 formulation parameters into 256-dim representation.
    
    Parameters:
        1. Concentration (%)
        2. Crosslink density
        3. Degradation rate (days^-1)
        4. Swelling ratio
        5. Biocompatibility score (0-1)
        6. Coverage (scar_only=0, scar_bz25=0.25, scar_bz50=0.5, scar_bz100=1.0)
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Formulation parameters (batch, 6)
        Returns:
            Encoded formulation (batch, 256)
        """
        return self.encoder(x)


class PatientEncoder(nn.Module):
    """
    Encodes patient-specific features.
    
    Core features:
        - Scar burden (% of LV)
        - Baseline circumferential strain (GCS, %)
        - Additional cardiac metrics as available
    """
    
    def __init__(
        self,
        input_dim: int = 10,  # Configurable patient feature count
        hidden_dim: int = 64,
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patient features (batch, input_dim)
        Returns:
            Encoded patient representation (batch, output_dim)
        """
        return self.encoder(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with pre-norm."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            attention_mask: Optional mask (batch, seq_len)
        Returns:
            output: Attended tensor (batch, seq_len, embed_dim)
            attn_weights: Attention weights (batch, num_heads, seq_len, seq_len)
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        
        if attention_mask is not None:
            # Expand mask for heads: (B, 1, 1, N)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x, attn


class FusionTransformerBlock(nn.Module):
    """
    Single transformer block for cross-modal fusion.
    
    Pre-norm architecture:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input (batch, seq_len, embed_dim)
            attention_mask: Optional mask
        Returns:
            output: Transformed tensor
            attn_weights: Attention weights
        """
        # Self-attention with residual
        attn_out, attn_weights = self.attn(self.norm1(x), attention_mask)
        x = x + attn_out
        
        # FFN with residual
        x = x + self.mlp(self.norm2(x))
        
        return x, attn_weights


class CrossModalFusionTransformer(nn.Module):
    """
    Cross-Modal Fusion Transformer for HYDRA-BERT.
    
    Fuses three input streams:
        1. Polymer: 600-dim from polyBERT
        2. Formulation: 256-dim from MLP encoder
        3. Patient: Variable dim from patient encoder
    
    Architecture:
        - Input projection to common dimension
        - Learned [AGG] aggregation token prepended
        - 4 transformer layers with 8 attention heads
        - Final output from [AGG] token
    
    Args:
        polymer_dim: Polymer embedding dimension (600)
        formulation_dim: Formulation encoding dimension (256)
        patient_dim: Patient encoding dimension (128)
        fusion_dim: Internal fusion dimension (512)
        num_layers: Number of transformer layers (4)
        num_heads: Number of attention heads (8)
        dropout: Dropout rate (0.1)
    """
    
    def __init__(
        self,
        polymer_dim: int = 600,
        formulation_dim: int = 256,
        patient_dim: int = 128,
        fusion_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.polymer_dim = polymer_dim
        self.formulation_dim = formulation_dim
        self.patient_dim = patient_dim
        self.fusion_dim = fusion_dim
        self.num_layers = num_layers
        
        # Input projections to common dimension
        self.polymer_proj = nn.Linear(polymer_dim, fusion_dim)
        self.formulation_proj = nn.Linear(formulation_dim, fusion_dim)
        self.patient_proj = nn.Linear(patient_dim, fusion_dim)
        
        # Learned aggregation token
        self.agg_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        nn.init.trunc_normal_(self.agg_token, std=0.02)
        
        # Positional embeddings for each modality + agg token
        # Positions: [AGG, Polymer, Formulation, Patient] = 4 positions
        self.pos_embed = nn.Parameter(torch.zeros(1, 4, fusion_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            FusionTransformerBlock(
                embed_dim=fusion_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(fusion_dim)
        
        # Store attention weights for interpretability
        self.attention_weights = []
    
    def forward(
        self,
        polymer_embedding: torch.Tensor,
        formulation_encoding: torch.Tensor,
        patient_encoding: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through fusion transformer.
        
        Args:
            polymer_embedding: (batch, 600) from polyBERT
            formulation_encoding: (batch, 256) from FormulationEncoder
            patient_encoding: (batch, 128) from PatientEncoder
            return_attention: Whether to return attention maps
        
        Returns:
            Dict containing:
                - fused_representation: (batch, fusion_dim) final fused output
                - polymer_attended: (batch, fusion_dim) polymer after attention
                - formulation_attended: (batch, fusion_dim) formulation after attention
                - patient_attended: (batch, fusion_dim) patient after attention
                - attention_weights: Optional list of attention maps
        """
        B = polymer_embedding.shape[0]
        
        # Project each modality to fusion dimension
        polymer_tokens = self.polymer_proj(polymer_embedding).unsqueeze(1)  # (B, 1, fusion_dim)
        formulation_tokens = self.formulation_proj(formulation_encoding).unsqueeze(1)
        patient_tokens = self.patient_proj(patient_encoding).unsqueeze(1)
        
        # Expand aggregation token for batch
        agg_tokens = self.agg_token.expand(B, -1, -1)  # (B, 1, fusion_dim)
        
        # Concatenate: [AGG, Polymer, Formulation, Patient]
        x = torch.cat([agg_tokens, polymer_tokens, formulation_tokens, patient_tokens], dim=1)
        # x shape: (B, 4, fusion_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Pass through transformer layers
        self.attention_weights = []
        for layer in self.layers:
            x, attn = layer(x)
            if return_attention:
                self.attention_weights.append(attn)
        
        # Apply final norm
        x = self.norm(x)
        
        # Extract outputs
        result = {
            "fused_representation": x[:, 0, :],  # AGG token
            "polymer_attended": x[:, 1, :],
            "formulation_attended": x[:, 2, :],
            "patient_attended": x[:, 3, :],
        }
        
        if return_attention:
            result["attention_weights"] = self.attention_weights
        
        return result


class FullFusionModule(nn.Module):
    """
    Complete fusion module including all encoders.
    
    This is the main interface for Stage 1 training.
    
    Input dimensions:
        - SMILES embedding: 600-dim (from polyBERT)
        - Formulation params: 6 values
        - Patient features: Configurable
    
    Output:
        - Fused representation: 512-dim
    """
    
    def __init__(
        self,
        polymer_dim: int = 600,
        formulation_input_dim: int = 6,
        formulation_hidden_dim: int = 128,
        formulation_output_dim: int = 256,
        patient_input_dim: int = 10,
        patient_hidden_dim: int = 64,
        patient_output_dim: int = 128,
        fusion_dim: int = 512,
        num_fusion_layers: int = 4,
        num_fusion_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Encoders for each modality
        self.formulation_encoder = FormulationEncoder(
            input_dim=formulation_input_dim,
            hidden_dim=formulation_hidden_dim,
            output_dim=formulation_output_dim,
            dropout=dropout
        )
        
        self.patient_encoder = PatientEncoder(
            input_dim=patient_input_dim,
            hidden_dim=patient_hidden_dim,
            output_dim=patient_output_dim,
            dropout=dropout
        )
        
        # Cross-modal fusion transformer
        self.fusion_transformer = CrossModalFusionTransformer(
            polymer_dim=polymer_dim,
            formulation_dim=formulation_output_dim,
            patient_dim=patient_output_dim,
            fusion_dim=fusion_dim,
            num_layers=num_fusion_layers,
            num_heads=num_fusion_heads,
            dropout=dropout
        )
        
        self.output_dim = fusion_dim
    
    def forward(
        self,
        polymer_embedding: torch.Tensor,
        formulation_params: torch.Tensor,
        patient_features: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            polymer_embedding: (batch, 600) from polyBERT
            formulation_params: (batch, 6) raw formulation parameters
            patient_features: (batch, patient_dim) patient features
            return_attention: Whether to return attention maps
        
        Returns:
            Dict with fused_representation and component outputs
        """
        # Encode formulation and patient
        formulation_encoding = self.formulation_encoder(formulation_params)
        patient_encoding = self.patient_encoder(patient_features)
        
        # Fuse all modalities
        fusion_output = self.fusion_transformer(
            polymer_embedding=polymer_embedding,
            formulation_encoding=formulation_encoding,
            patient_encoding=patient_encoding,
            return_attention=return_attention
        )
        
        # Add intermediate encodings to output
        fusion_output["formulation_encoding"] = formulation_encoding
        fusion_output["patient_encoding"] = patient_encoding
        
        return fusion_output
