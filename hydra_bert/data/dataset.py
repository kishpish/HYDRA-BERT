#!/usr/bin/env python3
"""
Dataset for HYDRA-BERT Stage 1 Training

Training data: 1.43M samples
- 565 unique polymer designs
- 10 patient geometries
- Formulation parameter sweeps

Data generated from FEBio finite element simulations.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from transformers import AutoTokenizer


class HydrogelDesignDataset(Dataset):
    """
    Dataset for hydrogel design training.
    
    Each sample contains:
        - SMILES string (tokenized)
        - Formulation parameters (6 values)
        - Patient features (configurable)
        - Targets: GCS, outcome class, safety scores
    
    Args:
        data_path: Path to CSV or parquet file
        tokenizer: HuggingFace tokenizer for SMILES
        max_length: Maximum SMILES sequence length
        patient_feature_cols: List of patient feature column names
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        patient_feature_cols: Optional[List[str]] = None
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        if self.data_path.suffix == '.parquet':
            self.df = pd.read_parquet(data_path)
        else:
            self.df = pd.read_csv(data_path)
        
        # Default patient feature columns
        if patient_feature_cols is None:
            self.patient_feature_cols = [
                "scar_burden_pct",
                "baseline_gcs",
                "baseline_lvef",
                "baseline_edv",
                "baseline_esv",
                "scar_location_anterior",
                "scar_location_lateral", 
                "scar_location_inferior",
                "wall_thickness_mean",
                "lv_mass",
            ]
        else:
            self.patient_feature_cols = patient_feature_cols
        
        # Formulation parameter columns
        self.formulation_cols = [
            "concentration",
            "crosslink_density",
            "degradation_rate",
            "swelling_ratio",
            "biocompatibility",
            "coverage",
        ]
        
        # Validate required columns exist
        self._validate_columns()
        
        # Precompute normalizations
        self._compute_normalizations()
    
    def _validate_columns(self):
        """Check that required columns exist in dataframe."""
        required = ["smiles", "gcs_target", "outcome_class"] + self.formulation_cols
        missing = [c for c in required if c not in self.df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Check patient features (allow missing with warning)
        for col in self.patient_feature_cols:
            if col not in self.df.columns:
                print(f"Warning: Patient feature '{col}' not found, will use zeros")
    
    def _compute_normalizations(self):
        """Compute normalization statistics for features."""
        # Formulation normalization (min-max to [0, 1])
        self.formulation_stats = {}
        for col in self.formulation_cols:
            self.formulation_stats[col] = {
                "min": self.df[col].min(),
                "max": self.df[col].max(),
            }
        
        # Patient feature normalization (z-score)
        self.patient_stats = {}
        for col in self.patient_feature_cols:
            if col in self.df.columns:
                self.patient_stats[col] = {
                    "mean": self.df[col].mean(),
                    "std": self.df[col].std() + 1e-8,
                }
    
    def _normalize_formulation(self, row: pd.Series) -> np.ndarray:
        """Normalize formulation parameters to [0, 1]."""
        values = []
        for col in self.formulation_cols:
            val = row[col]
            stats = self.formulation_stats[col]
            normalized = (val - stats["min"]) / (stats["max"] - stats["min"] + 1e-8)
            values.append(normalized)
        return np.array(values, dtype=np.float32)
    
    def _normalize_patient(self, row: pd.Series) -> np.ndarray:
        """Normalize patient features (z-score)."""
        values = []
        for col in self.patient_feature_cols:
            if col in self.df.columns:
                val = row[col]
                stats = self.patient_stats[col]
                normalized = (val - stats["mean"]) / stats["std"]
            else:
                normalized = 0.0
            values.append(normalized)
        return np.array(values, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns dict with:
            - input_ids: Tokenized SMILES
            - attention_mask: Attention mask
            - formulation_params: (6,) normalized
            - patient_features: (patient_dim,) normalized
            - gcs_target: scalar
            - outcome_target: class index (0, 1, or 2)
            - safety_target: (3,) toxicity, fibrosis, integrity
        """
        row = self.df.iloc[idx]
        
        # Tokenize SMILES
        smiles = row["smiles"]
        encoded = self.tokenizer(
            smiles,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Get formulation and patient features
        formulation = self._normalize_formulation(row)
        patient = self._normalize_patient(row)
        
        # Get targets
        gcs_target = float(row["gcs_target"])
        outcome_target = int(row["outcome_class"])
        
        # Safety targets (if available, otherwise use defaults)
        if "toxicity" in self.df.columns:
            safety_target = np.array([
                row["toxicity"],
                row.get("fibrosis_risk", 0.1),
                row.get("structural_integrity", 0.9),
            ], dtype=np.float32)
        else:
            # Generate plausible defaults based on outcome
            if outcome_target == 2:  # Optimal
                safety_target = np.array([0.05, 0.08, 0.92], dtype=np.float32)
            elif outcome_target == 1:  # Beneficial
                safety_target = np.array([0.10, 0.15, 0.85], dtype=np.float32)
            else:  # Harmful
                safety_target = np.array([0.25, 0.30, 0.65], dtype=np.float32)
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "formulation_params": torch.tensor(formulation, dtype=torch.float32),
            "patient_features": torch.tensor(patient, dtype=torch.float32),
            "gcs_target": torch.tensor(gcs_target, dtype=torch.float32),
            "outcome_target": torch.tensor(outcome_target, dtype=torch.long),
            "safety_target": torch.tensor(safety_target, dtype=torch.float32),
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for outcome classification.
        
        Returns weights inversely proportional to class frequency,
        with 2× upweighting of optimal class.
        """
        counts = self.df["outcome_class"].value_counts().sort_index()
        total = counts.sum()
        
        weights = total / (len(counts) * counts.values)
        
        # 2× upweight optimal class (class 2)
        weights[2] *= 2.0
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_patient_ids(self) -> List[str]:
        """Get unique patient IDs in dataset."""
        if "patient_id" in self.df.columns:
            return self.df["patient_id"].unique().tolist()
        return []


class DataCollator:
    """
    Collate function for DataLoader.
    
    Handles batching of variable-length sequences with padding.
    """
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples."""
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "formulation_params": torch.stack([x["formulation_params"] for x in batch]),
            "patient_features": torch.stack([x["patient_features"] for x in batch]),
            "gcs_target": torch.stack([x["gcs_target"] for x in batch]),
            "outcome_target": torch.stack([x["outcome_target"] for x in batch]),
            "safety_target": torch.stack([x["safety_target"] for x in batch]),
        }


def create_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int = 256,
    num_workers: int = 4,
    max_length: int = 128
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_length: Max SMILES sequence length
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = HydrogelDesignDataset(train_path, tokenizer, max_length)
    val_dataset = HydrogelDesignDataset(val_path, tokenizer, max_length)
    
    collator = DataCollator()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    return train_loader, val_loader
