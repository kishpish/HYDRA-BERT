#!/usr/bin/env python3
"""
HYDRA-BERT Stage 1: Supervised Multi-Task Learning

Trains the HYDRA-BERT model on cardiac hydrogel outcome prediction using
supervised learning with multiple prediction heads.

Training Tasks:
1. Primary Regression: delta_EF_pct (ejection fraction improvement)
2. Primary Classification: is_optimal (binary: meets therapeutic criteria)
3. Auxiliary Regression: delta_BZ_stress_reduction_pct, strain_normalization_pct

Model Architecture:

- polyBERT encoder (frozen initially) → 600-dim SMILES embedding
- SMILES adapter: 600 → 384 → 256
- Property encoder: 19 features → 64 → 128
- Category embedding: 8 categories → 16-dim
- Fusion layer: 400 → 512 → 512
- Multiple prediction heads

Training Strategy:

- Phase 1 (epochs 1-15): polyBERT frozen, train adapters + heads
- Phase 2 (epochs 16-25): unfreeze polyBERT with differential LR

Usage:

    # Single GPU training
    python train_supervised.py

    # Multi-GPU distributed training
    accelerate launch --config_file configs/accelerate_config.yaml train_supervised.py

    # Custom configuration
    python train_supervised.py --config configs/custom_config.yaml

Output:

    checkpoints/stage1/
    ├── best_model.pt          # Best model by validation loss
    ├── final_model.pt         # Final epoch model
    ├── training_history.json  # Loss/metric curves
    └── config.yaml            # Training configuration

Author: Krishiv Potluri
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
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from hydra_bert_v2.models import HydraBERT
from hydra_bert_v2.data import CardiacHydrogelDataset, create_dataloaders
from hydra_bert_v2.losses import HydraMultiTaskLoss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# DEFAULT CONFIGURATION

DEFAULT_CONFIG = {
    "model": {
        "polybert_path": "kuelumbus/polyBERT",
        "freeze_polybert": True,
        "dropout": 0.1,
        "num_numerical_features": 19,
        "num_categories": 8,
    },
    "training": {
        "epochs": 15,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "warmup_steps": 1000,
        "gradient_clip": 1.0,
        "mixed_precision": True,
    },
    "loss": {
        "regression_weight": 1.0,
        "classification_weight": 0.5,
        "auxiliary_weight": 0.3,
        "focal_gamma": 2.0,
        "pos_weight": 3.17,  # For class imbalance (24% positive)
    },
    "data": {
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
        "num_workers": 4,
        "pin_memory": True,
    },
    "paths": {
        "data_path": "data/processed/POLYBERT_TRAINING_FINAL.csv",
        "output_dir": "checkpoints/stage1",
        "log_dir": "logs/stage1",
    },
    "seed": 42,
}


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    gradient_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""

    model.train()
    total_loss = 0.0
    loss_components = {"ef": 0.0, "optimal": 0.0, "stress": 0.0}
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        # Move to device
        smiles_tokens = batch["smiles_tokens"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        numerical_features = batch["numerical_features"].to(device)
        category_ids = batch["category_ids"].to(device)
        targets = {k: v.to(device) for k, v in batch["targets"].items()}

        optimizer.zero_grad()

        # Mixed precision forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    smiles_tokens=smiles_tokens,
                    attention_mask=attention_mask,
                    numerical_features=numerical_features,
                    category_ids=category_ids,
                )
                loss, components = criterion(
                    pred_ef=outputs["delta_ef"],
                    pred_optimal=outputs["is_optimal"],
                    pred_stress=outputs.get("stress_reduction"),
                    true_ef=targets["delta_ef"],
                    true_optimal=targets["is_optimal"],
                    true_stress=targets.get("stress_reduction"),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                smiles_tokens=smiles_tokens,
                attention_mask=attention_mask,
                numerical_features=numerical_features,
                category_ids=category_ids,
            )
            loss, components = criterion(
                pred_ef=outputs["delta_ef"],
                pred_optimal=outputs["is_optimal"],
                pred_stress=outputs.get("stress_reduction"),
                true_ef=targets["delta_ef"],
                true_optimal=targets["is_optimal"],
                true_stress=targets.get("stress_reduction"),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        total_loss += loss.item()
        for key in components:
            loss_components[key] += components[key].item()
        num_batches += 1

        progress_bar.set_postfix({"loss": loss.item()})

    return {
        "loss": total_loss / num_batches,
        "loss_ef": loss_components["ef"] / num_batches,
        "loss_optimal": loss_components["optimal"] / num_batches,
        "loss_stress": loss_components["stress"] / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate the model."""

    model.eval()
    total_loss = 0.0
    all_pred_ef = []
    all_true_ef = []
    all_pred_optimal = []
    all_true_optimal = []
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        smiles_tokens = batch["smiles_tokens"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        numerical_features = batch["numerical_features"].to(device)
        category_ids = batch["category_ids"].to(device)
        targets = {k: v.to(device) for k, v in batch["targets"].items()}

        outputs = model(
            smiles_tokens=smiles_tokens,
            attention_mask=attention_mask,
            numerical_features=numerical_features,
            category_ids=category_ids,
        )

        loss, _ = criterion(
            pred_ef=outputs["delta_ef"],
            pred_optimal=outputs["is_optimal"],
            pred_stress=outputs.get("stress_reduction"),
            true_ef=targets["delta_ef"],
            true_optimal=targets["is_optimal"],
            true_stress=targets.get("stress_reduction"),
        )

        total_loss += loss.item()
        num_batches += 1

        all_pred_ef.extend(outputs["delta_ef"].cpu().numpy())
        all_true_ef.extend(targets["delta_ef"].cpu().numpy())
        all_pred_optimal.extend(
            torch.sigmoid(outputs["is_optimal"]).cpu().numpy()
        )
        all_true_optimal.extend(targets["is_optimal"].cpu().numpy())

    # Calculate metrics
    pred_ef = np.array(all_pred_ef)
    true_ef = np.array(all_true_ef)
    pred_optimal = np.array(all_pred_optimal)
    true_optimal = np.array(all_true_optimal)

    mae = np.mean(np.abs(pred_ef - true_ef))
    rmse = np.sqrt(np.mean((pred_ef - true_ef) ** 2))

    # R² calculation
    ss_res = np.sum((true_ef - pred_ef) ** 2)
    ss_tot = np.sum((true_ef - np.mean(true_ef)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Classification metrics
    pred_binary = (pred_optimal >= 0.5).astype(int)
    accuracy = np.mean(pred_binary == true_optimal)

    # AUROC
    from sklearn.metrics import roc_auc_score, f1_score
    auroc = roc_auc_score(true_optimal, pred_optimal)
    f1 = f1_score(true_optimal, pred_binary)

    return {
        "val_loss": total_loss / num_batches,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "accuracy": accuracy,
        "auroc": auroc,
        "f1": f1,
    }


def main():
    """Main training function."""

    parser = argparse.ArgumentParser(description="HYDRA-BERT Stage 1 Training")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load configuration
    config = DEFAULT_CONFIG.copy()
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            user_config = yaml.safe_load(f)
        # Merge configs
        for key in user_config:
            if isinstance(user_config[key], dict):
                config[key].update(user_config[key])
            else:
                config[key] = user_config[key]

    # Set seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directories
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Create dataloaders
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=config["paths"]["data_path"],
        batch_size=config["training"]["batch_size"],
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        seed=config["seed"],
    )
    logger.info(f"Train: {len(train_loader.dataset)}, "
               f"Val: {len(val_loader.dataset)}, "
               f"Test: {len(test_loader.dataset)}")

    # Create model
    logger.info("Creating model...")
    model = HydraBERT(
        polybert_path=config["model"]["polybert_path"],
        freeze_polybert=config["model"]["freeze_polybert"],
        dropout=config["model"]["dropout"],
    )
    model.to(device)

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(train_loader),
        T_mult=2,
    )

    # Create loss function
    criterion = HydraMultiTaskLoss(
        regression_weight=config["loss"]["regression_weight"],
        classification_weight=config["loss"]["classification_weight"],
        auxiliary_weight=config["loss"]["auxiliary_weight"],
        focal_gamma=config["loss"]["focal_gamma"],
        pos_weight=config["loss"]["pos_weight"],
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config["training"]["mixed_precision"] else None

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float("inf")
    history = {"train": [], "val": []}

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]
        history = checkpoint.get("history", history)
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    logger.info("Starting training...")
    print("HYDRA-BERT STAGE 1: SUPERVISED MULTI-TASK LEARNING")

    for epoch in range(start_epoch, config["training"]["epochs"]):
        logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler,
            gradient_clip=config["training"]["gradient_clip"],
        )
        history["train"].append(train_metrics)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        history["val"].append(val_metrics)

        # Update scheduler
        scheduler.step()

        # Log metrics
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"  MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}")
        logger.info(f"  AUROC: {val_metrics['auroc']:.4f}, F1: {val_metrics['f1']:.4f}")

        # Save best model
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": config,
            }, output_dir / "best_model.pt")
            logger.info(f"  Saved best model (val_loss: {best_val_loss:.4f})")

    # Save final model
    torch.save({
        "epoch": config["training"]["epochs"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": config,
        "history": history,
    }, output_dir / "final_model.pt")

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_metrics = validate(model, test_loader, criterion, device)
    logger.info(f"Test Results:")
    logger.info(f"  MAE: {test_metrics['mae']:.4f}")
    logger.info(f"  R²: {test_metrics['r2']:.4f}")
    logger.info(f"  AUROC: {test_metrics['auroc']:.4f}")
    logger.info(f"  F1: {test_metrics['f1']:.4f}")

    print("\nTRAINING COMPLETE")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
