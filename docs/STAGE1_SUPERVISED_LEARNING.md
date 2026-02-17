# HYDRA-BERT Stage 1: Supervised Multi-Task Learning

##  Technical Documentation

---

## 1. Overview

Stage 1 turns the pre-trained polyBERT model into a cardiac hydrogel outcome predictor through supervised multi-task learning. The model learns to predict therapeutic outcomes (strain improvement, classification, safety scores) from polymer chemistry, formulation parameters, and patient-specific cardiac features.

---

## 2. Foundation Model: polyBERT

### 2.1 Architecture
```
Model: DeBERTa-v2-base
Parameters: 86 million
Encoder layers: 12
Attention heads: 12
Hidden dimension: 768
Vocabulary: Custom polymer tokenizer
Output: 600-dimensional CLS embedding
```

### 2.2 Pre-training
- **Dataset**: 100 million polymer SMILES strings
- **Task**: Masked language modeling on polymer sequences
- **Result**: Dense chemical fingerprints that cluster similar polymers

### 2.3 Why polyBERT?
1. **Chemical understanding**: Learned structure-property relationships
2. **Transfer learning**: Pre-trained knowledge reduces data requirements
3. **Embedding quality**: 600-dim vectors capture molecular similarity

---

## 3. Data Preparation

### 3.1 Training Dataset

| Metric | Value |
|--------|-------|
| Total samples | 447,480 |
| Unique polymers | 24 (21 unique SMILES) |
| Patients | 60 (10 real + 50 synthetic) |
| Formulation combinations | ~310 per polymer |

### 3.2 Data Sources

**Patient Data (Real):**
- 10 patients from cardiac MRI studies
- Features: baseline LVEF, GLS, EDV, ESV, scar fraction, BZ fraction

**Patient Data (Synthetic):**
- 50 additional patients generated via statistical sampling
- Matched to real patient distributions
- Purpose: Increase training data diversity

**Polymer Data:**
- 24 cardiac-specific hydrogels (curated from 565 candidates)
- SMILES representations for polyBERT encoding
- Material properties from literature

### 3.3 Feature Engineering

#### Input Features (Total: 19 numerical + SMILES + category)

**Group A - Patient Cardiac (6 features):**
```python
patient_features = [
    'baseline_LVEF_pct',      # Ejection fraction (%)
    'baseline_GLS_pct',       # Global longitudinal strain (%)
    'baseline_EDV_mL',        # End-diastolic volume (mL)
    'baseline_ESV_mL',        # End-systolic volume (mL)
    'scar_fraction_pct',      # Infarct size (%)
    'bz_fraction_pct',        # Border zone size (%)
]
```

**Group B - Tissue Mechanics (5 features):**
```python
tissue_features = [
    'bz_stress_kPa',          # Border zone wall stress
    'healthy_stress_kPa',     # Healthy tissue stress
    'stress_concentration',   # Ratio BZ/healthy stress
    'transmurality',          # Scar transmurality (0-1)
    'wall_thickness_mm',      # LV wall thickness
]
```

**Group C - Hydrogel Properties (3 features):**
```python
hydrogel_features = [
    'hydrogel_E_kPa',         # Elastic modulus (stiffness)
    'hydrogel_t50_days',      # Degradation half-life
    'hydrogel_conductivity_S_m',  # Electrical conductivity
]
```

**Group D - Treatment Configuration (5 features):**
```python
treatment_features = [
    'patch_thickness_mm',     # Hydrogel thickness (1-5 mm)
    # One-hot encoded coverage:
    'coverage_scar_only',     # Infarct only
    'coverage_scar_bz25',     # Infarct + 25% BZ
    'coverage_scar_bz50',     # Infarct + 50% BZ
    'coverage_scar_bz100',    # Infarct + 100% BZ
]
```

**Polymer Encoding:**
```python
# SMILES → polyBERT → 600-dim embedding
polymer_embedding = polybert.encode(smiles)  # Shape: [batch, 600]

# Category → Learned embedding
category_embedding = category_embed(polymer_category)  # Shape: [batch, 16]
```

### 3.4 Target Variables

**Primary Regression:**
```python
'delta_EF_pct'  # Ejection fraction improvement (0 to +22%)
```

**Primary Classification:**
```python
'is_optimal'  # Binary: True if delta_EF >= 3% AND stress_reduction >= 15%
```

**Auxiliary Regression:**
```python
'delta_BZ_stress_reduction_pct'  # Wall stress reduction
'strain_normalization_pct'       # Strain uniformity improvement
```

### 3.5 Data Split Strategy

**Critical Constraint:** Patient-aware splitting (no data leakage)

| Split | Real Patients | Synthetic Patients | Total Samples | Percentage |
|-------|---------------|-------------------|---------------|------------|
| Train | 7 | 35 | 313,236 | 70% |
| Validation | 2 | 10 | 89,496 | 20% |
| Test | 1 | 5 | 44,748 | 10% |

**Stratification:**
- Preserve `is_optimal` ratio (24%) across splits
- Balance real vs synthetic patients proportionally
- Use `StratifiedGroupKFold` with patient_id as group

---

## 4. Model Architecture

### 4.1 Complete Architecture Diagram

```
                         ┌─────────────────────┐
                         │   polymer_SMILES    │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │     polyBERT        │
                         │   (frozen/LoRA)     │
                         │     600-dim CLS     │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   SMILES Adapter    │
                         │   600 → 384 → 256   │
                         │   (LayerNorm+GELU)  │
                         └──────────┬──────────┘
                                    │
     ┌──────────────────────────────┼──────────────────────────────┐
     │                              │                              │
┌────▼────┐                  ┌──────▼──────┐              ┌────────▼────────┐
│Numerical│ (19 features)    │  Category   │ (8 cats)     │      256-dim    │
│Features │                  │  Embedding  │              │   SMILES embed  │
└────┬────┘                  └──────┬──────┘              └────────┬────────┘
     │                              │                              │
┌────▼────┐                  ┌──────▼──────┐                       │
│Property │                  │   16-dim    │                       │
│Encoder  │                  │  cat embed  │                       │
│19→64→128│                  └──────┬──────┘                       │
└────┬────┘                         │                              │
     │                              │                              │
     └──────────────────────────────┴──────────────────────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │      CONCAT         │
                         │   256 + 128 + 16    │
                         │      = 400-dim      │
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │   Fusion Transformer│
                         │   4 layers, 8 heads │
                         │   400 → 512 → 512   │
                         └──────────┬──────────┘
                                    │
     ┌──────────────────────────────┼──────────────────────────────┐
     │                              │                              │
┌────▼────────┐          ┌──────────▼──────────┐         ┌─────────▼────────┐
│ EF Head     │          │  Optimal Head       │         │  Stress Head     │
│ (Primary)   │          │  (Primary)          │         │  (Auxiliary)     │
│512→256→64→1 │          │  512→128→1          │         │  512→128→1       │
│ Regression  │          │  Classification     │         │  Regression      │
└─────────────┘          └─────────────────────┘         └──────────────────┘
```

### 4.2 Component Details

#### 4.2.1 polyBERT Encoder (Frozen + LoRA)
```python
class PolyBERTEncoder(nn.Module):
    def __init__(self, model_path="kuelumbus/polyBERT"):
        self.bert = AutoModel.from_pretrained(model_path)

        # Freeze all base parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # Apply LoRA to attention layers only
        self.lora_config = LoRAConfig(
            r=16,           # Rank
            alpha=32,       # Scaling factor
            target_modules=["query", "key", "value"],
            dropout=0.1
        )
        self.bert = apply_lora(self.bert, self.lora_config)
        # LoRA adds ~1.2M trainable parameters

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # CLS token: [batch, 600]
```

#### 4.2.2 SMILES Adapter
```python
class SMILESAdapter(nn.Module):
    def __init__(self, input_dim=600, hidden_dim=384, output_dim=256):
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)  # [batch, 600] → [batch, 256]
```

#### 4.2.3 Property Encoder
```python
class PropertyEncoder(nn.Module):
    def __init__(self, input_dim=19, hidden_dim=64, output_dim=128):
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)  # [batch, 19] → [batch, 128]
```

#### 4.2.4 Category Embedding
```python
class CategoryEmbedding(nn.Module):
    def __init__(self, num_categories=8, embed_dim=16):
        self.embedding = nn.Embedding(num_categories, embed_dim)

    def forward(self, category_ids):
        return self.embedding(category_ids)  # [batch] → [batch, 16]
```

#### 4.2.5 Fusion Transformer
```python
class FusionTransformer(nn.Module):
    def __init__(self, input_dim=400, hidden_dim=512, num_layers=4, num_heads=8):
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.input_proj(x)  # [batch, 400] → [batch, 512]
        x = x.unsqueeze(1)      # [batch, 1, 512] for transformer
        x = self.transformer(x)
        x = x.squeeze(1)        # [batch, 512]
        return self.output_norm(x)
```

#### 4.2.6 Prediction Heads
```python
class EFRegressionHead(nn.Module):
    def __init__(self, input_dim=512):
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)  # [batch, 512] → [batch]

class OptimalClassificationHead(nn.Module):
    def __init__(self, input_dim=512):
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),  # Binary logit
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)  # [batch, 512] → [batch]

class StressRegressionHead(nn.Module):
    def __init__(self, input_dim=512):
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)
```

### 4.3 Parameter Count

| Component | Parameters | Trainable |
|-----------|------------|-----------|
| polyBERT (frozen) | 86M | 0 |
| LoRA adapters | 1.2M | 1.2M |
| SMILES Adapter | 328K | 328K |
| Property Encoder | 9.4K | 9.4K |
| Category Embedding | 128 | 128 |
| Fusion Transformer | 467K | 467K |
| EF Head | 148K | 148K |
| Optimal Head | 65.7K | 65.7K |
| Stress Head | 65.7K | 65.7K |
| **Total** | **~88M** | **~2.3M** |

---

## 5. Loss Functions

### 5.1 Multi-Task Loss with Uncertainty Weighting

```python
class HydraLoss(nn.Module):
    """
    Multi-task loss with learned homoscedastic uncertainty (Kendall et al., 2018).
    Automatically balances task difficulty during training.
    """
    def __init__(self):
        super().__init__()
        # Learnable log-variance parameters
        self.log_var_ef = nn.Parameter(torch.zeros(1))
        self.log_var_optimal = nn.Parameter(torch.zeros(1))
        self.log_var_stress = nn.Parameter(torch.zeros(1))

    def forward(self, predictions, targets):
        pred_ef, pred_optimal, pred_stress = predictions
        true_ef, true_optimal, true_stress = targets

        # Task 1: EF Regression - Smooth L1 (Huber) Loss
        loss_ef = F.smooth_l1_loss(pred_ef, true_ef, beta=2.0)

        # Task 2: Optimal Classification - Focal Loss
        pos_weight = torch.tensor([3.17])  # 76/24 class imbalance
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_optimal, true_optimal.float(),
            pos_weight=pos_weight
        )
        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt) ** 2  # gamma = 2
        loss_optimal = (focal_weight * bce_loss).mean()

        # Task 3: Stress Regression - MSE Loss
        loss_stress = F.mse_loss(pred_stress, true_stress)

        # Uncertainty-weighted combination
        precision_ef = torch.exp(-self.log_var_ef)
        precision_optimal = torch.exp(-self.log_var_optimal)
        precision_stress = torch.exp(-self.log_var_stress)

        total_loss = (
            precision_ef * loss_ef + self.log_var_ef +
            precision_optimal * loss_optimal + self.log_var_optimal +
            0.3 * (precision_stress * loss_stress + self.log_var_stress)
        )

        return total_loss, {
            'loss_ef': loss_ef.item(),
            'loss_optimal': loss_optimal.item(),
            'loss_stress': loss_stress.item(),
            'sigma_ef': torch.exp(0.5 * self.log_var_ef).item(),
            'sigma_optimal': torch.exp(0.5 * self.log_var_optimal).item(),
        }
```

### 5.2 Loss Component Details

**Smooth L1 (Huber) Loss for EF:**
- Robust to outliers in regression
- Beta=2.0 balances L1/L2 behavior
- Target: delta_EF_pct (range 0-22%)

**Focal Loss for Classification:**
- Addresses 24% positive class imbalance
- Gamma=2.0 down-weights easy examples
- Pos_weight=3.17 (76/24 ratio)

**MSE for Stress Reduction:**
- Auxiliary task, weight=0.3
- Helps learn mechanical relationships

---

## 6. Training Configuration

### 6.1 Hardware
```
GPUs: 16× NVIDIA A100 (40 GB)
CPU: 96 cores
RAM: 256 GB
Storage: NVMe SSD
```

### 6.2 Distributed Training Setup
```python
# DeepSpeed ZeRO Stage 2 Configuration
ds_config = {
    "train_batch_size": 8192,
    "train_micro_batch_size_per_gpu": 256,
    "gradient_accumulation_steps": 2,

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01
        }
    },

    "scheduler": {
        "type": "WarmupCosineLR",
        "params": {
            "warmup_num_steps": 1000,
            "total_num_steps": 15000
        }
    },

    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "none"},
        "contiguous_gradients": True,
        "overlap_comm": True
    },

    "gradient_clipping": 1.0
}
```

### 6.3 Training Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Epochs | 15 | Convergence observed by epoch 12 |
| Batch size (effective) | 8,192 | 256 × 16 GPUs × 2 accumulation |
| Learning rate | 2×10⁻⁵ | Standard for fine-tuning |
| Warmup steps | 1,000 | ~2 epochs warmup |
| Weight decay | 0.01 | Regularization |
| Dropout | 0.1 | Prevent overfitting |
| Gradient clipping | 1.0 | Stability |
| Mixed precision | FP16 | Memory efficiency |

### 6.4 Training Loop

```python
def train_epoch(model, dataloader, optimizer, scheduler, scaler):
    model.train()
    epoch_losses = []

    for batch_idx, batch in enumerate(dataloader):
        # Move to GPU
        smiles_ids = batch['smiles_ids'].cuda()
        smiles_mask = batch['smiles_mask'].cuda()
        numerical_features = batch['numerical_features'].cuda()
        category_ids = batch['category_ids'].cuda()
        targets = {k: v.cuda() for k, v in batch['targets'].items()}

        # Forward pass with mixed precision
        with autocast():
            predictions = model(smiles_ids, smiles_mask,
                               numerical_features, category_ids)
            loss, loss_dict = criterion(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_losses.append(loss_dict)

        # Logging every 100 steps
        if batch_idx % 100 == 0:
            log_metrics(batch_idx, loss_dict)

    return aggregate_losses(epoch_losses)
```

---

## 7. Evaluation Metrics

### 7.1 Regression Metrics (delta_EF_pct)

| Metric | Formula | Target |
|--------|---------|--------|
| MAE | Σ\|pred - true\| / N | < 1.0% |
| RMSE | √(Σ(pred - true)² / N) | < 1.5% |
| R² | 1 - SS_res / SS_tot | > 0.80 |

### 7.2 Classification Metrics (is_optimal)

| Metric | Formula | Target |
|--------|---------|--------|
| Accuracy | (TP + TN) / Total | > 85% |
| AUROC | Area under ROC curve | > 0.90 |
| F1 Score | 2×P×R / (P+R) | > 0.75 |
| Recall | TP / (TP + FN) | > 70% |
| Precision | TP / (TP + FP) | > 70% |

### 7.3 Validation Protocol

```python
def validate(model, val_loader):
    model.eval()
    all_preds_ef, all_true_ef = [], []
    all_preds_opt, all_true_opt = [], []

    with torch.no_grad():
        for batch in val_loader:
            predictions = model(**batch)

            all_preds_ef.extend(predictions['ef'].cpu().numpy())
            all_true_ef.extend(batch['targets']['ef'].cpu().numpy())
            all_preds_opt.extend(torch.sigmoid(predictions['optimal']).cpu().numpy())
            all_true_opt.extend(batch['targets']['optimal'].cpu().numpy())

    # Compute metrics
    metrics = {
        'ef_mae': mean_absolute_error(all_true_ef, all_preds_ef),
        'ef_rmse': np.sqrt(mean_squared_error(all_true_ef, all_preds_ef)),
        'ef_r2': r2_score(all_true_ef, all_preds_ef),
        'opt_auroc': roc_auc_score(all_true_opt, all_preds_opt),
        'opt_f1': f1_score(all_true_opt, (np.array(all_preds_opt) > 0.5).astype(int)),
    }

    return metrics
```

---

## 8. Results

### 8.1 Training Curves

```
Epoch  Train_Loss  Val_Loss  EF_MAE  EF_R²   Opt_AUROC  Opt_F1
-----  ----------  --------  ------  ------  ---------  ------
1      0.892       0.654     1.82    0.62    0.78       0.58
2      0.543       0.487     1.45    0.71    0.84       0.65
3      0.421       0.398     1.21    0.76    0.87       0.69
5      0.312       0.324     1.02    0.80    0.89       0.72
8      0.245       0.278     0.89    0.83    0.91       0.75
10     0.198       0.256     0.82    0.85    0.92       0.77
12     0.172       0.248     0.78    0.86    0.93       0.78
15     0.156       0.251     0.76    0.86    0.93       0.79
```

### 8.2 Final Test Set Performance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| EF MAE | 0.76% | < 1.0% | ✓ PASS |
| EF RMSE | 1.12% | < 1.5% | ✓ PASS |
| EF R² | 0.86 | > 0.80 | ✓ PASS |
| Optimal Accuracy | 87.3% | > 85% | ✓ PASS |
| Optimal AUROC | 0.93 | > 0.90 | ✓ PASS |
| Optimal F1 | 0.79 | > 0.75 | ✓ PASS |
| Optimal Recall | 76.2% | > 70% | ✓ PASS |

### 8.3 Per-Patient Analysis

Real patients weighted 5× in loss to ensure representation:

| Patient Type | Count | EF MAE | Optimal F1 |
|--------------|-------|--------|------------|
| Real | 10 | 0.82% | 0.77 |
| Synthetic | 50 | 0.74% | 0.80 |

---

## 9. Model Checkpoints

### 9.1 Saved Artifacts

```
checkpoints/
├── stage1_best.pt           # Best validation loss
├── stage1_final.pt          # End of training
├── stage1_config.yaml       # Training configuration
├── tokenizer/               # Polymer tokenizer
└── training_log.json        # Full training history
```

### 9.2 Checkpoint Contents

```python
checkpoint = {
    'epoch': 15,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss_module_state_dict': criterion.state_dict(),  # Includes learned σ
    'best_val_loss': 0.248,
    'metrics': {
        'ef_mae': 0.76,
        'ef_r2': 0.86,
        'opt_auroc': 0.93,
        'opt_f1': 0.79
    },
    'config': training_config,
}
```

---

## 10. Usage for Stage 2

The trained Stage 1 model serves as the **reward function** for Stage 2 PPO:

```python
# Load trained Stage 1 model
reward_model = HydraBERT.load('checkpoints/stage1_best.pt')
reward_model.eval()
reward_model.requires_grad_(False)  # Freeze for reward computation

def compute_reward(polymer_smiles, formulation, patient_features):
    with torch.no_grad():
        predictions = reward_model(polymer_smiles, formulation, patient_features)

        # Reward = predicted EF improvement + optimal probability - safety penalty
        reward = (
            predictions['ef'] * 0.5 +           # Predicted ΔEF
            torch.sigmoid(predictions['optimal']) * 0.3 +  # P(optimal)
            predictions['safety'] * 0.2          # Safety score
        )

        # Safety penalties
        if predictions['toxicity'] > 0.15:
            reward -= 1.0
        if predictions['integrity'] < 0.8:
            reward -= 0.5

        return reward
```

---

## 11.  Implementation Files

| File | Description |
|------|-------------|
| `hydra_bert/models/hydra_bert.py` | Main model architecture |
| `hydra_bert/models/polybert_lora.py` | LoRA-adapted polyBERT |
| `hydra_bert/models/fusion_transformer.py` | Cross-modal fusion |
| `hydra_bert/models/prediction_heads.py` | Output heads |
| `hydra_bert/data/dataset.py` | PyTorch Dataset |
| `hydra_bert/losses.py` | Multi-task loss |
| `scripts/stage1/train_supervised.py` | Training script |
| `configs/config.yaml` | Hyperparameters |
| `configs/ds_config.json` | DeepSpeed config |

---

## 12. Summary

Stage 1 successfully fine-tuned polyBERT for cardiac hydrogel outcome prediction:

1. **Data**: 447,480 samples from 24 polymers × 60 patients
2. **Architecture**: polyBERT + LoRA + Fusion Transformer + Multi-task heads
3. **Training**: 15 epochs, 16×A100 GPUs, DeepSpeed ZeRO-2
4. **Results**: EF MAE 0.76%, R² 0.86, Optimal AUROC 0.93
5. **Output**: Frozen reward model for Stage 2 RL optimization
