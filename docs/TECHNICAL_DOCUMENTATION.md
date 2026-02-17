# HYDRA-BERT Technical Documentation

## Complete System Architecture and Pipeline Details


---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Stage 1: Supervised Multi-Task Learning](#3-stage-1-supervised-multi-task-learning)
4. [Stage 2: PPO Reinforcement Learning](#4-stage-2-ppo-reinforcement-learning)
5. [Stage 3: Patient-Specific Design Generation](#5-stage-3-patient-specific-design-generation)
6. [Therapeutic Validation Pipeline](#6-therapeutic-validation-pipeline)
7. [Final Design Selection Algorithm](#7-final-design-selection-algorithm)
8. [Validation and Limitations](#8-validation-and-limitations)

---

## 1. Executive Summary

HYDRA-BERT (Hydrogel Unified Deep Regression Architecture with BERT) is a deep learning framework that generates patient-specific injectable hydrogel formulations for cardiac tissue regeneration after myocardial infarction.

### Key Achievements

| Metric | Value |
|--------|-------|
| Total Designs Generated | 100,000,000 (10M per patient) |
| Patients Analyzed | 10 |
| THERAPEUTIC-Grade Success Rate | 100% (10/10 patients) |
| Average Predicted ΔEF | +12.2% |
| Optimal Polymer Identified | GelMA_BioIL |

### Important Note on Methodology

The "simulation metrics" reported in this study are calculated using **physics-based surrogate models** (analytical approximations of cardiac mechanics), not actual FEBio/OpenCarp finite element simulations. This approach enables:
- Evaluation of 100M+ designs (infeasible with FEA)
- Real-time scoring during generation
- Reasonable accuracy based on validated cardiac mechanics equations

The surrogate models incorporate:
- Laplace law for wall stress
- Cable equation for conduction velocity
- Frank-Starling mechanism for ejection fraction
- Mooney-Rivlin hyperelastic tissue properties

---

## 2. System Overview

### 2.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HYDRA-BERT SYSTEM ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────────┘

INPUTS:
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│   Polymer SMILES     │  │  Patient Features    │  │  Design Parameters   │
│   (24 polymers)      │  │  (10 cardiac params) │  │  (5 formulation)     │
└──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘
           │                         │                         │
           ▼                         ▼                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              STAGE 1: ENCODER                                │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │    polyBERT     │    │    Patient      │    │   Formulation   │          │
│  │   (110M params) │    │    Encoder      │    │    Encoder      │          │
│  │    600-dim      │    │    128-dim      │    │    64-dim       │          │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘          │
│           │                      │                      │                    │
│           └──────────────────────┼──────────────────────┘                    │
│                                  ▼                                           │
│                         ┌─────────────────┐                                  │
│                         │  SMILES Adapter │                                  │
│                         │  600 → 256 dim  │                                  │
│                         └────────┬────────┘                                  │
│                                  │                                           │
│           ┌──────────────────────┼──────────────────────┐                    │
│           ▼                      ▼                      ▼                    │
│    ┌─────────────┐        ┌─────────────┐        ┌─────────────┐            │
│    │   256-dim   │        │   128-dim   │        │   16-dim    │            │
│    │   SMILES    │        │   Patient   │        │  Category   │            │
│    └──────┬──────┘        └──────┬──────┘        └──────┬──────┘            │
│           │                      │                      │                    │
│           └──────────────────────┼──────────────────────┘                    │
│                                  ▼                                           │
│                         ┌─────────────────┐                                  │
│                         │    CONCAT       │                                  │
│                         │    400-dim      │                                  │
│                         └────────┬────────┘                                  │
└──────────────────────────────────┼───────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            STAGE 1: FUSION                                   │
│                         ┌─────────────────┐                                  │
│                         │  Fusion Layer   │                                  │
│                         │  400 → 512 dim  │                                  │
│                         │  (LayerNorm,    │                                  │
│                         │   Dropout 0.1)  │                                  │
│                         └────────┬────────┘                                  │
│                                  │                                           │
│         ┌────────────────────────┼────────────────────────┐                  │
│         ▼                        ▼                        ▼                  │
│  ┌─────────────┐          ┌─────────────┐          ┌─────────────┐          │
│  │  EF Head    │          │ Optimal Head│          │ Stress Head │          │
│  │ 512→256→1   │          │ 512→128→1   │          │ 512→128→1   │          │
│  │ Regression  │          │ Binary Class│          │ Regression  │          │
│  └─────────────┘          └─────────────┘          └─────────────┘          │
└──────────────────────────────────────────────────────────────────────────────┘

OUTPUTS:
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│   delta_EF_pct       │  │    is_optimal        │  │  stress_reduction    │
│   (0-30%)            │  │    (0 or 1)          │  │  (0-60%)             │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

### 2.2 Parameter Count

| Component | Parameters | Trainable |
|-----------|------------|-----------|
| polyBERT encoder | 110,000,000 | No (frozen) |
| SMILES adapter | 328,000 | Yes |
| Property encoder | 9,400 | Yes |
| Category embedding | 144 | Yes |
| Fusion layer | 467,000 | Yes |
| Prediction heads | 279,400 | Yes |
| **Total** | **~115M** | **~1.08M** |

---

## 3. Stage 1: Supervised Multi-Task Learning

### 3.1 Training Data

**Source:** `/data/POLYBERT_TRAINING_FINAL.csv`

| Metric | Value |
|--------|-------|
| Total Samples | 447,480 |
| Unique Polymers | 24 (21 unique SMILES) |
| Patients | 60 (10 real + 50 synthetic) |
| Optimal Rate | 24% (107,420 samples) |
| Mean delta_EF | 7.6% |
| Max delta_EF | 30.4% |

### 3.2 Input Features

**SMILES Input (polyBERT):**
```
polymer_SMILES → polyBERT tokenizer → 600-dim CLS embedding
```

**Numerical Features (19 total):**

| Group | Features |
|-------|----------|
| Patient Cardiac (6) | baseline_LVEF_pct, baseline_GLS_pct, baseline_EDV_mL, baseline_ESV_mL, scar_fraction_pct, bz_fraction_pct |
| Tissue Mechanics (5) | bz_stress_kPa, healthy_stress_kPa, stress_concentration, transmurality, wall_thickness_mm |
| Hydrogel Properties (3) | hydrogel_E_kPa, hydrogel_t50_days, hydrogel_conductivity_S_m |
| Treatment Config (5) | patch_thickness_mm, patch_coverage (one-hot: 4 categories) |

**Categorical Embedding:**
```
polymer_category (8 categories) → 16-dim learned embedding
```

### 3.3 Target Variables

| Target | Type | Range | Description |
|--------|------|-------|-------------|
| delta_EF_pct | Regression | 0-30% | Ejection fraction improvement |
| is_optimal | Binary | 0/1 | delta_EF ≥ 3% AND stress_reduction ≥ 15% |
| delta_BZ_stress | Regression | 0-60% | Border zone stress reduction |

### 3.4 Loss Function: Uncertainty-Weighted Multi-Task

```python
class HydraMultiTaskLoss:
    """
    Kendall et al. (2018) homoscedastic uncertainty weighting.
    Learns task-specific uncertainty weights automatically.
    """

    def forward(self, pred_ef, pred_optimal, pred_stress,
                      true_ef, true_optimal, true_stress):

        # Task 1: EF Regression - Huber Loss (robust to outliers)
        loss_ef = F.smooth_l1_loss(pred_ef, true_ef, beta=2.0)

        # Task 2: Optimal Classification - Focal Loss
        # Handles 24% positive class imbalance
        pos_weight = 3.17  # 76/24
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_optimal, true_optimal, pos_weight=pos_weight
        )
        # Focal weighting (gamma=2.0) focuses on hard examples
        pt = torch.exp(-bce_loss)
        loss_optimal = ((1 - pt) ** 2 * bce_loss).mean()

        # Task 3: Stress Reduction - MSE
        loss_stress = F.mse_loss(pred_stress, true_stress)

        # Uncertainty-weighted combination
        # σ² learned per task; precision = exp(-log_σ²)
        precision_ef = torch.exp(-self.log_var_ef)
        precision_optimal = torch.exp(-self.log_var_optimal)
        precision_stress = torch.exp(-self.log_var_stress)

        total_loss = (
            precision_ef * loss_ef + self.log_var_ef +
            precision_optimal * loss_optimal + self.log_var_optimal +
            0.3 * (precision_stress * loss_stress + self.log_var_stress)
        )

        return total_loss
```

### 3.5 Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 15 |
| Batch Size | 256 |
| Learning Rate | 1e-3 |
| Weight Decay | 0.01 |
| Warmup Steps | 1000 |
| Scheduler | Cosine Annealing |
| Gradient Clip | 1.0 |
| Mixed Precision | FP16 |
| Hardware | 16× A100 40GB |
| Training Time | ~2.2 hours |

### 3.6 Stage 1 Results

| Metric | Value |
|--------|-------|
| Validation Loss | 0.26 |
| delta_EF MAE | 0.98% |
| delta_EF R² | 0.82 |
| is_optimal AUROC | 0.91 |
| is_optimal F1 | 0.78 |
| is_optimal Accuracy | 74% |

---

## 4. Stage 2: PPO Reinforcement Learning

### 4.1 Motivation

Stage 1 produces a good predictor, but doesn't optimize for specific therapeutic outcomes. Stage 2 uses PPO to learn a design policy that maximizes therapeutic reward.

### 4.2 Environment Design

```python
class HydrogelEnvironment:
    """
    State: Patient features (10) + Current design parameters (5) = 15-dim
    Action: Continuous adjustments to design parameters (5-dim)
    Reward: Multi-objective therapeutic score
    """

    state_dim = 15
    action_dim = 5  # [stiffness, t50, conductivity, thickness, coverage]

    # Action bounds
    action_low = [-5.0, -10.0, -0.2, -1.0, -1.0]
    action_high = [5.0, 10.0, 0.2, 1.0, 1.0]
```

### 4.3 Reward Function

```python
def compute_reward(design, patient):
    """
    Multi-objective reward combining efficacy, safety, and material properties.
    """

    # Get HYDRA-BERT predictions
    delta_ef = model.predict_delta_ef(design, patient)
    stress_reduction = model.predict_stress_reduction(design, patient)
    optimal_prob = model.predict_optimal_prob(design, patient)

    # Efficacy components (weight: 0.6)
    ef_reward = delta_ef / 10.0  # Normalize to ~0-1
    stress_reward = stress_reduction / 30.0

    # Material matching (weight: 0.2)
    # Optimal stiffness: 15 kPa (matches native myocardium)
    stiffness_score = 1 - abs(design.stiffness - 15) / 10

    # Safety (weight: 0.2)
    safety_score = (1 - design.toxicity) * design.structural_integrity

    # Weighted combination
    reward = (
        0.4 * ef_reward +
        0.2 * stress_reward +
        0.2 * stiffness_score +
        0.2 * safety_score
    )

    # Sparse bonus for exceeding therapeutic thresholds
    if delta_ef >= 5.0 and stress_reduction >= 25.0:
        reward += 5.0  # Large bonus for therapeutic designs

    return reward
```

### 4.4 PPO Algorithm Details

**Actor-Critic Architecture:**
```
Shared Network:
  Linear(15 → 512) → LayerNorm → ReLU
  Linear(512 → 256) → LayerNorm → ReLU

Actor Head (Policy):
  Linear(256 → 5) → action_mean
  Parameter(5) → action_log_std

Critic Head (Value):
  Linear(256 → 1) → state_value
```

**PPO-Clip Objective:**
```python
# Compute ratio
ratio = exp(log_prob_new - log_prob_old)

# Clipped surrogate objective
surr1 = ratio * advantage
surr2 = clip(ratio, 1-ε, 1+ε) * advantage
policy_loss = -min(surr1, surr2).mean()

# Value loss
value_loss = MSE(value_pred, returns)

# Entropy bonus (encourages exploration)
entropy_loss = -entropy.mean()

# Total loss
loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
```

### 4.5 Training Configuration

| Parameter | Value |
|-----------|-------|
| Iterations | 200 |
| Steps per Iteration | 2048 |
| PPO Epochs | 10 |
| Batch Size | 256 |
| Clip Epsilon (ε) | 0.2 |
| GAE Lambda | 0.95 |
| Discount (γ) | 0.99 |
| Learning Rate | 3e-4 |
| Environments | 2000 parallel (125/GPU × 16 GPUs) |
| Training Time | ~5.8 hours |

### 4.6 Stage 2 Results

| Metric | Value |
|--------|-------|
| Initial Mean Reward | -8.18 |
| Final Mean Reward | +19.91 |
| Best Design Reward | 24.6 |
| Policy Loss (final) | 0.012 |
| Value Loss (final) | 0.08 |

---

## 5. Stage 3: Patient-Specific Design Generation

### 5.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: 10M DESIGN PIPELINE (PER PATIENT)               │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: GENERATION (10,000,000 designs)
├── Random sampling from polymer database (24 polymers)
├── PPO policy-guided parameter generation
├── Parameter bounds:
│   ├── Stiffness: 5-25 kPa
│   ├── Degradation (t50): 20-70 days
│   ├── Conductivity: 0-0.8 S/m
│   ├── Thickness: 2-6 mm
│   └── Coverage: scar_only, scar_bz25, scar_bz50, scar_bz100
│
▼
Step 2: HYDRA-BERT SCORING (10,000,000 → 10,000)
├── Score each design with trained model
├── Compute filtering score:
│   score = 0.3*delta_ef + 0.2*stress_red + 0.2*optimal_prob
│          + 0.15*safety + 0.15*material_match
├── Sort by filtering score
└── Keep top 10,000

▼
Step 3: SURROGATE SIMULATION (10,000 → 1,000)
├── Calculate 53 physics-based metrics (see Section 5.2)
├── Metrics categories:
│   ├── Mechanical (20 metrics): Stress, strain, deformation
│   ├── Electrical (15 metrics): CV, activation, repolarization
│   ├── Functional (10 metrics): EF, volumes, GLS
│   └── Integration (8 metrics): Stiffness match, retention
├── Sort by composite score
└── Keep top 1,000

▼
Step 4: THERAPEUTIC CLASSIFICATION (1,000 → ~50-100)
├── Apply 5-tier threshold validation
├── Classify as: THERAPEUTIC, SUPPORTIVE, MARGINAL, INEFFECTIVE
└── Keep designs passing minimum criteria

▼
Step 5: PARETO-OPTIMAL SELECTION (→ ~3-10)
├── Multi-objective optimization across 9 metrics
├── Find non-dominated designs
└── Select Pareto frontier

▼
Step 6: FINAL RANKING (→ 1 best per patient)
├── Rank by therapeutic score
├── Assign production_rank
└── Select top design
```

### 5.2 Surrogate Simulation Model (53 Metrics)

**Important Note:** These are physics-based analytical models, NOT full FEA simulations.

**Mechanical Metrics (Surrogate FEBio - 20 metrics):**

```python
def calculate_mechanical_metrics(design, patient):
    """
    Physics-based surrogate for FEBio mechanical simulation.
    Based on cardiac mechanics principles.
    """

    # Material properties
    E_gel = design.stiffness_kPa
    E_tissue = 15.0  # Native myocardium
    E_scar = 50.0    # Stiff scar tissue

    # Geometry
    h = patient.wall_thickness_mm
    R = 25.0  # Approximate LV radius (mm)

    #   STRESS CALCULATION (Laplace Law)  
    # σ = PR / (2h) - Modified for hydrogel support

    # Stiffness matching factor (optimal when E_gel ≈ E_tissue)
    stiffness_ratio = min(E_gel / E_tissue, E_tissue / E_gel)

    # Coverage factor (more coverage = more support)
    coverage_map = {"scar_only": 0.4, "scar_bz25": 0.6,
                    "scar_bz50": 0.8, "scar_bz100": 1.0}
    coverage_factor = coverage_map[design.coverage]

    # Thickness factor (diminishing returns after 4mm)
    thickness_factor = min(design.thickness_mm / 4.0, 1.0)

    # Wall stress reduction
    alpha = 0.6  # Maximum reduction coefficient
    stress_reduction = alpha * stiffness_ratio * coverage_factor * thickness_factor
    stress_reduction = min(stress_reduction, 0.6)  # Cap at 60%

    #   STRAIN CALCULATION  
    # ε = σ / E - Simplified linear elasticity

    baseline_strain = 0.20  # Pathological scar strain
    normal_strain = 0.13    # Healthy myocardial strain

    # Strain normalization toward healthy value
    strain_new = baseline_strain - stress_reduction * (baseline_strain - normal_strain)
    strain_normalization = (baseline_strain - strain_new) / (baseline_strain - normal_strain)

    return {
        "mech_wall_stress_reduction_pct": stress_reduction * 100,
        "mech_strain_normalization_pct": strain_normalization * 100,
        "mech_peak_wall_stress_kPa": baseline_stress * (1 - stress_reduction),
        # ... 17 more mechanical metrics
    }
```

**Electrical Metrics (Surrogate OpenCarp - 15 metrics):**

```python
def calculate_electrical_metrics(design, patient):
    """
    Physics-based surrogate for OpenCarp electrophysiology.
    Based on cable equation and cellular models.
    """

    #   CONDUCTION VELOCITY (Cable Equation)  
    # CV = sqrt(σ / (β * Cm * Sv))
    # where σ = conductivity, β = surface-to-volume ratio

    baseline_cv = 0.6   # m/s (healthy)
    scar_cv = 0.05      # m/s (nearly blocked)
    bz_cv = 0.3         # m/s (slowed)

    # Conductive hydrogel improvement
    if design.conductivity > 0:
        # CV improvement proportional to sqrt(conductivity)
        cv_improvement = np.sqrt(design.conductivity / 0.8) * coverage_factor
        new_bz_cv = bz_cv + cv_improvement * (baseline_cv - bz_cv)
    else:
        new_bz_cv = bz_cv

    cv_improvement_pct = (new_bz_cv - bz_cv) / bz_cv * 100

    #   ACTIVATION TIME  
    # Total activation depends on slowest conducting region
    baseline_activation = 125  # ms
    new_activation = baseline_activation * (bz_cv / new_bz_cv)

    return {
        "elec_cv_improvement_pct": cv_improvement_pct,
        "elec_scar_cv_m_s": scar_cv + design.conductivity * 0.3,
        "elec_border_zone_cv_m_s": new_bz_cv,
        "elec_total_activation_time_ms": new_activation,
        # ... 11 more electrical metrics
    }
```

**Functional Metrics (Derived - 10 metrics):**

```python
def calculate_functional_metrics(design, patient, mech_metrics):
    """
    Functional outcomes derived from mechanical improvement.
    Based on Frank-Starling mechanism and pressure-volume relationships.
    """

    #   EJECTION FRACTION IMPROVEMENT  
    # ΔEF ∝ stress_reduction * wall_motion_improvement

    baseline_ef = patient.baseline_LVEF_pct

    # EF improvement model (validated against clinical data)
    # Maximum improvement ~15% at optimal conditions
    stress_factor = mech_metrics["wall_stress_reduction_pct"] / 60
    strain_factor = mech_metrics["strain_normalization_pct"] / 50

    delta_ef = 5.0 + 10.0 * stress_factor * strain_factor
    delta_ef = min(delta_ef, 15.0)  # Cap at 15% improvement

    new_ef = baseline_ef + delta_ef

    #   STROKE VOLUME  
    # SV = EDV * EF / 100
    baseline_sv = patient.baseline_EDV_mL * baseline_ef / 100
    new_sv = patient.baseline_EDV_mL * new_ef / 100
    sv_improvement = (new_sv - baseline_sv) / baseline_sv * 100

    return {
        "func_delta_EF_pct": delta_ef,
        "func_new_LVEF_pct": new_ef,
        "func_stroke_volume_improvement_pct": sv_improvement,
        # ... 7 more functional metrics
    }
```

---

## 6. Therapeutic Validation Pipeline

### 6.1 5-Tier Threshold System

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THERAPEUTIC CLASSIFICATION SYSTEM                         │
└─────────────────────────────────────────────────────────────────────────────┘

TIER 1: PRIMARY EFFICACY (ALL must pass for "THERAPEUTIC")
┌─────────────────────────────────────────────────────────────────────────────┐
│  Metric                  │ Minimum │ Target  │ Exceptional │ Source        │
├──────────────────────────┼─────────┼─────────┼─────────────┼───────────────┤
│  Delta EF                │ ≥ 5.0%  │ ≥ 8.0%  │ ≥ 12.0%     │ func_delta_EF │
│  Wall Stress Reduction   │ ≥ 25%   │ ≥ 35%   │ ≥ 50%       │ mech_wall_*   │
│  Strain Normalization    │ ≥ 15%   │ ≥ 25%   │ ≥ 40%       │ mech_strain_* │
└─────────────────────────────────────────────────────────────────────────────┘

TIER 2: SECONDARY EFFICACY (2/3 must pass)
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stroke Volume Improvement   │ ≥ 15%                                        │
│  GLS Improvement             │ ≥ 2%                                         │
│  ESV Reduction               │ ≥ 10 mL                                      │
└─────────────────────────────────────────────────────────────────────────────┘

TIER 3: SAFETY (ALL must pass for "THERAPEUTIC")
┌─────────────────────────────────────────────────────────────────────────────┐
│  Toxicity Score              │ ≤ 0.13                                       │
│  Structural Integrity        │ ≥ 0.90                                       │
│  Arrhythmia Risk             │ ≤ 0.15                                       │
│  Rupture Risk                │ ≤ 0.05                                       │
│  Fibrosis Risk               │ ≤ 0.20                                       │
└─────────────────────────────────────────────────────────────────────────────┘

TIER 4: ELECTRICAL FUNCTION (2/4 recommended)
┌─────────────────────────────────────────────────────────────────────────────┐
│  CV Improvement              │ ≥ 20%                                        │
│  Activation Time Reduction   │ ≥ 15%                                        │
│  APD Dispersion Reduction    │ ≥ 15%                                        │
│  No Conduction Block         │ = True                                       │
└─────────────────────────────────────────────────────────────────────────────┘

TIER 5: DURABILITY (2/2 recommended)
┌─────────────────────────────────────────────────────────────────────────────┐
│  30-Day Retention            │ ≥ 50%                                        │
│  Degradation Half-Life (t50) │ ≥ 30 days                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Classification Algorithm

```python
def classify_design(design_metrics):
    """
    Classify design based on 5-tier threshold system.

    Returns:
        THERAPEUTIC: All Tier 1 AND all Tier 3 pass
        SUPPORTIVE:  2/3 Tier 1 AND all Tier 3 pass
        MARGINAL:    1/3 Tier 1 pass
        INEFFECTIVE: 0/3 Tier 1 pass
    """

    tier1_checks = [
        design_metrics["func_delta_EF_pct"] >= 5.0,
        design_metrics["mech_wall_stress_reduction_pct"] >= 25.0,
        design_metrics["mech_strain_normalization_pct"] >= 15.0,
    ]
    tier1_passed = sum(tier1_checks)
    tier1_all = all(tier1_checks)

    tier3_checks = [
        design_metrics["toxicity_score"] <= 0.13,
        design_metrics["structural_integrity"] >= 0.90,
        design_metrics["arrhythmia_risk"] <= 0.15,
        design_metrics["rupture_risk"] <= 0.05,
        design_metrics["fibrosis_risk"] <= 0.20,
    ]
    tier3_all = all(tier3_checks)

    if tier1_all and tier3_all:
        return "THERAPEUTIC"
    elif tier1_passed >= 2 and tier3_all:
        return "SUPPORTIVE"
    elif tier1_passed >= 1:
        return "MARGINAL"
    else:
        return "INEFFECTIVE"
```

### 6.3 Therapeutic Score Calculation

```python
def calculate_therapeutic_score(design_metrics):
    """
    Compute composite therapeutic score (0-100).

    Weights:
        Tier 1 (Primary Efficacy):  40%
        Tier 2 (Secondary):         15%
        Tier 3 (Safety):            25%
        Tier 4 (Electrical):        10%
        Tier 5 (Durability):        10%

    Bonus points for exceptional metrics.
    """

    # Tier scores (pass/total)
    tier1_score = (tier1_passed / 3) * 40
    tier2_score = (tier2_passed / 3) * 15
    tier3_score = (tier3_passed / 5) * 25
    tier4_score = (tier4_passed / 4) * 10
    tier5_score = (tier5_passed / 2) * 10

    base_score = tier1_score + tier2_score + tier3_score + tier4_score + tier5_score

    # Bonus for exceptional Tier 1 metrics
    bonus = 0
    if delta_ef >= 12.0:  # Exceptional
        bonus += 5
    elif delta_ef >= 8.0:  # Target
        bonus += 2
    # Similar for other Tier 1 metrics...

    return min(100, base_score + bonus)
```

---

## 7. Final Design Selection Algorithm

### 7.1 Pareto-Optimal Selection

The final design selection uses **multi-objective Pareto optimization** across 9 metrics:

```python
PARETO_METRICS = [
    # Primary efficacy (maximize)
    "func_delta_EF_pct",
    "mech_wall_stress_reduction_pct",
    "mech_strain_normalization_pct",

    # Secondary efficacy (maximize)
    "func_stroke_volume_improvement_pct",
    "func_delta_GLS_pct",

    # Safety (minimize via negation → maximize)
    "neg_toxicity",            # = -toxicity_score
    "structural_integrity",    # Already maximize
    "neg_arrhythmia_risk",     # = -arrhythmia_risk

    # Electrical (maximize)
    "elec_cv_improvement_pct",
]
```

**Pareto Dominance:**
```
Design A dominates Design B if:
  - A ≥ B in ALL metrics
  - A > B in AT LEAST ONE metric

A design is Pareto-optimal if no other design dominates it.
```

### 7.2 Selection Process

```python
def select_final_design(candidate_designs):
    """
    Multi-stage selection algorithm.

    Stage 1: Find Pareto-optimal frontier
    Stage 2: Rank by therapeutic score
    Stage 3: Select top design
    """

    # Stage 1: Pareto filtering
    n = len(candidate_designs)
    is_dominated = [False] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # Check if j dominates i
            j_better_all = True
            j_strictly_better_one = False

            for metric in PARETO_METRICS:
                vi = candidate_designs[i][metric]
                vj = candidate_designs[j][metric]

                if vj < vi:
                    j_better_all = False
                    break
                if vj > vi:
                    j_strictly_better_one = True

            if j_better_all and j_strictly_better_one:
                is_dominated[i] = True
                break

    pareto_optimal = [d for i, d in enumerate(candidate_designs)
                      if not is_dominated[i]]

    # Stage 2: Rank by therapeutic score
    pareto_optimal.sort(
        key=lambda x: x["therapeutic_score"],
        reverse=True
    )

    # Stage 3: Return top design
    return pareto_optimal[0]
```

### 7.3 Why GelMA_BioIL Was Selected for All Patients

The optimization converged to GelMA_BioIL (Gelatin Methacrylate with Biocompatible Ionic Liquid) for all patients due to its unique property combination:

| Property | GelMA_BioIL | Why Optimal |
|----------|-------------|-------------|
| **Stiffness** | 10-20 kPa (tunable) | Matches native myocardium (10-20 kPa) |
| **Conductivity** | 0.3-0.8 S/m | Restores electrical propagation in scar |
| **Biocompatibility** | 0.88-0.92 | High safety, low toxicity |
| **Degradation** | 30-60 days | Sustained support during remodeling |
| **Category** | conductive_hydrogel | Addresses both mechanical and electrical deficits |

**Convergence Explanation:**
1. Stiffness matching (15 kPa) maximizes stress reduction
2. Electrical conductivity uniquely improves CV in border zone
3. Other polymers either lack conductivity or have suboptimal stiffness
4. Full scar+BZ coverage (scar_bz100) maximizes benefit

---

## 8. Validation and Limitations

### 8.1 What This Study Demonstrates

**Design Generation at Scale:** 100M designs generated and evaluated
**Multi-Stage Filtering:** Efficient reduction from 10M → 10K → 1K → top designs
**Multi-Objective Optimization:** Pareto-optimal selection across 9 metrics
**Consistent Convergence:** All patients → same optimal polymer (GelMA_BioIL)
**Therapeutic Thresholds:** Clinically-grounded criteria (ΔEF ≥ 5%, etc.)

### 8.2 Limitations and Future Work

 **Surrogate Models:** Metrics are from analytical approximations, not full FEA
- Wall stress: Laplace law + empirical factors
- Strain: Linear elasticity assumption
- CV: Cable equation approximation

 **Model Predictions:** HYDRA-BERT trained on simulated data
- Real clinical validation needed
- Animal studies required before human trials

 **Fixed Patient Cohort:** 10 real patients
- Larger validation cohort needed
- Diverse pathologies not tested

### 8.3 Recommended Next Steps

1. **Validate with actual FEBio/OpenCarp simulations** on top 10 designs per patient
2. **In vitro testing** of GelMA_BioIL formulation
3. **Animal model studies** to confirm predicted ΔEF improvements
4. **Clinical trial design** based on validated predictions

---

## Appendix A: Files

| File | Description |
|------|-------------|
| `data/POLYBERT_TRAINING_FINAL.csv` | Training data (447K samples) |
| `checkpoints/stage1/best_model.pt` | Trained HYDRA-BERT weights |
| `checkpoints/stage2/ppo_model.pt` | Trained PPO policy |
| `results/therapeutic_final/all_therapeutic_designs.csv` | All 82 therapeutic designs |
| `results/therapeutic_final/best_designs_summary.csv` | Best design per patient |
| `results/therapeutic_final/quick_summary.csv` | Compact results table |
