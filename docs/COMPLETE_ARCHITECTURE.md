# HYDRA-BERT Complete Architecture

## Overview

**HYDRA-BERT** (Hydrogel Unified Deep Regression Architecture with BERT) is a three-stage deep learning pipeline for patient-specific cardiac hydrogel optimization. This document provides a comprehensive architectural overview of all components.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [polyBERT Foundation Model](#2-polybert-foundation-model)
3. [LoRA Adaptation Layer](#3-lora-adaptation-layer)
4. [Multi-Modal Feature Encoding](#4-multi-modal-feature-encoding)
5. [Fusion Network](#5-fusion-network)
6. [Multi-Task Prediction Heads](#6-multi-task-prediction-heads)
7. [PPO Policy Network (Stage 2)](#7-ppo-policy-network-stage-2)
8. [Design Generation Engine (Stage 3)](#8-design-generation-engine-stage-3)
9. [Complete Parameter Count](#9-complete-parameter-count)
10. [Forward Pass Flow](#10-forward-pass-flow)

---

## 1. System Architecture Overview

### 1.1 Three-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           HYDRA-BERT SYSTEM ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                         STAGE 1: SUPERVISED LEARNING                        │    │
│  │                                                                             │    │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────┐   │    │
│  │   │  polyBERT    │───▶│  LoRA        │───▶│  Fusion      │───▶│ Multi-  │   │    │
│  │   │  (86M frozen)│    │  Adapters    │    │  Network     │    │ Task    │   │    │
│  │   │              │    │  (1.2M)      │    │  (467K)      │    │ Heads   │   │    │
│  │   └──────────────┘    └──────────────┘    └──────────────┘    └─────────┘   │    │
│  │                                                                             │    │
│  │   Input: SMILES + Patient Features + Hydrogel Properties                    │    │
│  │   Output: Predicted ΔEF, Stress Reduction, is_optimal                       │    │
│  │   Training: 447,480 samples, 15 epochs                                      │    │
│  │                                                                             │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                         │                                           │
│                                         ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                     STAGE 2: REINFORCEMENT LEARNING (PPO)                   │    │
│  │                                                                             │    │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────┐   │    │
│  │   │  State       │───▶│  Policy      │───▶│  Action      │───▶│ Reward  │   │    │
│  │   │  Encoder     │    │  Network     │    │  Space       │    │ Model   │   │    │
│  │   │  (Stage 1)   │    │  (Actor)     │    │  (Hybrid)    │    │         │   │    │
│  │   └──────────────┘    └──────────────┘    └──────────────┘    └─────────┘   │    │
│  │                                                                             │    │
│  │   State: Patient profile embedded                                           │    │
│  │   Actions: Discrete (polymer) + Continuous (properties)                     │    │
│  │   Reward: Combined therapeutic score                                        │    │
│  │   Training: 1000 episodes, 50 steps each                                    │    │
│  │                                                                             │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                         │                                           │
│                                         ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                       STAGE 3: DESIGN GENERATION                            │    │
│  │                                                                             │    │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────┐   │    │
│  │   │  10M Design  │───▶│  HYDRA-BERT  │───▶│  Multi-Stage │───▶│ Optimal │   │    │
│  │   │  Generator   │    │  Inference   │    │  Filtering   │    │ Design  │   │    │
│  │   │              │    │              │    │  (6 stages)  │    │         │   │    │
│  │   └──────────────┘    └──────────────┘    └──────────────┘    └─────────┘   │    │
│  │                                                                             │    │
│  │   Generation: 10M candidates per patient                                    │    │
│  │   Filtering: 10M → 100K → 10K → 1K → 100 → 10 → 1                           │    │
│  │   Validation: FEBio mechanics + OpenCarp electrophysiology                  │    │
│  │                                                                             │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Relationships

```
Input Modalities          Encoders                Fusion              Outputs
───────────────          ────────                ──────              ───────

SMILES String ─────────▶ polyBERT + LoRA ─┐
                         (600-dim CLS)    │
                                          │
Patient Cardiac ───────▶ Patient Encoder ─┼──▶ Concat ──▶ Fusion ──▶ ΔEF Head
(6 features)             (64-dim)         │    (400-dim)   Network     (regression)
                                          │                 (512)
Tissue Mechanics ──────▶ Tissue Encoder ──┤                   │
(5 features)             (64-dim)         │                   ├──────▶ Optimal Head
                                          │                   │        (classification)
Hydrogel Props ────────▶ Hydrogel Enc ────┤                   │
(3 features)             (64-dim)         │                   └──────▶ Stress Head
                                          │                            (regression)
Treatment Config ──────▶ Treatment Enc ───┤
(5 features)             (64-dim)         │
                                          │
Polymer Category ──────▶ Category Embed ──┘
(8 categories)           (16-dim)
```

---

## 2. polyBERT Foundation Model

### 2.1 Model Overview

**polyBERT** is a DeBERTa-v2-based transformer pre-trained on 100 million polymer SMILES strings. It serves as the molecular encoder for HYDRA-BERT.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         polyBERT ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Base Architecture: microsoft/deberta-v2-xlarge                          │
│  Pre-training Data: 100M polymer SMILES (PolyInfo + synthetic)          │
│  Pre-training Task: Masked Language Modeling (MLM)                       │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                     DeBERTa-v2-xlarge                          │     │
│  │                                                                 │     │
│  │  Layers: 24 transformer blocks                                  │     │
│  │  Hidden Size: 1024                                              │     │
│  │  Attention Heads: 16                                            │     │
│  │  Intermediate Size: 4096                                        │     │
│  │  Max Position Embeddings: 512                                   │     │
│  │  Vocabulary Size: 256 (SMILES tokens)                          │     │
│  │                                                                 │     │
│  │  Special Features:                                              │     │
│  │  - Disentangled attention (content + position)                 │     │
│  │  - Enhanced mask decoder                                        │     │
│  │  - Relative position encoding                                   │     │
│  │                                                                 │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  Input:  Tokenized SMILES [CLS] C C ( = O ) N C C C ... [SEP]           │
│  Output: 600-dimensional [CLS] token embedding (pooled output)          │
│                                                                          │
│  Parameters: 86,087,936 (frozen during Stage 1)                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 SMILES Tokenization

polyBERT uses a character-level tokenizer optimized for chemical notation:

```python
class PolyBERTTokenizer:
    """
    SMILES tokenization for polyBERT.

    Vocabulary includes:
    - Standard SMILES characters: C, N, O, S, P, F, Cl, Br, I
    - Bond symbols: =, #, -, :
    - Ring closures: 1-9, %10-%99
    - Branches: (, )
    - Stereochemistry: /, \, @, @@
    - Charge: +, -
    - Special tokens: [CLS], [SEP], [PAD], [UNK]
    """

    VOCABULARY = {
        # Special tokens
        '[CLS]': 0, '[SEP]': 1, '[PAD]': 2, '[UNK]': 3,

        # Atoms
        'C': 4, 'N': 5, 'O': 6, 'S': 7, 'P': 8,
        'F': 9, 'Cl': 10, 'Br': 11, 'I': 12,
        'B': 13, 'Si': 14, 'Se': 15, 'Te': 16,

        # Bonds
        '-': 17, '=': 18, '#': 19, ':': 20, '.': 21,

        # Ring closures
        '1': 22, '2': 23, '3': 24, '4': 25, '5': 26,
        '6': 27, '7': 28, '8': 29, '9': 30,

        # Branches
        '(': 31, ')': 32,

        # Stereochemistry
        '/': 33, '\\': 34, '@': 35, '@@': 36,

        # Charge
        '+': 37, '-': 38,

        # Lowercase (aromatic)
        'c': 39, 'n': 40, 'o': 41, 's': 42,

        # Brackets
        '[': 43, ']': 44,

        # ... additional tokens up to 256
    }
```

### 2.3 Embedding Extraction

```python
def extract_polymer_embedding(smiles, polybert_model, tokenizer):
    """
    Extract 600-dimensional embedding from polyBERT.

    Uses [CLS] token pooled output as molecular representation.
    """

    # Tokenize SMILES
    tokens = tokenizer(
        smiles,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    # Forward pass through polyBERT
    with torch.no_grad():
        outputs = polybert_model(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask']
        )

    # Extract [CLS] token embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, 1024]

    # Apply projection to 600 dimensions
    # (This is done in the adapter layer in HYDRA-BERT)
    return cls_embedding
```

---

## 3. LoRA Adaptation Layer

### 3.1 LoRA Overview

**LoRA** (Low-Rank Adaptation) enables efficient fine-tuning by adding small trainable matrices to frozen transformer weights.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LoRA ADAPTATION                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Original Weight Matrix: W ∈ ℝ^(d × k)                                  │
│  LoRA Decomposition: ΔW = B × A where B ∈ ℝ^(d × r), A ∈ ℝ^(r × k)     │
│  Rank: r = 16 (much smaller than d or k)                                │
│                                                                          │
│  Forward Pass:                                                           │
│  ─────────────                                                           │
│                                                                          │
│  Original: h = W × x                                                     │
│  With LoRA: h = W × x + (α/r) × B × A × x                               │
│             = W × x + (α/r) × ΔW × x                                    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                                                                 │     │
│  │  Input x ──┬──────────────▶ W (frozen) ──────────┬──▶ Output h │     │
│  │            │                                     │              │     │
│  │            └──▶ A (r×k) ──▶ B (d×r) ──(α/r)──┘              │     │
│  │                  ↑              ↑                              │     │
│  │              trainable      trainable                          │     │
│  │                                                                 │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  Configuration:                                                          │
│  ───────────────                                                         │
│  - Rank (r): 16                                                          │
│  - Alpha (α): 32 (scaling factor)                                       │
│  - Target modules: query, key, value, output projections                 │
│  - Dropout: 0.1                                                          │
│                                                                          │
│  Parameters:                                                             │
│  ────────────                                                            │
│  Per attention layer: 4 × (2 × 1024 × 16) = 131,072                     │
│  Total (24 layers): 24 × 131,072 = 3,145,728                            │
│  With α=32: Effective parameters ~1.2M                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 LoRA Implementation

```python
from peft import LoraConfig, get_peft_model

class LoRAAdapter:
    """
    LoRA adaptation for polyBERT.
    """

    def __init__(self, polybert_model):
        self.config = LoraConfig(
            r=16,                           # LoRA rank
            lora_alpha=32,                  # Scaling factor
            target_modules=[
                'query_proj',               # Attention query
                'key_proj',                 # Attention key
                'value_proj',               # Attention value
                'dense'                     # Output projection
            ],
            lora_dropout=0.1,
            bias='none',                    # Don't train biases
            task_type='FEATURE_EXTRACTION'
        )

        self.model = get_peft_model(polybert_model, self.config)

    def get_trainable_params(self):
        """Count trainable parameters."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    def forward(self, input_ids, attention_mask):
        """Forward pass with LoRA adaptation."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state[:, 0, :]  # [CLS] embedding
```

### 3.3 SMILES Adapter Network

The SMILES adapter projects polyBERT output to the fusion dimensionality:

```python
class SMILESAdapter(nn.Module):
    """
    Adapter network to project polyBERT embeddings.

    polyBERT output (1024-dim) → SMILES embedding (256-dim)
    """

    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=256, dropout=0.1):
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(self, polybert_output):
        """
        Args:
            polybert_output: [batch, 1024] from polyBERT [CLS]

        Returns:
            smiles_embedding: [batch, 256]
        """
        return self.adapter(polybert_output)

# Parameters: 1024×512 + 512 + 512×256 + 256 = 655,616
```

---

## 4. Multi-Modal Feature Encoding

### 4.1 Feature Encoder Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MULTI-MODAL FEATURE ENCODING                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  NUMERICAL FEATURE ENCODER (19 features → 128 dim)                      │
│  ─────────────────────────────────────────────────                       │
│                                                                          │
│  ┌────────────────┐      ┌────────────────┐      ┌────────────────┐     │
│  │ Input (19)     │─────▶│ Hidden (64)    │─────▶│ Output (128)   │     │
│  │ BatchNorm      │      │ LayerNorm      │      │ LayerNorm      │     │
│  │                │      │ GELU           │      │ GELU           │     │
│  │                │      │ Dropout(0.1)   │      │                │     │
│  └────────────────┘      └────────────────┘      └────────────────┘     │
│                                                                          │
│  Feature Groups:                                                         │
│  ────────────────                                                        │
│  Group A (Patient Cardiac): baseline_LVEF, baseline_GLS, EDV, ESV,      │
│                             scar_fraction, bz_fraction [6 features]     │
│                                                                          │
│  Group B (Tissue Mechanics): bz_stress, healthy_stress, stress_conc,    │
│                              transmurality, wall_thickness [5 features] │
│                                                                          │
│  Group C (Hydrogel Props): hydrogel_E, t50, conductivity [3 features]   │
│                                                                          │
│  Group D (Treatment): thickness, coverage_onehot(4) [5 features]        │
│                                                                          │
│  CATEGORICAL EMBEDDING (polymer_category → 16 dim)                      │
│  ─────────────────────────────────────────────────                       │
│                                                                          │
│  ┌────────────────┐      ┌────────────────┐                              │
│  │ Category ID    │─────▶│ Embedding      │                              │
│  │ (0-8)          │      │ (9 × 16)       │                              │
│  │                │      │                │                              │
│  └────────────────┘      └────────────────┘                              │
│                                                                          │
│  Categories:                                                             │
│  ───────────                                                             │
│  0: GelMA, 1: PEGDA, 2: Alginate, 3: Chitosan, 4: HA,                   │
│  5: dECM, 6: Protein, 7: Conductive, 8: Unknown                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Property Encoder Implementation

```python
class PropertyEncoder(nn.Module):
    """
    Encode numerical features into dense representation.

    Processes 19 numerical features through a 2-layer MLP.
    """

    def __init__(self, input_dim=19, hidden_dim=64, output_dim=128, dropout=0.1):
        super().__init__()

        self.input_norm = nn.BatchNorm1d(input_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(self, numerical_features):
        """
        Args:
            numerical_features: [batch, 19]

        Returns:
            encoded: [batch, 128]
        """
        x = self.input_norm(numerical_features)
        return self.encoder(x)

# Parameters: 19×64 + 64 + 64 + 64×128 + 128 + 128 = 9,920
```

### 4.3 Category Embedding

```python
class CategoryEmbedding(nn.Module):
    """
    Learnable embedding for polymer categories.
    """

    def __init__(self, num_categories=9, embedding_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)

    def forward(self, category_ids):
        """
        Args:
            category_ids: [batch] integer tensor (0-8)

        Returns:
            embedded: [batch, 16]
        """
        return self.embedding(category_ids)

# Parameters: 9 × 16 = 144
```

---

## 5. Fusion Network

### 5.1 Feature Concatenation and Fusion

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FUSION NETWORK                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input Concatenation:                                                    │
│  ─────────────────────                                                   │
│                                                                          │
│  SMILES Embedding (256) ─┐                                               │
│                          │                                               │
│  Property Encoding (128) ┼─▶ CONCAT ─▶ [batch, 400]                     │
│                          │                                               │
│  Category Embedding (16) ┘                                               │
│                                                                          │
│  Fusion MLP:                                                             │
│  ───────────                                                             │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                                                                  │    │
│  │  Concat(400) ─▶ Linear(512) ─▶ LayerNorm ─▶ GELU ─▶ Dropout    │    │
│  │                                                                  │    │
│  │       ↓                                                          │    │
│  │                                                                  │    │
│  │  Linear(512) ─▶ LayerNorm ─▶ GELU ─▶ Dropout ─▶ Shared(512)    │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Output: Shared representation [batch, 512]                              │
│                                                                          │
│  This representation is then passed to task-specific heads.              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Fusion Implementation

```python
class FusionNetwork(nn.Module):
    """
    Fuse multi-modal representations into shared embedding.
    """

    def __init__(self, smiles_dim=256, property_dim=128, category_dim=16,
                 hidden_dim=512, output_dim=512, dropout=0.1):
        super().__init__()

        input_dim = smiles_dim + property_dim + category_dim  # 400

        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, smiles_embedding, property_encoding, category_embedding):
        """
        Args:
            smiles_embedding: [batch, 256]
            property_encoding: [batch, 128]
            category_embedding: [batch, 16]

        Returns:
            fused: [batch, 512]
        """
        concat = torch.cat([
            smiles_embedding,
            property_encoding,
            category_embedding
        ], dim=-1)  # [batch, 400]

        return self.fusion(concat)

# Parameters: 400×512 + 512 + 512 + 512×512 + 512 + 512 = 467,456
```

---

## 6. Multi-Task Prediction Heads

### 6.1 Head Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      MULTI-TASK PREDICTION HEADS                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                      Shared Representation [512]                         │
│                               │                                          │
│              ┌────────────────┼────────────────┐                        │
│              │                │                │                        │
│              ▼                ▼                ▼                        │
│  ┌───────────────────┐ ┌───────────────┐ ┌───────────────────┐         │
│  │   EF HEAD         │ │ OPTIMAL HEAD  │ │   STRESS HEAD     │         │
│  │   (Primary Reg)   │ │ (Primary Cls) │ │   (Auxiliary Reg) │         │
│  │                   │ │               │ │                   │         │
│  │   512 → 256       │ │   512 → 128   │ │   512 → 128       │         │
│  │   256 → 64        │ │   128 → 1     │ │   128 → 1         │         │
│  │   64 → 1          │ │   (sigmoid)   │ │                   │         │
│  │                   │ │               │ │                   │         │
│  │   Output: ΔEF     │ │ Output: p(opt)│ │ Output: ΔStress   │         │
│  │   Range: 0-30%    │ │ Range: 0-1    │ │ Range: 0-50%      │         │
│  └───────────────────┘ └───────────────┘ └───────────────────┘         │
│                                                                          │
│  Loss Functions:                                                         │
│  ────────────────                                                        │
│  EF Head: Huber Loss (smooth L1, β=2.0)                                 │
│  Optimal Head: Focal Loss (γ=2.0, pos_weight=3.17)                      │
│  Stress Head: MSE Loss                                                   │
│                                                                          │
│  Multi-Task Weighting:                                                   │
│  ─────────────────────                                                   │
│  Uncertainty-weighted combination (Kendall et al., 2018)                │
│  Each task has learnable log-variance parameter                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Head Implementations

```python
class EFHead(nn.Module):
    """
    Ejection Fraction improvement prediction head.
    Primary regression target.
    """

    def __init__(self, input_dim=512, dropout=0.1):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)

# Parameters: 512×256 + 256 + 256×64 + 64 + 64×1 + 1 = 148,033


class OptimalHead(nn.Module):
    """
    Optimal treatment classification head.
    Binary classification: is_optimal (yes/no)
    """

    def __init__(self, input_dim=512, dropout=0.1):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)  # Logits (apply sigmoid for probability)

# Parameters: 512×128 + 128 + 128×1 + 1 = 65,793


class StressHead(nn.Module):
    """
    Stress reduction prediction head.
    Auxiliary regression target.
    """

    def __init__(self, input_dim=512, dropout=0.1):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)

# Parameters: 512×128 + 128 + 128×1 + 1 = 65,793
```

### 6.3 Multi-Task Loss Function

```python
class HydraLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss.

    Based on: Kendall et al., "Multi-Task Learning Using Uncertainty
    to Weigh Losses for Scene Geometry and Semantics" (CVPR 2018)
    """

    def __init__(self):
        super().__init__()

        # Learnable log-variance parameters (homoscedastic uncertainty)
        self.log_var_ef = nn.Parameter(torch.zeros(1))
        self.log_var_optimal = nn.Parameter(torch.zeros(1))
        self.log_var_stress = nn.Parameter(torch.zeros(1))

    def forward(self, predictions, targets):
        """
        Compute uncertainty-weighted multi-task loss.
        """

        pred_ef = predictions['delta_EF']
        pred_optimal = predictions['is_optimal']
        pred_stress = predictions['stress_reduction']

        true_ef = targets['delta_EF']
        true_optimal = targets['is_optimal']
        true_stress = targets['stress_reduction']

        # Task 1: EF Regression - Huber Loss
        loss_ef = F.smooth_l1_loss(pred_ef, true_ef, beta=2.0)

        # Task 2: Optimal Classification - Focal Loss
        pos_weight = torch.tensor([3.17]).to(pred_optimal.device)  # Class imbalance
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_optimal, true_optimal.float(), pos_weight=pos_weight, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt) ** 2.0  # gamma = 2.0
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
            0.3 * (precision_stress * loss_stress + self.log_var_stress)  # Auxiliary weight
        )

        return total_loss, {
            'loss_ef': loss_ef.item(),
            'loss_optimal': loss_optimal.item(),
            'loss_stress': loss_stress.item(),
            'precision_ef': precision_ef.item(),
            'precision_optimal': precision_optimal.item(),
            'precision_stress': precision_stress.item()
        }
```

---

## 7. PPO Policy Network (Stage 2)

### 7.1 Policy Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PPO POLICY NETWORK (STAGE 2)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STATE ENCODER (reuses Stage 1 components)                               │
│  ──────────────────────────────────────────                              │
│                                                                          │
│  Patient Features ──▶ Property Encoder ──▶ State Embedding [128]        │
│                                                                          │
│  ACTOR NETWORK (Policy)                                                  │
│  ──────────────────────                                                  │
│                                                                          │
│  State [128] ──▶ Hidden [256] ──▶ Hidden [256] ──┬──▶ Polymer Logits [24]│
│                                                   │                       │
│                                                   ├──▶ Stiffness μ,σ [2] │
│                                                   │                       │
│                                                   ├──▶ Degradation μ,σ [2]│
│                                                   │                       │
│                                                   ├──▶ Conductivity μ,σ[2]│
│                                                   │                       │
│                                                   ├──▶ Thickness μ,σ [2] │
│                                                   │                       │
│                                                   └──▶ Coverage Logits [4]│
│                                                                          │
│  CRITIC NETWORK (Value Function)                                         │
│  ───────────────────────────────                                         │
│                                                                          │
│  State [128] ──▶ Hidden [256] ──▶ Hidden [256] ──▶ Value [1]            │
│                                                                          │
│  ACTION SPACE (Hybrid Discrete + Continuous)                             │
│  ────────────────────────────────────────────                            │
│                                                                          │
│  Discrete:                                                               │
│  - Polymer selection: Categorical(24)                                    │
│  - Coverage pattern: Categorical(4)                                      │
│                                                                          │
│  Continuous (Normal distribution):                                       │
│  - Stiffness: N(μ_E, σ_E²), range [0.5, 50] kPa                         │
│  - Degradation: N(μ_t, σ_t²), range [1, 180] days                       │
│  - Conductivity: N(μ_σ, σ_σ²), range [0.001, 1.0] S/m                   │
│  - Thickness: N(μ_d, σ_d²), range [0.1, 2.0] mm                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 PPO Implementation

```python
class PPOPolicy(nn.Module):
    """
    Proximal Policy Optimization policy for hydrogel design.
    """

    def __init__(self, state_dim=128, hidden_dim=256):
        super().__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Actor heads (policy)
        self.polymer_head = nn.Linear(hidden_dim, 24)    # Discrete: 24 polymers
        self.coverage_head = nn.Linear(hidden_dim, 4)    # Discrete: 4 patterns

        # Continuous action means
        self.stiffness_mean = nn.Linear(hidden_dim, 1)
        self.degradation_mean = nn.Linear(hidden_dim, 1)
        self.conductivity_mean = nn.Linear(hidden_dim, 1)
        self.thickness_mean = nn.Linear(hidden_dim, 1)

        # Continuous action log-stds (learnable)
        self.stiffness_logstd = nn.Parameter(torch.zeros(1))
        self.degradation_logstd = nn.Parameter(torch.zeros(1))
        self.conductivity_logstd = nn.Parameter(torch.zeros(1))
        self.thickness_logstd = nn.Parameter(torch.zeros(1))

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        """
        Forward pass through policy.

        Returns action distributions and value estimate.
        """
        features = self.feature_extractor(state)

        # Discrete action distributions
        polymer_logits = self.polymer_head(features)
        coverage_logits = self.coverage_head(features)

        polymer_dist = Categorical(logits=polymer_logits)
        coverage_dist = Categorical(logits=coverage_logits)

        # Continuous action distributions
        stiffness_mean = torch.sigmoid(self.stiffness_mean(features)) * 49.5 + 0.5
        degradation_mean = torch.sigmoid(self.degradation_mean(features)) * 179 + 1
        conductivity_mean = torch.sigmoid(self.conductivity_mean(features)) * 0.999 + 0.001
        thickness_mean = torch.sigmoid(self.thickness_mean(features)) * 1.9 + 0.1

        stiffness_dist = Normal(stiffness_mean, torch.exp(self.stiffness_logstd))
        degradation_dist = Normal(degradation_mean, torch.exp(self.degradation_logstd))
        conductivity_dist = Normal(conductivity_mean, torch.exp(self.conductivity_logstd))
        thickness_dist = Normal(thickness_mean, torch.exp(self.thickness_logstd))

        # Value estimate
        value = self.critic(state)

        return {
            'polymer_dist': polymer_dist,
            'coverage_dist': coverage_dist,
            'stiffness_dist': stiffness_dist,
            'degradation_dist': degradation_dist,
            'conductivity_dist': conductivity_dist,
            'thickness_dist': thickness_dist,
            'value': value
        }

    def sample_action(self, state):
        """Sample action from policy."""
        dists = self.forward(state)

        polymer = dists['polymer_dist'].sample()
        coverage = dists['coverage_dist'].sample()
        stiffness = dists['stiffness_dist'].sample()
        degradation = dists['degradation_dist'].sample()
        conductivity = dists['conductivity_dist'].sample()
        thickness = dists['thickness_dist'].sample()

        # Compute log probabilities
        log_prob = (
            dists['polymer_dist'].log_prob(polymer) +
            dists['coverage_dist'].log_prob(coverage) +
            dists['stiffness_dist'].log_prob(stiffness) +
            dists['degradation_dist'].log_prob(degradation) +
            dists['conductivity_dist'].log_prob(conductivity) +
            dists['thickness_dist'].log_prob(thickness)
        )

        return {
            'polymer': polymer,
            'coverage': coverage,
            'stiffness': stiffness,
            'degradation': degradation,
            'conductivity': conductivity,
            'thickness': thickness,
            'log_prob': log_prob,
            'value': dists['value']
        }
```

---

## 8. Design Generation Engine (Stage 3)

### 8.1 Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DESIGN GENERATION ENGINE (STAGE 3)                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CANDIDATE GENERATOR                                                     │
│  ────────────────────                                                    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                                                                  │    │
│  │  Parameter Grid:                                                 │    │
│  │  - 24 polymers                                                   │    │
│  │  - 20 stiffness values (log-spaced 0.5-50 kPa)                  │    │
│  │  - 15 degradation values (log-spaced 1-180 days)                │    │
│  │  - 10 conductivity values (log-spaced 0.001-1.0 S/m)            │    │
│  │  - 8 thickness values (linear 0.1-2.0 mm)                       │    │
│  │  - 4 coverage patterns                                           │    │
│  │                                                                  │    │
│  │  Expansion Method: Latin Hypercube Sampling + Interpolation     │    │
│  │  Target: 10,000,000 candidates per patient                      │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  BATCH INFERENCE                                                         │
│  ───────────────                                                         │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                                                                  │    │
│  │  HYDRA-BERT Inference:                                          │    │
│  │  - Batch size: 4096                                             │    │
│  │  - Mixed precision (FP16)                                       │    │
│  │  - Multi-GPU distribution                                        │    │
│  │                                                                  │    │
│  │  Outputs per candidate:                                          │    │
│  │  - Predicted ΔEF                                                │    │
│  │  - Predicted stress reduction                                    │    │
│  │  - Predicted strain normalization                                │    │
│  │  - is_optimal probability                                        │    │
│  │                                                                  │    │
│  │  MC Dropout: 10 forward passes for uncertainty                   │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  HIERARCHICAL FILTER                                                     │
│  ───────────────────                                                     │
│                                                                          │
│  10M ─▶ 100K ─▶ 10K ─▶ 1K ─▶ 100 ─▶ 10 ─▶ 1                           │
│    A      B       C      D       E       F                               │
│                                                                          │
│  A: Therapeutic threshold (ΔEF≥3%, stress≥10%)                          │
│  B: Percentile ranking (top 10%)                                         │
│  C: Diversity + quality filter                                           │
│  D: FEBio pre-screening (10 cycles)                                      │
│  E: Full FEBio + OpenCarp simulation                                    │
│  F: Multi-criteria Pareto optimization                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Simulation Integration

```python
class SimulationValidator:
    """
    FEBio and OpenCarp simulation validation.
    """

    def __init__(self):
        self.febio = FEBioInterface()
        self.opencarp = OpenCarpInterface()

    def validate_design(self, patient, design, full_simulation=False):
        """
        Validate hydrogel design with physics simulations.

        Args:
            patient: Patient cardiac data
            design: Hydrogel design parameters
            full_simulation: If True, run full 100-cycle simulation

        Returns:
            Validation results including simulated outcomes
        """

        # FEBio mechanical simulation
        n_cycles = 100 if full_simulation else 10

        febio_result = self.febio.simulate(
            patient_geometry=patient['geometry'],
            hydrogel_stiffness=design['stiffness'],
            hydrogel_coverage=design['coverage'],
            n_cycles=n_cycles
        )

        if not febio_result['converged']:
            return {'valid': False, 'reason': 'FEBio diverged'}

        # OpenCarp electrophysiology
        opencarp_result = self.opencarp.simulate(
            patient_geometry=patient['geometry'],
            hydrogel_conductivity=design['conductivity'],
            duration_ms=5000 if full_simulation else 1000
        )

        # Compile results
        return {
            'valid': True,
            'delta_EF': febio_result['delta_EF'],
            'stress_reduction': febio_result['stress_reduction'],
            'strain_normalization': febio_result['strain_normalization'],
            'conduction_velocity': opencarp_result['cv'],
            'arrhythmia_risk': opencarp_result['arrhythmia_score'],
            'mechanical_stability': febio_result['stability']
        }
```

---

## 9. Complete Parameter Count

### 9.1 Stage 1 Parameters

| Component | Parameters | Trainable |
|-----------|------------|-----------|
| polyBERT (base) | 86,087,936 | No (frozen) |
| LoRA Adapters | 1,179,648 | Yes |
| SMILES Adapter | 655,616 | Yes |
| Property Encoder | 9,920 | Yes |
| Category Embedding | 144 | Yes |
| Fusion Network | 467,456 | Yes |
| EF Head | 148,033 | Yes |
| Optimal Head | 65,793 | Yes |
| Stress Head | 65,793 | Yes |
| Loss Parameters | 3 | Yes |
| **Stage 1 Total** | **88,680,342** | **2,592,406** |

### 9.2 Stage 2 Parameters (PPO)

| Component | Parameters | Trainable |
|-----------|------------|-----------|
| Feature Extractor | 99,328 | Yes |
| Polymer Head | 6,168 | Yes |
| Coverage Head | 1,028 | Yes |
| Continuous Mean Heads | 1,028 | Yes |
| Continuous LogStd | 4 | Yes |
| Critic Network | 99,585 | Yes |
| **Stage 2 Total** | **207,141** | **207,141** |

### 9.3 Complete System Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      COMPLETE PARAMETER SUMMARY                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STAGE 1: Supervised Learning                                            │
│  ────────────────────────────                                            │
│  Frozen (polyBERT):      86,087,936 params (96.8%)                      │
│  Trainable:               2,592,406 params ( 2.9%)                      │
│  Total Stage 1:          88,680,342 params                               │
│                                                                          │
│  STAGE 2: Reinforcement Learning (PPO)                                   │
│  ──────────────────────────────────────                                  │
│  Trainable (Policy):        207,141 params                               │
│                                                                          │
│  COMPLETE SYSTEM                                                         │
│  ───────────────                                                         │
│  Total Parameters:       88,887,483                                      │
│  Trainable Parameters:    2,799,547 (3.1%)                              │
│  Frozen Parameters:      86,087,936 (96.9%)                              │
│                                                                          │
│  Memory Requirements (FP16):                                             │
│  - Model weights: ~170 MB                                                │
│  - Optimizer states (AdamW): ~340 MB                                     │
│  - Activations (batch=256): ~12 GB                                       │
│  - Total GPU memory: ~15 GB                                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Forward Pass Flow

### 10.1 Complete Forward Pass

```python
class HydraBERT(nn.Module):
    """
    Complete HYDRA-BERT model for Stage 1.
    """

    def __init__(self, config):
        super().__init__()

        # polyBERT with LoRA
        self.polybert = load_polybert_with_lora(config.polybert_path)

        # SMILES adapter
        self.smiles_adapter = SMILESAdapter(
            input_dim=1024,
            hidden_dim=512,
            output_dim=256
        )

        # Property encoder
        self.property_encoder = PropertyEncoder(
            input_dim=19,
            hidden_dim=64,
            output_dim=128
        )

        # Category embedding
        self.category_embedding = CategoryEmbedding(
            num_categories=9,
            embedding_dim=16
        )

        # Fusion network
        self.fusion = FusionNetwork(
            smiles_dim=256,
            property_dim=128,
            category_dim=16,
            hidden_dim=512,
            output_dim=512
        )

        # Prediction heads
        self.ef_head = EFHead(input_dim=512)
        self.optimal_head = OptimalHead(input_dim=512)
        self.stress_head = StressHead(input_dim=512)

    def forward(self, smiles_tokens, numerical_features, category_ids):
        """
        Complete forward pass.

        Args:
            smiles_tokens: Dict with input_ids, attention_mask [batch, seq_len]
            numerical_features: [batch, 19]
            category_ids: [batch]

        Returns:
            Dict with predictions: delta_EF, is_optimal, stress_reduction
        """

        # Step 1: polyBERT encoding with LoRA
        polybert_output = self.polybert(
            input_ids=smiles_tokens['input_ids'],
            attention_mask=smiles_tokens['attention_mask']
        ).last_hidden_state[:, 0, :]  # [batch, 1024]

        # Step 2: SMILES adaptation
        smiles_embedding = self.smiles_adapter(polybert_output)  # [batch, 256]

        # Step 3: Property encoding
        property_encoding = self.property_encoder(numerical_features)  # [batch, 128]

        # Step 4: Category embedding
        category_embed = self.category_embedding(category_ids)  # [batch, 16]

        # Step 5: Fusion
        fused = self.fusion(smiles_embedding, property_encoding, category_embed)  # [batch, 512]

        # Step 6: Task-specific predictions
        delta_ef = self.ef_head(fused)          # [batch]
        is_optimal = self.optimal_head(fused)   # [batch] (logits)
        stress_red = self.stress_head(fused)    # [batch]

        return {
            'delta_EF': delta_ef,
            'is_optimal': is_optimal,
            'stress_reduction': stress_red,
            'embedding': fused  # For PPO state encoding
        }
```

### 10.2 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW DIAGRAM                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  INPUTS                                                                              │
│  ──────                                                                              │
│                                                                                      │
│  SMILES: "CC(=C)C(=O)NCCCCCC..." ──▶ Tokenizer ──▶ [1, 128] tokens                 │
│                                                                                      │
│  Numerical: [32.4, -14.8, 198, 134, 18.6, 9.4, 45, 12, 2.1,                         │
│              8.5, 62, 0.15, 0.82, 0, 0, 1, 0, 0.5, 0.1] ──▶ [1, 19] tensor         │
│                                                                                      │
│  Category: "GelMA" ──▶ 0 ──▶ [1] tensor                                             │
│                                                                                      │
│  PROCESSING                                                                          │
│  ──────────                                                                          │
│                                                                                      │
│  polyBERT:  [1, 128] ──▶ 24 transformer layers ──▶ [1, 128, 1024] ──▶ [1, 1024]    │
│                              (LoRA adapted)            (all tokens)     ([CLS])     │
│                                                                                      │
│  SMILES Adapter: [1, 1024] ──▶ [1, 512] ──▶ [1, 256]                               │
│                                                                                      │
│  Property Encoder: [1, 19] ──▶ [1, 64] ──▶ [1, 128]                                │
│                                                                                      │
│  Category Embed: [1] ──▶ lookup ──▶ [1, 16]                                        │
│                                                                                      │
│  Fusion: concat([256, 128, 16]) = [1, 400] ──▶ [1, 512] ──▶ [1, 512]               │
│                                                                                      │
│  OUTPUTS                                                                             │
│  ───────                                                                             │
│                                                                                      │
│  EF Head: [1, 512] ──▶ [1, 256] ──▶ [1, 64] ──▶ [1] ──▶ 9.2 (ΔEF%)                │
│                                                                                      │
│  Optimal Head: [1, 512] ──▶ [1, 128] ──▶ [1] ──▶ sigmoid ──▶ 0.87 (prob)           │
│                                                                                      │
│  Stress Head: [1, 512] ──▶ [1, 128] ──▶ [1] ──▶ 28.5 (stress reduction %)          │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Conclusion

The HYDRA-BERT architecture combines molecular representation learning (polyBERT), efficient fine-tuning (LoRA), multi-modal fusion, and multi-task learning into a cohesive system for cardiac hydrogel optimization. The three-stage pipeline progressively refines hydrogel designs from 447,480 training samples through reinforcement learning to patient-specific optimal designs validated by physics simulations.

**Key Architectural Innovations:**
1. LoRA-adapted polyBERT for polymer encoding with 97% parameter efficiency
2. Uncertainty-weighted multi-task learning for joint regression and classification
3. Hybrid discrete-continuous action space PPO for design optimization
4. Hierarchical filtering with physics simulation validation
