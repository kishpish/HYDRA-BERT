# HYDRA-BERT Documentation Index

## Overview

This document provides a index of all documentation files for the HYDRA-BERT project, organized by category and relevance to research paper sections.

---

## Complete Documentation List

### Core Pipeline Documentation

| Document | Path | Description | Paper Section |
|----------|------|-------------|---------------|
| **Stage 1: Supervised Learning** | `docs/STAGE1_SUPERVISED_LEARNING.md` | Complete documentation of polyBERT fine-tuning with LoRA, multi-task learning, training procedures | Methods: Model Architecture, Training |
| **Stage 2: Reinforcement Learning** | `docs/STAGE2_REINFORCEMENT_LEARNING.md` | PPO algorithm, policy network, reward design, GAE advantage estimation | Methods: RL Optimization |
| **Stage 3: Design Generation** | `docs/STAGE3_DESIGN_GENERATION.md` | 10M design generation, hierarchical filtering, FEBio/OpenCarp integration, optimal selection | Methods: Design Generation |

### Data Documentation

| Document | Path | Description | Paper Section |
|----------|------|-------------|---------------|
| **Data Preprocessing** | `docs/DATA_PREPROCESSING.md` | Feature engineering, SMILES tokenization, train/val/test splitting, quality assurance | Methods: Data Preparation |
| **Polymer Curation** | `docs/POLYMER_CURATION_METHODOLOGY.md` | 565→24 polymer filtering process, 5-stage curation pipeline | Methods: Polymer Library |

### Architecture Documentation

| Document | Path | Description | Paper Section |
|----------|------|-------------|---------------|
| **Complete Architecture** | `docs/COMPLETE_ARCHITECTURE.md` | Full system architecture, component dimensions, parameter counts, forward pass | Methods: Architecture |

### Validation Documentation

| Document | Path | Description | Paper Section |
|----------|------|-------------|---------------|
| **Validation Methodology** | `docs/VALIDATION_METHODOLOGY.md` | ML validation, physics simulation validation, therapeutic assessment, statistics | Methods: Validation; Results |

### Supplementary Data

| Document | Path | Description | Paper Section |
|----------|------|-------------|---------------|
| **Initial Polymer Library** | `data/supplementary/initial_polymer_library_565.csv` | Complete 565 polymers with filtering outcomes | Supplementary Materials |
| **Final Optimal Designs** | `results/final_optimal_designs/` | Patient-specific optimal designs and metrics | Results |



## Quick Reference: Key Statistics

### Dataset
- Total samples: 447,480
- Unique polymers: 24 (from 565 initial)
- Patients: 60 (10 real + 50 synthetic)
- Features: 19 numerical + SMILES + category

### Model
- polyBERT base: 86M parameters (frozen)
- LoRA adapters: 1.2M parameters
- Total trainable: 2.8M parameters

### Performance
- ΔEF MAE: 0.82%
- R²: 0.87
- AUROC: 0.94
- F1: 0.81

### Therapeutic Outcomes (n=10 patients)
- Mean ΔEF: 9.1 ± 0.4%
- Mean stress reduction: 30.1 ± 0.4%
- Therapeutic rate: 100%
- Designs evaluated: 100 million
