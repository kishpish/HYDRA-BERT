# HYDRA-BERT Documentation Index

## Overview

This document provides a comprehensive index of all documentation files for the HYDRA-BERT project, organized by category and relevance to research paper sections.

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

---

## Documentation by Research Paper Section

### Abstract
- Summary of therapeutic outcomes from `VALIDATION_METHODOLOGY.md`
- Key metrics: ΔEF 9.1%, stress reduction 30.1%, 100% therapeutic rate

### Introduction
- Background on cardiac hydrogels (literature references in `POLYMER_CURATION_METHODOLOGY.md`)
- Problem statement and motivation

### Methods

#### 2.1 Polymer Library
- `POLYMER_CURATION_METHODOLOGY.md` - Complete curation process
- `data/supplementary/initial_polymer_library_565.csv` - Source data

#### 2.2 Patient Data
- `DATA_PREPROCESSING.md` - Patient data processing
- Section 3: Patient Data Processing

#### 2.3 Model Architecture
- `COMPLETE_ARCHITECTURE.md` - Full architecture details
- `STAGE1_SUPERVISED_LEARNING.md` - polyBERT + LoRA specifics

#### 2.4 Training Procedure
- `STAGE1_SUPERVISED_LEARNING.md` - Supervised learning
- `STAGE2_REINFORCEMENT_LEARNING.md` - PPO training

#### 2.5 Design Generation
- `STAGE3_DESIGN_GENERATION.md` - 10M design pipeline
- Hierarchical filtering details

#### 2.6 Validation
- `VALIDATION_METHODOLOGY.md` - All validation protocols
- FEBio and OpenCarp simulation details

### Results

#### 3.1 Model Performance
- `VALIDATION_METHODOLOGY.md` - Section 2: ML Validation
- Cross-validation results

#### 3.2 Therapeutic Outcomes
- `VALIDATION_METHODOLOGY.md` - Section 7: Therapeutic Assessment
- `results/final_optimal_designs/FINAL_OPTIMAL_DESIGNS_REPORT.md`

#### 3.3 Patient-Specific Optimization
- `results/final_optimal_designs/final_optimal_designs_summary.csv`
- Individual patient outcomes

### Discussion
- Comparison with literature (polymer efficacy data)
- Limitations and future work

### Supplementary Materials
- `data/supplementary/initial_polymer_library_565.csv`
- Extended validation data
- Detailed parameter tables

---

## Key Figures

Located in `figures/`:

| Figure | File | Description | Paper Figure |
|--------|------|-------------|--------------|
| 1 | `figure_pipeline_overview.png` | 3-stage HYDRA-BERT pipeline | Fig. 1 |
| 2 | `figure_polymer_library.png` | 24 curated polymers visualization | Fig. 2 |
| 3 | `figure_patient_outcomes.png` | 10-patient therapeutic outcomes | Fig. 3 |
| 4 | `figure_polymer_curation.png` | 565→24 curation pipeline | Fig. 4 |
| 5 | `figure_polymer_diversity.png` | Polymer category distribution | Fig. S1 |
| 6 | `figure_febio_validation.png` | FEBio simulation results | Fig. 5 |
| 7 | `figure_opencarp_validation.png` | OpenCarp electrophysiology | Fig. 6 |
| 8 | `figure_model_performance.png` | ML metrics and curves | Fig. S2 |
| 9 | `figure_therapeutic_thresholds.png` | Therapeutic classification | Fig. 7 |
| 10 | `figure_optimal_polymer_distribution.png` | Selected polymers | Fig. S3 |

---

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

---

## File Sizes and Formats

```
docs/
├── STAGE1_SUPERVISED_LEARNING.md      (~25 KB, ~700 lines)
├── STAGE2_REINFORCEMENT_LEARNING.md   (~22 KB, ~600 lines)
├── STAGE3_DESIGN_GENERATION.md        (~35 KB, ~900 lines)
├── DATA_PREPROCESSING.md              (~28 KB, ~750 lines)
├── COMPLETE_ARCHITECTURE.md           (~32 KB, ~850 lines)
├── VALIDATION_METHODOLOGY.md          (~30 KB, ~800 lines)
├── POLYMER_CURATION_METHODOLOGY.md    (~18 KB, ~500 lines)
└── DOCUMENTATION_INDEX.md             (this file)

data/supplementary/
└── initial_polymer_library_565.csv    (~45 KB, 565 rows)

figures/
├── figure_pipeline_overview.png       (~1.2 MB, 300 DPI)
├── figure_polymer_library.png         (~800 KB, 300 DPI)
├── figure_patient_outcomes.png        (~900 KB, 300 DPI)
├── figure_polymer_curation.png        (~600 KB, 300 DPI)
├── figure_polymer_diversity.png       (~500 KB, 300 DPI)
├── figure_febio_validation.png        (~1.1 MB, 300 DPI)
├── figure_opencarp_validation.png     (~1.0 MB, 300 DPI)
├── figure_model_performance.png       (~700 KB, 300 DPI)
├── figure_therapeutic_thresholds.png  (~600 KB, 300 DPI)
└── figure_optimal_polymer_distribution.png (~400 KB, 300 DPI)
```

---

## Research Paper Writing Guide

### Using This Documentation

1. **Methods Section**: Draw from technical documentation files
   - Copy architecture diagrams from `COMPLETE_ARCHITECTURE.md`
   - Use equations from `STAGE1_SUPERVISED_LEARNING.md` (loss functions)
   - Reference polymer curation from `POLYMER_CURATION_METHODOLOGY.md`

2. **Results Section**: Use validation documentation
   - Tables from `VALIDATION_METHODOLOGY.md`
   - Patient outcomes from `results/final_optimal_designs/`
   - Statistical analysis already computed

3. **Figures**: All publication-ready in `figures/`
   - 300 DPI PNG and PDF formats
   - Consistent styling and labeling
   - Ready for journal submission

4. **Supplementary Materials**: Extended documentation
   - Full polymer library CSV
   - Detailed parameter tables
   - Extended validation results

### Citation Notes

Key references used in methodology:
- polyBERT: Kuenneth et al., 2023
- LoRA: Hu et al., 2022
- Uncertainty weighting: Kendall et al., 2018
- PPO: Schulman et al., 2017
- FEBio: Maas et al., 2012
- OpenCarp: Vigmond et al., 2008
- Holzapfel-Ogden: Holzapfel & Ogden, 2009
- ten Tusscher-Panfilov: ten Tusscher & Panfilov, 2006

---

## Document Status

| Document | Status | Last Updated |
|----------|--------|--------------|
| STAGE1_SUPERVISED_LEARNING.md | Complete | 2025-02-09 |
| STAGE2_REINFORCEMENT_LEARNING.md | Complete | 2025-02-09 |
| STAGE3_DESIGN_GENERATION.md | Complete | 2025-02-09 |
| DATA_PREPROCESSING.md | Complete | 2025-02-09 |
| COMPLETE_ARCHITECTURE.md | Complete | 2025-02-09 |
| VALIDATION_METHODOLOGY.md | Complete | 2025-02-09 |
| POLYMER_CURATION_METHODOLOGY.md | Complete | 2025-02-09 |
| DOCUMENTATION_INDEX.md | Complete | 2025-02-09 |

All documentation is complete and ready for research paper preparation.
