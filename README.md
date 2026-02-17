# HYDRA-BERT: HYdrogel Domain Refined Adaptation of BERT

A deep learning pipeline for patient-specific injectable hydrogel design optimization and development in cardiac regeneration therapy, in aim to prevent re-ocurring episodes of Myocardial Infarction.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Key Features

- **10 Million Design Generation**: Comprehensive sampling across 24 polymer types
- **Multi-Physics Validation**: FEBio mechanics + OpenCarp electrophysiology simulations
- **Patient-Specific Optimization**: Tailored designs based on individual cardiac pathophysiology
- **Therapeutic Validation**: All designs verified against clinical thresholds

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HYDRA-BERT DESIGN OPTIMIZATION PIPELINE                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: DESIGN GENERATION (10M per patient, GPU-accelerated)               │
│  ├─ Sample from 24 polymer types × continuous parameter space               │
│  ├─ Evaluate using HYDRA-BERT neural network predictions                    │
│  └─ Output: Top 100 designs ranked by predicted therapeutic score           │
│                                                                             │
│  STEP 2: FEBio MECHANICAL SIMULATION (CPU-parallel)                         │
│  ├─ Run finite element simulations for each top 100 design                  │
│  ├─ Place hydrogel in patient-specific infarct zone                         │
│  └─ Extract: Wall stress, strain, LVEF                                      │
│                                                                             │
│  STEP 3: OpenCarp EP SIMULATION (CPU-parallel)                              │
│  ├─ Run electrophysiology simulations for each design                       │
│  ├─ Apply conductive hydrogel properties                                    │
│  └─ Extract: Conduction velocity, APD, arrhythmia risk                      │
│                                                                             │
│  STEP 4: OPTIMAL SELECTION                                                  │
│  ├─ Combine FEBio + OpenCarp metrics                                        │
│  ├─ Rank by combined therapeutic score                                      │
│  └─ Output: Single BEST design per patient                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Therapeutic Thresholds

All selected designs must meet these criteria:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| **ΔEF** | ≥ 5% | Ejection fraction improvement |
| **Wall Stress Reduction** | ≥ 25% | Peak systolic stress reduction |
| **Strain Normalization** | ≥ 15% | Strain uniformity improvement |

---

## Quick Start

### Prerequisites

```bash
# Python 3.10+ with PyTorch 2.0+
conda create -n hydra_bert python=3.10 -y
conda activate hydra_bert
pip install -r requirements.txt
```

### Run Complete Pipeline

```bash
# Process all 10 patients with full resources (16 GPUs, 96 CPUs)
python scripts/pipeline/run_full_pipeline.py --all --gpus 16 --cpus 96

# Process single patient
python scripts/pipeline/run_full_pipeline.py --patient SCD0000101
```

### Run Individual Steps

```bash
# Step 1: Generate 10M designs per patient
python scripts/pipeline/generate_10M_designs.py --all --gpus 16

# Step 2: Run FEBio mechanical simulations
python scripts/pipeline/run_febio_simulations.py --all --n-cpus 96

# Step 3: Run OpenCarp EP simulations
python scripts/pipeline/run_opencarp_simulations.py --all --n-cpus 96

# Step 4: Select optimal design from results
python scripts/pipeline/select_optimal_design.py --all
```

---

## Polymer Library (24 Types)

| Category | Polymers |
|----------|----------|
| **Protein-Modified** | GelMA (3%, 5%, 7%, 10%) |
| **Conductive GelMA** | BioIL, Polypyrrole, rGO, MXene |
| **Synthetic** | PEGDA (575, 700, 3400 Da) |
| **Polysaccharide** | Alginate, Chitosan variants |
| **Glycosaminoglycan** | HA_acellular, HA_ECM, MeHA |
| **Decellularized ECM** | VentriGel, cardiac dECM |
| **Protein** | Fibrin, Gelatin |
| **Conductive** | PEDOT:PSS |

---

## Project Structure

```
HYDRA-BERT-FINAL/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
│
├── scripts/
│   ├── pipeline/                       # MAIN PIPELINE SCRIPTS
│   │   ├── run_full_pipeline.py       # Master orchestration
│   │   ├── generate_10M_designs.py    # 10M design generation
│   │   ├── run_febio_simulations.py   # FEBio simulations
│   │   ├── run_opencarp_simulations.py # OpenCarp simulations
│   │   └── select_optimal_design.py   # Optimal selection
│   ├── stage1/                         # HYDRA-BERT training
│   ├── stage2/                         # PPO RL training
│   └── stage3/                         # Legacy generation
│
├── hydra_bert/                         # Core model package
│   ├── models/                         # Neural network architectures
│   ├── data/                           # Dataset handling
│   └── stage3/                         # Generation modules
│
├── docs/
│   ├── PIPELINE_DOCUMENTATION.md      # Complete pipeline docs
│   ├── SIMULATION_GUIDE.md            # FEBio/OpenCarp setup
│   └── TECHNICAL_DOCUMENTATION.md     # Technical details
│
├── configs/                            # Configuration files
│
└── results/
    ├── design_generation/              # Generated 10M designs
    ├── febio_simulations/              # FEBio results
    ├── opencarp_simulations/           # OpenCarp results
    └── final_optimal_designs/          # FINAL OPTIMAL DESIGNS
        ├── FINAL_OPTIMAL_DESIGNS_REPORT.md
        ├── final_optimal_designs_summary.csv
        └── {patient_id}/optimal_design.csv
```

---

## Output

### Final Results

After running the complete pipeline:

```bash
# View summary report
cat results/final_optimal_designs/FINAL_OPTIMAL_DESIGNS_REPORT.md

# View all optimal designs
cat results/final_optimal_designs/final_optimal_designs_summary.csv

# View specific patient
cat results/final_optimal_designs/SCD0000101/optimal_design.csv
```

### Example Results

| Patient | Optimal Polymer | ΔEF | Stress Red | CV Imp | Score | Status |
|---------|----------------|-----|------------|--------|-------|--------|
| SCD0000101 | PEGDA_3400 | +9.5% | 30.0% | 26.0% | 171 | THERAPEUTIC |
| SCD0000201 | GelMA_rGO | +8.8% | 28.5% | 22.3% | 165 | THERAPEUTIC |
| SCD0000401 | Chitosan_HA | +9.2% | 29.8% | 18.5% | 162 | THERAPEUTIC |
| ... | ... | ... | ... | ... | ... | ... |

---

## Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPUs | 1× NVIDIA GPU | 16× A100 40GB |
| CPU Cores | 8 | 96 |
| RAM | 32GB | 256GB |
| Storage | 100GB | 500GB |

### Estimated Runtime

| Configuration | All 10 Patients |
|---------------|-----------------|
| 16 GPUs, 96 CPUs | ~1 hour |
| 4 GPUs, 24 CPUs | ~4 hours |
| 1 GPU, 8 CPUs | ~15 hours |

---

## Documentation

- **[Pipeline Documentation](docs/PIPELINE_DOCUMENTATION.md)** - Complete pipeline details
- **[Simulation Guide](docs/SIMULATION_GUIDE.md)** - FEBio/OpenCarp setup
- **[Technical Documentation](docs/TECHNICAL_DOCUMENTATION.md)** - Architecture & algorithms

---

## Citation

```bibtex
@software{hydra_bert_2026,
  author={Krishiv Potluri},
  year={2026},
  url={https://github.com/your-org/HYDRA-BERT}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) for details.
