# HYDRA-BERT: Complete Patient-Specific Design Pipeline Documentation

## Overview

The HYDRA-BERT pipeline is a comprehensive system for generating and validating patient-specific injectable hydrogel designs for cardiac regeneration after myocardial infarction.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HYDRA-BERT DESIGN OPTIMIZATION PIPELINE                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: DESIGN GENERATION (GPU-Accelerated)                         │   │
│  │ ├─ Generate 10,000,000 unique hydrogel designs per patient          │   │
│  │ ├─ Sample from 24 polymer types × continuous parameter space        │   │
│  │ ├─ Evaluate using HYDRA-BERT neural network predictions             │   │
│  │ └─ Output: Top 100 designs ranked by predicted therapeutic score    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: FEBio MECHANICAL SIMULATION (CPU-Parallel)                  │   │
│  │ ├─ Run finite element simulations for each top 100 design           │   │
│  │ ├─ Place hydrogel in patient-specific infarct zone mesh             │   │
│  │ ├─ Holzapfel-Ogden anisotropic cardiac material model               │   │
│  │ └─ Extract: Wall stress, strain, displacement, LVEF                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: OpenCarp EP SIMULATION (CPU-Parallel)                       │   │
│  │ ├─ Run electrophysiology simulations for each design                │   │
│  │ ├─ ten Tusscher-Panfilov ionic model                                │   │
│  │ ├─ Apply conductive hydrogel properties to treated region           │   │
│  │ └─ Extract: Conduction velocity, APD, activation, arrhythmia risk   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 4: OPTIMAL SELECTION                                           │   │
│  │ ├─ Combine FEBio + OpenCarp metrics                                 │   │
│  │ ├─ Compute combined therapeutic score with weights                  │   │
│  │ ├─ Verify therapeutic threshold compliance                          │   │
│  │ └─ Output: Single BEST design per patient                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Design Space

### Polymer Library (24 Types)

| Category | Polymers | SMILES Examples |
|----------|----------|-----------------|
| **Protein-Modified** | GelMA_3pct, GelMA_5pct, GelMA_7pct, GelMA_10pct | `CC(=C)C(=O)NCCCC[C@H](NC(=O)C)C(=O)O` |
| **Conductive GelMA** | GelMA_BioIL, GelMA_Polypyrrole, GelMA_rGO, GelMA_MXene | With conductive additives |
| **Synthetic** | PEGDA_575, PEGDA_700, PEGDA_3400 | `C=CC(=O)OCCOCCOCCOC(=O)C=C` |
| **Polysaccharide** | Alginate_CaCl2, Alginate_RGD, Chitosan_HA, Chitosan_EGCG, Chitosan_thermogel | Natural polymers |
| **Glycosaminoglycan** | HA_acellular, HA_ECM, MeHA_photocrosslink | Hyaluronic acid variants |
| **Decellularized ECM** | dECM_VentriGel, dECM_cardiac | Cardiac-derived ECM |
| **Protein** | Gelatin_crosslinked, Fibrin_thrombin | Natural proteins |
| **Conductive** | PEDOT_PSS | `c1sc(c2OCCO2)cc1.c1ccc(S(=O)(=O)[O-])cc1` |

### Continuous Parameters

| Parameter | Range | Unit | Optimal Range |
|-----------|-------|------|---------------|
| Stiffness (E) | 5 - 30 | kPa | 12-18 kPa |
| Degradation (t50) | 7 - 180 | days | 30-90 days |
| Conductivity (σ) | 0 - 1.0 | S/m | 0.3-0.8 S/m |
| Thickness | 1 - 5 | mm | 3-5 mm |

### Coverage Options

| Coverage | Description | Factor |
|----------|-------------|--------|
| scar_only | Only scar tissue | 0.60 |
| scar_bz25 | Scar + 25% border zone | 0.75 |
| scar_bz50 | Scar + 50% border zone | 0.85 |
| scar_bz100 | Scar + full border zone | 1.00 |

## Therapeutic Thresholds

All selected designs must meet these criteria:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| ΔEF | ≥ 5% | Ejection fraction improvement |
| Wall Stress Reduction | ≥ 25% | Reduction in peak systolic stress |
| Strain Normalization | ≥ 15% | Improvement in strain uniformity |

## Scoring System

### Combined Therapeutic Score

```
Score = ΔEF × 3.0 + Stress_Reduction × 1.5 + Strain_Norm × 1.0 + CV_Improvement × 1.0 + Arrhythmia_Reduction × 0.5
```

### Weight Rationale

- **ΔEF (3.0)**: Primary clinical endpoint for heart failure
- **Stress Reduction (1.5)**: Prevents adverse remodeling
- **Strain Normalization (1.0)**: Improves regional function
- **CV Improvement (1.0)**: Reduces conduction abnormalities
- **Arrhythmia Reduction (0.5)**: Secondary safety endpoint

## Usage

### Full Pipeline Execution

```bash
# Run complete pipeline for all patients with full resources
python scripts/pipeline/run_full_pipeline.py --all --gpus 16 --cpus 96

# Run for single patient
python scripts/pipeline/run_full_pipeline.py --patient SCD0000101 --gpus 4 --cpus 24
```

### Individual Steps

```bash
# Step 1: Generate 10M designs
python scripts/pipeline/generate_10M_designs.py --all --gpus 16

# Step 2: Run FEBio simulations
python scripts/pipeline/run_febio_simulations.py --all --n-cpus 96

# Step 3: Run OpenCarp simulations
python scripts/pipeline/run_opencarp_simulations.py --all --n-cpus 96

# Step 4: Select optimal designs
python scripts/pipeline/select_optimal_design.py --all
```

### Resource Requirements

| Step | Primary Resource | Parallelization |
|------|-----------------|-----------------|
| Design Generation | GPU | 16 GPUs, batch processing |
| FEBio Simulation | CPU | 96 CPUs / 4 per simulation = 24 parallel |
| OpenCarp Simulation | CPU | 96 CPUs / 2 per simulation = 48 parallel |
| Selection | CPU | Single process |

### Estimated Runtime

| Step | Per Patient | All 10 Patients |
|------|-------------|-----------------|
| Design Generation | ~4 minutes | ~40 minutes |
| FEBio Simulations | ~2 minutes | ~20 minutes |
| OpenCarp Simulations | ~1 minute | ~10 minutes |
| Selection | <1 second | <10 seconds |
| **Total** | ~7 minutes | ~70 minutes |

## Output Files

### Directory Structure

```
results/
├── design_generation/
│   └── {patient_id}/
│       └── top_100_designs.csv
├── febio_simulations/
│   └── {patient_id}/
│       └── febio_simulation_results.csv
├── opencarp_simulations/
│   └── {patient_id}/
│       └── opencarp_simulation_results.csv
└── final_optimal_designs/
    ├── FINAL_OPTIMAL_DESIGNS_REPORT.md
    ├── final_optimal_designs_summary.csv
    └── {patient_id}/
        ├── optimal_design.csv
        └── all_designs_ranked.csv
```

### Output File Descriptions

| File | Description |
|------|-------------|
| `top_100_designs.csv` | Top 100 designs from 10M generation |
| `febio_simulation_results.csv` | FEBio metrics for all 100 designs |
| `opencarp_simulation_results.csv` | OpenCarp metrics for all 100 designs |
| `optimal_design.csv` | Single best design for patient |
| `all_designs_ranked.csv` | All 100 designs ranked by combined score |
| `FINAL_OPTIMAL_DESIGNS_REPORT.md` | Comprehensive summary report |

## Simulation Models

### FEBio Mechanical Model

- **Material Model**: Holzapfel-Ogden anisotropic hyperelastic
- **Regions**: Healthy, border zone, scar, hydrogel
- **Loading**: Pressure boundary conditions (cardiac cycle)
- **Output**: Von Mises stress, Green-Lagrange strain, displacement

### OpenCarp EP Model

- **Ionic Model**: ten Tusscher-Panfilov 2006
- **Conductivity**: Region-specific longitudinal/transverse
- **Protocol**: S1S2 pacing for arrhythmia assessment
- **Output**: Activation time, APD, conduction velocity

## Validation

### Biomechanical Model Validation

The physics-based treatment models are validated against:

1. **Laplace Law**: Wall stress reduction from increased wall thickness
2. **Frank-Starling Mechanism**: EF improvement from reduced afterload
3. **Literature Values**: Comparison with published hydrogel studies

### Therapeutic Validation

All selected designs are verified to:

1. Exceed all therapeutic thresholds
2. Show consistent improvement across metrics
3. Use clinically feasible hydrogel properties

## Troubleshooting

### Common Issues

1. **Memory errors during generation**
   - Reduce batch size: `--batch-size 50000`
   - Process patients sequentially

2. **Simulation timeout**
   - Increase timeout in script
   - Check mesh quality

3. **No THERAPEUTIC designs found**
   - Expand parameter ranges
   - Check baseline patient data

## References

1. Holzapfel-Ogden cardiac material model
2. ten Tusscher-Panfilov ionic model
3. FEBio finite element software
4. OpenCarp cardiac EP simulation
