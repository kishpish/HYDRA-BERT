# HYDRA-BERT Simulation Guide

## Running ACTUAL FEBio and OpenCarp Simulations

This guide explains how to run real finite element simulations (not surrogate models) to validate HYDRA-BERT's therapeutic predictions.

---

## Overview

HYDRA-BERT uses physics-based surrogate models for the initial 100M design evaluation. Once optimal designs are selected, you can validate them with actual finite element simulations:

| Simulation | Software | Purpose | Metrics |
|------------|----------|---------|---------|
| **FEBio** | FEBio 4.0 | Cardiac mechanics | Wall stress, strain, EF |
| **OpenCarp** | OpenCarp 18.1 | Electrophysiology | CV, activation time, arrhythmia risk |

---

## Prerequisites

### System Requirements

- **CPU**: 24+ cores recommended (96 cores optimal)
- **RAM**: 64GB+ (256GB for parallel runs)
- **Storage**: 100GB for simulation outputs

### Software Dependencies

```bash
# FEBio 4.0
ls $FEBIO_HOME/bin/febio4

# OpenCarp (check installation)
which openCARP || ls /usr/local/bin/openCARP
```

### Mesh Data Required

Patient meshes must be available in:
```
$SCD_MODELS_DIR/
├── simulation_ready/{patient}/{patient}_tet.pts     # Node coordinates
├── infarct_results_comprehensive/{patient}/{patient}_tagged.elem  # Tagged elements
└── laplace_complete_v2/{patient}/{patient}.lon      # Fiber directions
```

---

## Quick Start

### 1. Run All Simulations (Parallel)

```bash
cd /path/to/HYDRA-BERT

# Use all 96 CPUs
python scripts/simulations/run_complete_simulations.py --all_cpus

# Or specify number of workers
python scripts/simulations/run_complete_simulations.py --parallel --n_workers 48
```

### 2. Run FEBio Only

```bash
python scripts/simulations/run_actual_febio_hydrogel.py --parallel --n_workers 48
```

### 3. Run OpenCarp Only

```bash
python scripts/simulations/run_actual_opencarp_hydrogel.py --parallel --n_workers 24
```

### 4. Run Single Patient (Testing)

```bash
python scripts/simulations/run_complete_simulations.py --patient SCD0000101
```

---

## Simulation Scripts

### `run_actual_febio_hydrogel.py`

Runs FEBio 4.0 cardiac mechanics simulations with hydrogel treatment.

**Key Features:**
- Holzapfel-Ogden orthotropic myocardium model
- 4 tissue regions: healthy, border zone, scar, hydrogel-treated
- Active contraction with calcium-dependent force
- Pressure loading (end-diastolic/systolic)

**Material Parameters:**
```python
# Healthy myocardium
a = 0.059 kPa      # Ground matrix
a_f = 18.472 kPa   # Fiber direction
a_s = 2.481 kPa    # Sheet direction

# Hydrogel-treated tissue
# Scaled based on hydrogel stiffness (default 15 kPa)
stiffness_ratio = hydrogel_E_kPa / 15.0
```

**Output Metrics (17 total):**
| Category | Metrics |
|----------|---------|
| Global Function | LVEF, EDV, ESV, stroke volume, cardiac output |
| Regional Function | GLS, BZ contractility, scar motion, remote strain |
| Wall Stress | Peak systolic/diastolic, heterogeneity, fiber, cross-fiber |
| Geometry | Wall thickening, radial/circumferential strain |

### `run_actual_opencarp_hydrogel.py`

Runs OpenCarp electrophysiology simulations with hydrogel-enhanced conductivity.

**Key Features:**
- Ten Tusscher-Panfilov ionic model
- 3 conductivity regions with hydrogel enhancement
- Apex stimulation (200 nodes)
- LAT (Local Activation Time) measurement

**Conductivity Parameters:**
```python
# Healthy tissue (S/m)
g_il = 0.174  # Longitudinal
g_it = 0.019  # Transverse

# Scar (5% of healthy)
g_il = 0.0087
g_it = 0.00095

# Hydrogel-treated (enhanced based on conductivity)
effective = base + hydrogel_conductivity * 0.3
```

**Output Metrics:**
| Metric | Description |
|--------|-------------|
| Total Activation Time | Max LAT (ms) |
| QRS Duration | 95th - 5th percentile LAT |
| CV Improvement | % reduction in activation time |
| LAT Dispersion | Standard deviation of LAT |
| Arrhythmia Index | Composite vulnerability score |

### `run_complete_simulations.py`

Master script that runs both FEBio and OpenCarp, then combines results.

**Arguments:**
```
--all_cpus          Use all available CPUs (96)
--n_workers N       Number of parallel workers
--febio_only        Run only FEBio
--opencarp_only     Run only OpenCarp
--patient ID        Run single patient
--designs_csv       Path to therapeutic designs CSV
```

### `extract_simulation_metrics.py`

Extracts metrics from simulation output files.

**Usage:**
```bash
# Extract all patients
python scripts/simulations/extract_simulation_metrics.py --all

# Single patient
python scripts/simulations/extract_simulation_metrics.py --patient SCD0000101
```

---

## Output Structure

```
results/
├── febio_hydrogel_simulations/
│   └── {patient}/
│       ├── {patient}_hydrogel.feb    # Input file
│       ├── {patient}_hydrogel.xplt   # Binary output
│       └── {patient}_results.json    # Extracted metrics
│
├── opencarp_hydrogel_simulations/
│   └── {patient}/
│       ├── mesh/                      # Converted mesh
│       ├── opencarp/                  # Simulation files
│       └── {patient}_ep_results.json  # EP metrics
│
├── simulation_validation/combined/
│   ├── complete_simulation_results.json
│   ├── simulation_summary.csv
│   └── SIMULATION_VALIDATION_REPORT.md
│
└── extracted_metrics/
    ├── all_extracted_metrics.json
    └── extracted_metrics_summary.csv
```

---

## Hydrogel Treatment Parameters

The optimal hydrogel formulation (GelMA_BioIL) parameters:

| Parameter | Optimal Value | Effect |
|-----------|---------------|--------|
| Stiffness | 15 kPa | Matches native myocardium |
| Conductivity | 0.25-0.80 S/m | Restores electrical propagation |
| Degradation | 45-55 days | Sustained remodeling |
| Thickness | 3.9-5.0 mm | Mechanical support |
| Coverage | scar_bz100 | Full scar + border zone |

---

## Expected Runtime

| Simulation | Per Patient | All 10 Patients (Parallel) |
|------------|-------------|---------------------------|
| FEBio | 5-15 min | 15-30 min (48 workers) |
| OpenCarp | 10-20 min | 20-40 min (24 workers) |
| Combined | 15-30 min | 30-60 min (48 workers) |

---

## Troubleshooting

### FEBio Issues

```bash
# Check binary
ls -la $FEBIO_HOME/bin/febio4

# Test run
export LD_LIBRARY_PATH=$FEBIO_HOME/lib
$FEBIO_HOME/bin/febio4 -help
```

### OpenCarp Issues

```bash
# Find binary
find /usr -name "openCARP" 2>/dev/null

# Check version
openCARP --version
```

### Mesh Not Found

```bash
# Verify mesh files exist
ls $SCD_MODELS_DIR/simulation_ready/SCD0000101/
ls $SCD_MODELS_DIR/infarct_results_comprehensive/SCD0000101/
ls $SCD_MODELS_DIR/laplace_complete_v2/SCD0000101/
```

### Memory Issues

Reduce parallel workers:
```bash
python scripts/simulations/run_complete_simulations.py --n_workers 16
```

---

## Validation Workflow

1. **Run surrogate model predictions** (100M designs, ~3 hours)
2. **Select therapeutic designs** (top 10K per patient)
3. **Run actual FEBio simulations** (validates mechanics)
4. **Run actual OpenCarp simulations** (validates EP)
5. **Extract and compare metrics**
6. **Generate validation report**

---

## Comparison: Surrogate vs Actual Simulation

| Aspect | Surrogate Model | Actual FEA |
|--------|-----------------|------------|
| Speed | 100K designs/sec | 1 design/10 min |
| Accuracy | Approximate | High fidelity |
| Use Case | Screening 100M designs | Final validation |
| Metrics | 53 physics-based | 17 FEBio + 10 OpenCarp |

The surrogate models provide directionally correct predictions that are validated by actual simulations.

---

## References

1. **FEBio**: Maas SA et al. "FEBio: Finite Elements for Biomechanics." J Biomech Eng. 2012
2. **OpenCarp**: Plank G et al. "The openCARP simulation environment." Comp Meth Prog Biomed. 2021
3. **Holzapfel-Ogden**: Holzapfel GA, Ogden RW. "Constitutive modelling of passive myocardium." Phil Trans R Soc A. 2009
4. **Ten Tusscher-Panfilov**: ten Tusscher KH, Panfilov AV. "Alternans and spiral breakup in a human ventricular tissue model." Am J Physiol Heart Circ Physiol. 2006
