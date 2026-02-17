# HYDRA-BERT Quick Start Guide

## Prerequisites

```bash
# Create conda environment
conda create -n hydra_bert python=3.10 -y
conda activate hydra_bert

# Install dependencies
pip install torch numpy pandas scikit-learn tqdm
```

## Run Complete Pipeline

The simplest way to run the complete pipeline:

```bash
cd /path/to/HYDRA-BERT

# Run for all 10 patients (16 GPUs, 96 CPUs)
python scripts/pipeline/run_full_pipeline.py --all --gpus 16 --cpus 96
```

This will:
1. Generate 10 million designs per patient
2. Select top 100 designs per patient
3. Run FEBio mechanical simulations
4. Run OpenCarp EP simulations
5. Select the single best design per patient

## View Results

After the pipeline completes:

```bash
# View final report
cat results/final_optimal_designs/FINAL_OPTIMAL_DESIGNS_REPORT.md

# View summary CSV
cat results/final_optimal_designs/final_optimal_designs_summary.csv

# View specific patient result
cat results/final_optimal_designs/SCD0000101/optimal_design.csv
```

## Run Individual Steps

```bash
# Step 1: Design Generation (10M per patient)
python scripts/pipeline/generate_10M_designs.py --all --gpus 16

# Step 2: FEBio Simulations
python scripts/pipeline/run_febio_simulations.py --all --n-cpus 96

# Step 3: OpenCarp Simulations
python scripts/pipeline/run_opencarp_simulations.py --all --n-cpus 96

# Step 4: Optimal Selection
python scripts/pipeline/select_optimal_design.py --all
```

## Single Patient Mode

To run for a single patient:

```bash
python scripts/pipeline/run_full_pipeline.py --patient SCD0000101
```

## Estimated Runtime

| Configuration | All 10 Patients |
|---------------|-----------------|
| 16 GPUs, 96 CPUs | ~1 hour |
| 4 GPUs, 24 CPUs | ~4 hours |
| 1 GPU, 8 CPUs | ~15 hours |

## Output Files

```
results/
├── design_generation/
│   └── {patient_id}/
│       └── top_100_designs.csv      # Top 100 from 10M
├── febio_simulations/
│   └── {patient_id}/
│       └── febio_simulation_results.csv
├── opencarp_simulations/
│   └── {patient_id}/
│       └── opencarp_simulation_results.csv
└── final_optimal_designs/
    ├── FINAL_OPTIMAL_DESIGNS_REPORT.md   # Summary report
    ├── final_optimal_designs_summary.csv  # All patients
    └── {patient_id}/
        └── optimal_design.csv             # Best design
```

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python scripts/pipeline/generate_10M_designs.py --all --batch-size 50000
```

### Slow Generation

Ensure GPUs are available:
```bash
python -c "import torch; print(torch.cuda.device_count())"
```

### Missing Results

Check if previous steps completed:
```bash
ls results/design_generation/*/top_100_designs.csv
ls results/febio_simulations/*/febio_simulation_results.csv
```
