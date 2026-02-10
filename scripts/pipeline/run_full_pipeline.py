#!/usr/bin/env python3
"""
HYDRA-BERT: Complete 10 Million Design Pipeline

Master script that orchestrates the full patient-specific hydrogel design pipeline:

1. Generate 10 million unique designs per patient (GPU-accelerated)
2. Rank top 100 based on HYDRA-BERT predictions
3. Run FEBio simulations on top 100 (parallel across CPUs)
4. Run OpenCarp simulations on top 100 (parallel across CPUs)
5. Select optimal design based on combined simulation results

Resource Utilization:
- 16 GPUs for design generation and prediction
- 96 CPUs for parallel FEBio/OpenCarp simulations

Usage:
    python run_full_pipeline.py --all --gpus 16 --cpus 96
    python run_full_pipeline.py --patient SCD0000101 --gpus 4 --cpus 24
    python run_full_pipeline.py --step generate --all
    python run_full_pipeline.py --step febio --all
    python run_full_pipeline.py --step opencarp --all
    python run_full_pipeline.py --step select --all

Author: HYDRA-BERT Pipeline
Date: 2026-02-09
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path("/home/ubuntu/HYDRA-BERT-FINAL")
SCRIPTS_DIR = BASE_DIR / "scripts" / "pipeline"

# Pipeline steps
PIPELINE_STEPS = [
    ('generate', 'generate_10M_designs.py', 'Generate 10M designs per patient'),
    ('febio', 'run_febio_simulations.py', 'Run FEBio mechanical simulations'),
    ('opencarp', 'run_opencarp_simulations.py', 'Run OpenCarp EP simulations'),
    ('select', 'select_optimal_design.py', 'Select optimal design from results'),
]


def print_banner():
    """Print pipeline banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     ██╗  ██╗██╗   ██╗██████╗ ██████╗  █████╗       ██████╗ ███████╗██████╗ ████████╗ ║
║     ██║  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██╔══██╗      ██╔══██╗██╔════╝██╔══██╗╚══██╔══╝ ║
║     ███████║ ╚████╔╝ ██║  ██║██████╔╝███████║█████╗██████╔╝█████╗  ██████╔╝   ██║    ║
║     ██╔══██║  ╚██╔╝  ██║  ██║██╔══██╗██╔══██║╚════╝██╔══██╗██╔══╝  ██╔══██╗   ██║    ║
║     ██║  ██║   ██║   ██████╔╝██║  ██║██║  ██║      ██████╔╝███████╗██║  ██║   ██║    ║
║     ╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝      ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝    ║
║                                                                              ║
║               Patient-Specific Hydrogel Design Optimization                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_pipeline_overview():
    """Print pipeline overview."""
    overview = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE OVERVIEW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: DESIGN GENERATION (GPU-accelerated)                               │
│  ├─ Generate 10,000,000 unique designs per patient                         │
│  ├─ Evaluate using HYDRA-BERT model                                        │
│  └─ Select top 100 designs based on predicted therapeutic score            │
│                                                                             │
│  STEP 2: FEBio SIMULATION (CPU-parallel)                                   │
│  ├─ Run finite element simulations for each top 100 design                 │
│  ├─ Place hydrogel in patient-specific infarct zone                        │
│  └─ Extract: Wall stress, strain, LVEF, displacement                       │
│                                                                             │
│  STEP 3: OpenCarp SIMULATION (CPU-parallel)                                │
│  ├─ Run electrophysiology simulations for each design                      │
│  ├─ Apply conductive hydrogel properties to infarct region                 │
│  └─ Extract: Conduction velocity, APD, arrhythmia vulnerability            │
│                                                                             │
│  STEP 4: OPTIMAL SELECTION                                                  │
│  ├─ Combine FEBio + OpenCarp metrics                                       │
│  ├─ Rank by combined therapeutic score                                     │
│  └─ Select single BEST design per patient                                  │
│                                                                             │
│  THERAPEUTIC THRESHOLDS:                                                    │
│  ├─ ΔEF ≥ 5%                                                               │
│  ├─ Wall Stress Reduction ≥ 25%                                            │
│  └─ Strain Normalization ≥ 15%                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
    print(overview)


def run_step(step_name: str, script_name: str, patients: list, gpus: int, cpus: int) -> bool:
    """Run a pipeline step."""
    script_path = SCRIPTS_DIR / script_name

    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False

    logger.info(f"Running step: {step_name}")
    logger.info(f"  Script: {script_path}")

    # Build command
    cmd = [sys.executable, str(script_path)]

    if len(patients) == 10:  # All patients
        cmd.append('--all')
    else:
        cmd.extend(['--patient', patients[0]])

    # Add resource arguments
    if step_name == 'generate':
        cmd.extend(['--gpus', str(gpus)])
    elif step_name in ['febio', 'opencarp']:
        cmd.extend(['--n-cpus', str(cpus)])

    # Run
    start_time = datetime.now()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            timeout=7200  # 2 hour timeout per step
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        if proc.returncode == 0:
            logger.info(f"  Step {step_name} completed in {elapsed:.1f}s")
            return True
        else:
            logger.error(f"  Step {step_name} failed with code {proc.returncode}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"  Step {step_name} timed out")
        return False
    except Exception as e:
        logger.error(f"  Step {step_name} error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='HYDRA-BERT Complete Patient-Specific Optimization Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline for all patients
  python run_full_pipeline.py --all --gpus 16 --cpus 96

  # Run for single patient
  python run_full_pipeline.py --patient SCD0000101

  # Run specific step only
  python run_full_pipeline.py --step generate --all
  python run_full_pipeline.py --step febio --all
  python run_full_pipeline.py --step opencarp --all
  python run_full_pipeline.py --step select --all
"""
    )

    parser.add_argument('--patient', type=str, help='Process single patient')
    parser.add_argument('--all', action='store_true', help='Process all patients')
    parser.add_argument('--step', type=str, choices=['generate', 'febio', 'opencarp', 'select'],
                       help='Run specific step only')
    parser.add_argument('--gpus', type=int, default=16, help='Number of GPUs (default: 16)')
    parser.add_argument('--cpus', type=int, default=96, help='Number of CPUs (default: 96)')
    parser.add_argument('--n-designs', type=int, default=10_000_000,
                       help='Number of designs per patient (default: 10M)')

    args = parser.parse_args()

    print_banner()

    # Validate arguments
    if not args.patient and not args.all:
        parser.print_help()
        return 1

    patients = [
        "SCD0000101", "SCD0000201", "SCD0000301", "SCD0000401",
        "SCD0000601", "SCD0000701", "SCD0000801", "SCD0001001",
        "SCD0001101", "SCD0001201"
    ]

    if args.patient:
        patients = [args.patient]

    print_pipeline_overview()

    logger.info("="*70)
    logger.info("HYDRA-BERT: Complete Pipeline Execution")
    logger.info("="*70)
    logger.info(f"Patients: {len(patients)}")
    logger.info(f"Designs per patient: {args.n_designs:,}")
    logger.info(f"GPUs: {args.gpus}")
    logger.info(f"CPUs: {args.cpus}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    # Determine which steps to run
    if args.step:
        steps = [(name, script, desc) for name, script, desc in PIPELINE_STEPS if name == args.step]
    else:
        steps = PIPELINE_STEPS

    # Run pipeline
    pipeline_start = datetime.now()
    success = True

    for step_name, script_name, description in steps:
        logger.info(f"\n{'='*60}")
        logger.info(f"STEP: {description.upper()}")
        logger.info(f"{'='*60}")

        if not run_step(step_name, script_name, patients, args.gpus, args.cpus):
            logger.error(f"Pipeline failed at step: {step_name}")
            success = False
            break

    pipeline_elapsed = (datetime.now() - pipeline_start).total_seconds()

    # Final summary
    logger.info("\n" + "="*70)
    if success:
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    else:
        logger.info("PIPELINE FAILED - Check logs for errors")

    logger.info(f"Total time: {pipeline_elapsed:.1f}s ({pipeline_elapsed/60:.1f} minutes)")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)

    # Print results location
    if success:
        print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RESULTS LOCATION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Design Generation:                                                         │
│  └─ {BASE_DIR}/results/design_generation/                          │
│     └─ [patient_id]/top_100_designs.csv                                    │
│                                                                             │
│  FEBio Simulations:                                                         │
│  └─ {BASE_DIR}/results/febio_simulations/                          │
│     └─ [patient_id]/febio_simulation_results.csv                           │
│                                                                             │
│  OpenCarp Simulations:                                                      │
│  └─ {BASE_DIR}/results/opencarp_simulations/                       │
│     └─ [patient_id]/opencarp_simulation_results.csv                        │
│                                                                             │
│  Final Optimal Designs:                                                     │
│  └─ {BASE_DIR}/results/final_optimal_designs/                      │
│     ├─ FINAL_OPTIMAL_DESIGNS_REPORT.md                                     │
│     ├─ final_optimal_designs_summary.csv                                   │
│     └─ [patient_id]/optimal_design.csv                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
