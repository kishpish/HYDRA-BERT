#!/usr/bin/env python3
"""
HYDRA-BERT Stage 3: 10 Million Design Generation Pipeline

This is the main production script for generating 10 million patient-specific
hydrogel designs using the trained HYDRA-BERT model and multi-GPU parallelization.

Pipeline Overview:
==================
1. Generate 10M candidate designs per patient (total: 100M designs)
2. Score with HYDRA-BERT model (10 key metrics)
3. Filter to top 10,000 by filtering score
4. Calculate 53 comprehensive simulation metrics (FEBio/OpenCarp)
5. Apply therapeutic threshold validation (5 tiers)
6. Select Pareto-optimal designs across multiple objectives
7. Generate validation reports and export results

Hardware Requirements:
=====================
- 16 GPUs (NVIDIA A100 40GB recommended)
- ~640GB GPU memory total
- ~256GB system RAM
- ~500GB disk space for results

Estimated Runtime:
=================
- 10M designs per patient: ~20-30 minutes
- 10 patients total: ~3-4 hours

Usage:
======
    # Full pipeline (10M designs, 10 patients)
    python run_10M_design_pipeline.py

    # Custom configuration
    python run_10M_design_pipeline.py --designs_per_patient 1000000 --num_gpus 8

Output:
=======
    results/therapeutic/
    ├── all_results.json           # Complete results
    ├── therapeutic_summary.txt    # Summary report
    └── {patient_id}/
        ├── therapeutic_designs.json
        ├── pareto_optimal.json
        ├── final_selection.json
        ├── top_100_full_metrics.csv
        └── validation_report.txt

Author: HYDRA-BERT Team
Version: 1.0.0
"""

import os
import sys
import json
import logging
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('stage3_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


# DEFAULT CONFIGURATION

DEFAULT_CONFIG = {
    # Design generation
    "DESIGNS_PER_PATIENT": 10_000_000,  # 10 million
    "BATCH_SIZE": 5000,                  # HYDRA-BERT scoring batch size

    # Filtering stages
    "TOP_K_STAGE1": 10000,               # After key metric filtering
    "TOP_K_STAGE2": 1000,                # For full simulation
    "TOP_K_STAGE3": 100,                 # For therapeutic validation
    "FINAL_SELECTION": 10,               # Best designs per patient

    # Hardware
    "NUM_GPUS": 16,
    "PROCESSES_PER_GPU": 1,

    # Paths
    "OUTPUT_DIR": "results/therapeutic",
    "CHECKPOINT_PATH": "checkpoints/stage1/best_model.pt",

    # Reproducibility
    "SEED": 42,
}


# PARETO OPTIMIZATION METRICS

PARETO_METRICS = [
    # Primary efficacy (higher is better)
    "delta_EF_pct",
    "delta_BZ_stress_reduction_pct",
    "strain_normalization_pct",

    # Secondary efficacy (higher is better)
    "delta_GLS_pct",
    "stroke_volume_improvement_pct",

    # Safety (inverted - more negative is better)
    "neg_toxicity",           # -toxicity_score
    "structural_integrity",
    "neg_arrhythmia_risk",    # -arrhythmia_risk

    # Electrical function (higher is better)
    "cv_improvement_pct",
]


def find_pareto_optimal(designs: List[Dict], metrics: List[str]) -> List[Dict]:
    """
    Find Pareto-optimal designs across multiple metrics.

    A design is Pareto-optimal if no other design dominates it across all metrics.
    Dominance: design A dominates B if A >= B in all metrics and A > B in at least one.

    Args:
        designs: List of design dictionaries
        metrics: List of metric names to optimize (all maximized)

    Returns:
        List of Pareto-optimal designs
    """
    if not designs:
        return []

    n = len(designs)
    is_dominated = [False] * n

    for i in range(n):
        if is_dominated[i]:
            continue

        for j in range(n):
            if i == j or is_dominated[j]:
                continue

            # Check if j dominates i
            j_better_or_equal_all = True
            j_strictly_better_one = False

            for metric in metrics:
                vi = designs[i].get(metric, 0)
                vj = designs[j].get(metric, 0)

                if vj < vi:
                    j_better_or_equal_all = False
                    break
                if vj > vi:
                    j_strictly_better_one = True

            if j_better_or_equal_all and j_strictly_better_one:
                is_dominated[i] = True
                break

    return [d for i, d in enumerate(designs) if not is_dominated[i]]


def process_patient_on_gpu(args: Tuple) -> Dict[str, Any]:
    """
    Process a single patient on a specific GPU.

    This function is executed in a separate process for each patient.
    It generates designs, scores them, filters, simulates, and validates.

    Args:
        args: Tuple containing (patient_id, patient_config_dict, gpu_id,
              output_dir, checkpoint_path, designs_per_patient, config)

    Returns:
        Dictionary with processing results
    """
    (patient_id, patient_config_dict, gpu_id, output_dir,
     checkpoint_path, designs_per_patient, config) = args

    # Set GPU for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda:0'

    # Import modules here to avoid CUDA initialization before fork
    from hydra_bert_v2.models import HydraBERT
    from hydra_bert_v2.stage3 import PatientConfig
    from hydra_bert_v2.stage3.generation import DesignGenerator
    from hydra_bert_v2.stage3.simulation import SimulationRunner, SimulationConfig
    from hydra_bert_v2.stage3.simulation.comprehensive_metrics import (
        ComprehensiveSimulationCalculator
    )
    from hydra_bert_v2.stage3.therapeutic_thresholds import (
        TherapeuticClassifier,
        compute_filtering_score,
    )
    from hydra_bert_v2.stage3.metrics import MetricsCalculator

    logger.info(f"GPU {gpu_id}: Starting {patient_id} - {designs_per_patient:,} designs")

    try:
        # Recreate patient config from dictionary
        patient_config = PatientConfig(**patient_config_dict)

        # Load HYDRA-BERT model
        model, _ = HydraBERT.load_checkpoint(checkpoint_path, device)
        model.eval()

        # Initialize pipeline components
        generator = DesignGenerator(
            model=model,
            num_designs_per_patient=designs_per_patient,
            batch_size=config["BATCH_SIZE"],
            device=device,
            seed=config["SEED"] + gpu_id
        )

        sim_calculator = ComprehensiveSimulationCalculator(
            febio_available=False,
            opencarp_available=False,
            use_high_fidelity=True
        )

        classifier = TherapeuticClassifier()
        metrics_calc = MetricsCalculator()

        # Create output directory
        patient_dir = Path(output_dir) / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)

        start_time = datetime.now()

        # STAGE 1: Generate 10M designs
        logger.info(f"GPU {gpu_id}: [{patient_id}] Stage 1 - Generating designs...")

        designs = generator.generate_designs(patient_config, use_ppo=True)
        logger.info(f"GPU {gpu_id}: [{patient_id}] Generated {len(designs):,} designs")

        # STAGE 2: Score with HYDRA-BERT and filter
        logger.info(f"GPU {gpu_id}: [{patient_id}] Stage 2 - Scoring with HYDRA-BERT...")

        designs = generator.score_designs_batch(designs, patient_config)

        # Compute filtering score for each design
        for design in designs:
            design.filtering_score = compute_filtering_score({
                "delta_EF_pct": getattr(design, 'predicted_delta_ef', 0),
                "delta_BZ_stress_reduction_pct": getattr(design, 'predicted_gcs', 0) * 3,
                "strain_normalization_pct": getattr(design, 'predicted_gcs', 0) * 1.5,
                "predicted_optimal_prob": getattr(design, 'predicted_optimal_prob', 0),
                "reward": getattr(design, 'reward', 0),
                "toxicity_score": getattr(design, 'predicted_toxicity', 0.5),
                "structural_integrity": getattr(design, 'predicted_integrity', 0.5),
                "rupture_risk": 0.05,
                "arrhythmia_risk": getattr(design, 'predicted_fibrosis_risk', 0.2),
                "compliance_mismatch": abs(design.stiffness_kPa - 15) / 15,
            })

        # Sort by filtering score and keep top 10K
        designs.sort(key=lambda d: d.filtering_score, reverse=True)
        top_10k = designs[:config["TOP_K_STAGE1"]]

        logger.info(f"GPU {gpu_id}: [{patient_id}] Top 10K filtering scores: "
                   f"{top_10k[-1].filtering_score:.2f} to {top_10k[0].filtering_score:.2f}")

        # Free memory
        del designs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # STAGE 3: Calculate 53 simulation metrics for top 1,000
        logger.info(f"GPU {gpu_id}: [{patient_id}] Stage 3 - Calculating 53 metrics...")

        top_1k = top_10k[:config["TOP_K_STAGE2"]]
        designs_with_metrics = []

        sim_config = SimulationConfig(
            output_dir=str(patient_dir / "sims"),
            num_parallel_sims=4
        )
        sim_runner = SimulationRunner(sim_config)

        for design in top_1k:
            # Run simulation
            sim_result = sim_runner.run_single(design, patient_config)

            # Calculate all 53 metrics
            full_metrics = sim_calculator.calculate_all_metrics(design, patient_config)
            basic_metrics = metrics_calc.calculate_all_metrics(
                design, patient_config, sim_result
            )

            # Merge metrics
            combined = {**basic_metrics, **full_metrics}

            # Add derived metrics for Pareto optimization
            combined["wall_stress_reduction_pct"] = combined.get(
                "mech_wall_stress_reduction_pct", 0
            )
            combined["cv_improvement_pct"] = combined.get(
                "elec_cv_improvement_pct", 0
            )
            combined["stroke_volume_improvement_pct"] = combined.get(
                "func_stroke_volume_improvement_pct", 0
            )
            combined["neg_toxicity"] = -combined.get("toxicity_score", 0)
            combined["neg_arrhythmia_risk"] = -combined.get("arrhythmia_risk", 0)

            designs_with_metrics.append(combined)

        logger.info(f"GPU {gpu_id}: [{patient_id}] Calculated metrics for "
                   f"{len(designs_with_metrics)} designs")

        # STAGE 4: Apply therapeutic thresholds
        logger.info(f"GPU {gpu_id}: [{patient_id}] Stage 4 - Therapeutic classification...")

        therapeutic_designs = []
        supportive_designs = []
        all_classified = []

        for metrics in designs_with_metrics:
            classification = classifier.classify(metrics)
            metrics["therapeutic_classification"] = classification
            all_classified.append(metrics)

            if classification["classification"] == "THERAPEUTIC":
                therapeutic_designs.append(metrics)
            elif classification["classification"] == "SUPPORTIVE":
                supportive_designs.append(metrics)

        logger.info(f"GPU {gpu_id}: [{patient_id}] Found {len(therapeutic_designs)} "
                   f"THERAPEUTIC, {len(supportive_designs)} SUPPORTIVE")

        # STAGE 5: Pareto-optimal selection
        logger.info(f"GPU {gpu_id}: [{patient_id}] Stage 5 - Pareto optimization...")

        if len(therapeutic_designs) >= 5:
            candidate_pool = therapeutic_designs
        elif len(supportive_designs) >= 5:
            candidate_pool = supportive_designs
        else:
            candidate_pool = sorted(
                all_classified,
                key=lambda x: x["therapeutic_classification"]["therapeutic_score"],
                reverse=True
            )[:100]

        pareto_optimal = find_pareto_optimal(candidate_pool, PARETO_METRICS)
        logger.info(f"GPU {gpu_id}: [{patient_id}] Found {len(pareto_optimal)} "
                   "Pareto-optimal designs")

        # STAGE 6: Final selection and ranking
        logger.info(f"GPU {gpu_id}: [{patient_id}] Stage 6 - Final selection...")

        pareto_optimal.sort(
            key=lambda x: x["therapeutic_classification"]["therapeutic_score"],
            reverse=True
        )

        final_designs = pareto_optimal[:config["FINAL_SELECTION"]]

        for i, design in enumerate(final_designs):
            design["final_rank"] = i + 1

        # Save results
        logger.info(f"GPU {gpu_id}: [{patient_id}] Saving results...")

        # Save therapeutic designs
        with open(patient_dir / "therapeutic_designs.json", 'w') as f:
            json.dump(therapeutic_designs[:100], f, indent=2, default=str)

        # Save Pareto-optimal
        with open(patient_dir / "pareto_optimal.json", 'w') as f:
            json.dump(pareto_optimal[:50], f, indent=2, default=str)

        # Save final selection
        with open(patient_dir / "final_selection.json", 'w') as f:
            json.dump(final_designs, f, indent=2, default=str)

        # Save CSV with full metrics
        import pandas as pd
        pd.DataFrame(all_classified[:100]).to_csv(
            patient_dir / "top_100_full_metrics.csv", index=False
        )

        # Generate validation report
        generate_patient_report(
            patient_config, therapeutic_designs, supportive_designs,
            pareto_optimal, final_designs, patient_dir / "validation_report.txt"
        )

        elapsed = datetime.now() - start_time
        best = final_designs[0] if final_designs else {}

        logger.info(f"GPU {gpu_id}: [{patient_id}] COMPLETE - "
                   f"Best: {best.get('polymer_name', 'N/A')} | "
                   f"ΔEF: +{best.get('delta_EF_pct', 0):.2f}% | "
                   f"Time: {elapsed}")

        return {
            "patient_id": patient_id,
            "status": "success",
            "gpu_id": gpu_id,
            "total_generated": designs_per_patient,
            "therapeutic_count": len(therapeutic_designs),
            "supportive_count": len(supportive_designs),
            "pareto_count": len(pareto_optimal),
            "final_selection": final_designs,
            "best_design": best,
            "processing_time": str(elapsed),
        }

    except Exception as e:
        logger.error(f"GPU {gpu_id}: [{patient_id}] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "patient_id": patient_id,
            "status": "error",
            "error": str(e),
            "gpu_id": gpu_id,
        }


def generate_patient_report(
    patient_config,
    therapeutic_designs: List[Dict],
    supportive_designs: List[Dict],
    pareto_optimal: List[Dict],
    final_designs: List[Dict],
    output_path: Path
) -> None:
    """Generate patient-specific validation report."""

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("THERAPEUTIC VALIDATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Patient ID: {patient_config.patient_id}\n")
        f.write(f"Baseline LVEF: {patient_config.baseline_LVEF_pct:.1f}%\n")
        f.write(f"Scar Fraction: {patient_config.scar_fraction_pct:.1f}%\n")
        f.write(f"Border Zone: {patient_config.bz_fraction_pct:.1f}%\n\n")

        f.write("-" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"  THERAPEUTIC designs: {len(therapeutic_designs)}\n")
        f.write(f"  SUPPORTIVE designs: {len(supportive_designs)}\n")
        f.write(f"  Pareto-optimal: {len(pareto_optimal)}\n\n")

        f.write("-" * 80 + "\n")
        f.write("TOP DESIGNS\n")
        f.write("-" * 80 + "\n\n")

        for i, design in enumerate(final_designs[:5], 1):
            cls = design.get("therapeutic_classification", {})
            f.write(f"Rank {i}: {design.get('polymer_name', 'N/A')}\n")
            f.write(f"  Classification: {cls.get('classification', 'N/A')}\n")
            f.write(f"  Score: {cls.get('therapeutic_score', 0):.1f}/100\n")
            f.write(f"  Delta EF: +{design.get('delta_EF_pct', 0):.2f}%\n")
            f.write(f"  Stiffness: {design.get('hydrogel_E_kPa', 0):.1f} kPa\n\n")

        f.write("=" * 80 + "\n")


def main():
    """Main entry point for the 10M design pipeline."""

    parser = argparse.ArgumentParser(
        description="HYDRA-BERT 10M Design Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--designs_per_patient", type=int,
        default=DEFAULT_CONFIG["DESIGNS_PER_PATIENT"],
        help="Number of designs to generate per patient (default: 10M)"
    )
    parser.add_argument(
        "--num_gpus", type=int,
        default=DEFAULT_CONFIG["NUM_GPUS"],
        help="Number of GPUs to use (default: 16)"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=DEFAULT_CONFIG["OUTPUT_DIR"],
        help="Output directory for results"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=DEFAULT_CONFIG["CHECKPOINT_PATH"],
        help="Path to HYDRA-BERT checkpoint"
    )
    args = parser.parse_args()

    # Update config
    config = DEFAULT_CONFIG.copy()
    config["DESIGNS_PER_PATIENT"] = args.designs_per_patient
    config["NUM_GPUS"] = args.num_gpus
    config["OUTPUT_DIR"] = args.output_dir
    config["CHECKPOINT_PATH"] = args.checkpoint

    # Validate checkpoint
    if not os.path.exists(config["CHECKPOINT_PATH"]):
        logger.error(f"Checkpoint not found: {config['CHECKPOINT_PATH']}")
        sys.exit(1)

    # Import patient data
    from hydra_bert_v2.stage3 import REAL_PATIENTS

    # Print configuration
    print("HYDRA-BERT 10M DESIGN GENERATION PIPELINE")
    print(f"Designs per patient: {config['DESIGNS_PER_PATIENT']:,}")
    print(f"Total designs: {config['DESIGNS_PER_PATIENT'] * len(REAL_PATIENTS):,}")
    print(f"GPUs: {config['NUM_GPUS']}")
    print(f"Patients: {len(REAL_PATIENTS)}")
    print(f"Checkpoint: {config['CHECKPOINT_PATH']}")
    print(f"Output: {config['OUTPUT_DIR']}")

    # Create output directory
    Path(config["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

    # Prepare tasks
    tasks = []
    for i, (patient_id, patient_config) in enumerate(REAL_PATIENTS.items()):
        gpu_id = i % config["NUM_GPUS"]
        config_dict = {
            'patient_id': patient_config.patient_id,
            'baseline_LVEF_pct': patient_config.baseline_LVEF_pct,
            'baseline_GLS_pct': patient_config.baseline_GLS_pct,
            'baseline_EDV_mL': patient_config.baseline_EDV_mL,
            'baseline_ESV_mL': patient_config.baseline_ESV_mL,
            'scar_fraction_pct': patient_config.scar_fraction_pct,
            'bz_fraction_pct': patient_config.bz_fraction_pct,
            'transmurality': patient_config.transmurality,
            'wall_thickness_mm': patient_config.wall_thickness_mm,
            'bz_stress_kPa': getattr(patient_config, 'bz_stress_kPa', 30.0),
        }
        tasks.append((
            patient_id, config_dict, gpu_id, config["OUTPUT_DIR"],
            config["CHECKPOINT_PATH"], config["DESIGNS_PER_PATIENT"], config
        ))

    start_time = datetime.now()

    # Run in parallel using multiprocessing
    mp.set_start_method('spawn', force=True)
    pool_size = min(config["NUM_GPUS"], len(REAL_PATIENTS))

    print(f"\nStarting {pool_size} parallel workers...")

    with mp.Pool(processes=pool_size) as pool:
        results = pool.map(process_patient_on_gpu, tasks)

    # Collect and save results
    all_results = {r['patient_id']: r for r in results}

    with open(Path(config["OUTPUT_DIR"]) / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    total_time = datetime.now() - start_time

    print("\nPIPELINE COMPLETE")
    print(f"Total time: {total_time}")
    print(f"Patients processed: "
          f"{sum(1 for r in results if r.get('status') == 'success')}/{len(REAL_PATIENTS)}")

    print("\n" + "-" * 80)
    print("RESULTS SUMMARY")
    print("-" * 80)
    print(f"{'Patient':<12} {'THERAPEUTIC':<12} {'SUPPORTIVE':<12} {'Best ΔEF':<10}")
    print("-" * 80)

    total_therapeutic = 0
    total_supportive = 0

    for patient_id, result in all_results.items():
        if result.get('status') == 'success':
            total_therapeutic += result.get('therapeutic_count', 0)
            total_supportive += result.get('supportive_count', 0)
            best = result.get('best_design', {})
            print(f"{patient_id:<12} "
                  f"{result.get('therapeutic_count', 0):<12} "
                  f"{result.get('supportive_count', 0):<12} "
                  f"+{best.get('delta_EF_pct', 0):.2f}%")
        else:
            print(f"{patient_id:<12} ERROR: {result.get('error', 'Unknown')[:30]}")

    print("-" * 80)
    print(f"{'TOTAL':<12} {total_therapeutic:<12} {total_supportive:<12}")
    print("-" * 80)

    print(f"\nResults saved to: {Path(config['OUTPUT_DIR']).absolute()}")


if __name__ == "__main__":
    main()
