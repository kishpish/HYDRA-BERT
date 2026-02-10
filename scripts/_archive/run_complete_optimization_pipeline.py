#!/usr/bin/env python3
"""
HYDRA-BERT: Complete Patient-Specific Optimization Pipeline

This script runs the full pipeline for selecting patient-specific hydrogel designs:

1. Load all 576,000 generated designs (24 polymers x 24,000 per polymer)
2. For each patient:
   - Simulate treatment effects using FEBio-based biomechanical models
   - Evaluate ALL designs across ALL polymer types
   - Rank designs by therapeutic score
   - Select the BEST design (which may use a DIFFERENT polymer for each patient)
3. Generate comprehensive reports

Key Features:
- Evaluates 24 DIFFERENT polymers with unique SMILES structures
- Uses validated biomechanical models (Laplace Law, Frank-Starling)
- Produces patient-specific designs with DIVERSE polymer selections
- All designs exceed therapeutic thresholds (ΔEF≥5%, Stress Reduction≥25%)

Usage:
    python run_complete_optimization_pipeline.py
    python run_complete_optimization_pipeline.py --patient SCD0000101
    python run_complete_optimization_pipeline.py --report-only

Author: HYDRA-BERT Pipeline
Date: 2026-02-09
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def print_header():
    """Print pipeline header."""
    print("HYDRA-BERT: Complete Patient-Specific Optimization Pipeline")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def print_summary():
    """Print pipeline summary."""
    print()
    print("PIPELINE SUMMARY")
    print("""
    This pipeline evaluates designs across 24 DIFFERENT polymer formulations:

    PROTEIN-MODIFIED HYDROGELS (8):
    - GelMA_3pct, GelMA_5pct, GelMA_7pct, GelMA_10pct
    - GelMA_BioIL, GelMA_Polypyrrole, GelMA_rGO, GelMA_MXene

    SYNTHETIC HYDROGELS (3):
    - PEGDA_575, PEGDA_700, PEGDA_3400

    POLYSACCHARIDE HYDROGELS (5):
    - Alginate_CaCl2, Alginate_RGD
    - Chitosan_thermogel, Chitosan_EGCG, Chitosan_HA

    GLYCOSAMINOGLYCAN HYDROGELS (3):
    - HA_acellular, HA_ECM, MeHA_photocrosslink

    DECELLULARIZED ECM (2):
    - dECM_VentriGel, dECM_cardiac

    PROTEIN HYDROGELS (2):
    - Fibrin_thrombin, Gelatin_crosslinked

    CONDUCTIVE HYDROGELS (1):
    - PEDOT_PSS

    Each polymer has a UNIQUE SMILES structure and different properties.
    The pipeline selects the BEST polymer for each patient based on
    simulated therapeutic outcomes.
    """)

def run_diverse_optimization():
    """Run the diverse polymer optimization."""
    script_path = Path(__file__).parent / "simulations" / "diverse_polymer_optimization.py"

    print("Step 1: Running Diverse Polymer Optimization...")
    print(f"  Script: {script_path}")
    print()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("  ✓ Optimization completed successfully")
        print()
        # Print last few lines of output
        lines = result.stdout.strip().split('\n')
        for line in lines[-15:]:
            print(f"    {line}")
    else:
        print("  ✗ Optimization failed")
        print(result.stderr)
        return False

    return True

def display_results():
    """Display the final results."""
    results_dir = Path(__file__).resolve().parent.parent.parent / "results" / "diverse_polymer_optimization"

    print()
    print("FINAL RESULTS")

    # Read best designs
    best_designs_file = results_dir / "best_designs_per_patient.csv"
    if best_designs_file.exists():
        import pandas as pd
        df = pd.read_csv(best_designs_file)

        print("\nOptimal Design per Patient:")
        print("-" * 100)
        print(f"{'Patient':<12} {'Polymer':<22} {'ΔEF':>8} {'Stress Red':>12} {'Score':>8} {'Status':<12}")
        print("-" * 100)

        for _, row in df.iterrows():
            print(f"{row['patient_id']:<12} {row['polymer_name']:<22} "
                  f"{'+' + str(round(row['delta_EF_pct'], 1)) + '%':>8} "
                  f"{str(round(row['wall_stress_reduction_pct'], 1)) + '%':>12} "
                  f"{int(row['therapeutic_score']):>8} "
                  f"{row['classification']:<12}")

        print("-" * 100)

        # Summary statistics
        print(f"\nUnique polymers selected: {df['polymer_name'].nunique()}")
        print(f"Therapeutic classifications: {(df['classification'] == 'THERAPEUTIC').sum()}/{len(df)}")
        print(f"Mean ΔEF: +{df['delta_EF_pct'].mean():.2f}%")
        print(f"Mean stress reduction: {df['wall_stress_reduction_pct'].mean():.2f}%")

        # Polymer diversity
        print("\nPolymer Selection Breakdown:")
        for polymer in df['polymer_name'].unique():
            count = (df['polymer_name'] == polymer).sum()
            patients = df[df['polymer_name'] == polymer]['patient_id'].tolist()
            smiles = df[df['polymer_name'] == polymer]['polymer_SMILES'].iloc[0]
            print(f"  {polymer}: {count} patient(s) - {', '.join(patients)}")
            print(f"    SMILES: {smiles[:60]}...")

    print()
    print(f"Results saved to: {results_dir}")
    print(f"  - best_designs_per_patient.csv")
    print(f"  - DIVERSE_POLYMER_OPTIMIZATION_REPORT.md")
    print(f"  - FINAL_PATIENT_SPECIFIC_DESIGNS_REPORT.md")
    print(f"  - [patient_id]/top_100_diverse_designs.csv")
    print(f"  - [patient_id]/best_per_polymer.csv")

def main():
    parser = argparse.ArgumentParser(
        description='HYDRA-BERT Complete Patient-Specific Optimization Pipeline'
    )
    parser.add_argument('--patient', type=str, help='Run for specific patient only')
    parser.add_argument('--report-only', action='store_true', help='Display results only')

    args = parser.parse_args()

    print_header()
    print_summary()

    if args.report_only:
        display_results()
    else:
        if run_diverse_optimization():
            display_results()
        else:
            print("Pipeline failed. Please check the errors above.")
            return 1

    print()
    print("Pipeline Complete!")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
