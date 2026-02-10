"""
HYDRA-BERT Simulation Module

This module contains scripts for running ACTUAL finite element simulations
(not surrogate models) using FEBio and OpenCarp.

Scripts:
    - run_actual_febio_hydrogel.py: FEBio cardiac mechanics with hydrogel
    - run_actual_opencarp_hydrogel.py: OpenCarp electrophysiology with hydrogel
    - run_complete_simulations.py: Master script for both simulations
    - extract_simulation_metrics.py: Metric extraction from simulation outputs

Usage:
    # Run all simulations with all CPUs
    python scripts/simulations/run_complete_simulations.py --all_cpus

    # Extract metrics after simulation
    python scripts/simulations/extract_simulation_metrics.py --all

Author: HYDRA-BERT Pipeline
Date: 2026-02-09
"""

from pathlib import Path

# Module paths
MODULE_DIR = Path(__file__).parent
RESULTS_DIR = MODULE_DIR.parent.parent / "results"

# Simulation types
SIMULATION_TYPES = ["febio", "opencarp"]

# Default patient list
DEFAULT_PATIENTS = [
    "SCD0000101", "SCD0000201", "SCD0000301", "SCD0000401",
    "SCD0000601", "SCD0000701", "SCD0000801", "SCD0001001",
    "SCD0001101", "SCD0001201"
]
