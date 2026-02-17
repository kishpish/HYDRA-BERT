"""
HYDRA-BERT Pipeline Scripts

Complete patient-specific hydrogel design optimization pipeline:

1. generate_10M_designs.py - Generate 10 million designs per patient
2. run_febio_simulations.py - Run FEBio mechanical simulations
3. run_opencarp_simulations.py - Run OpenCarp EP simulations
4. select_optimal_design.py - Select optimal design from results
5. run_full_pipeline.py - Master orchestration script

Usage:
    python run_full_pipeline.py --all --gpus 16 --cpus 96
"""

from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
BASE_DIR = PIPELINE_DIR.parent.parent
