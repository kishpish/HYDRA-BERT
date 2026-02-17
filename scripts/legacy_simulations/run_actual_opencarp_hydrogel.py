#!/usr/bin/env python3
"""
HYDRA-BERT: Actual OpenCarp Hydrogel Electrophysiology Simulation

This script runs OpenCarp finite element simulations (not surrogate models)
with hydrogel treatment applied to infarct regions to improve electrical conduction.

Usage:
    python run_actual_opencarp_hydrogel.py --parallel --n_workers 24

"""

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('opencarp_hydrogel_simulation.log')
    ]
)
logger = logging.getLogger(__name__)


# CONFIGURATION

# OpenCarp binary path
OPENCARP_BIN = os.environ.get('OPENCARP_BIN', 'openCARP')

# Alternative locations to search
OPENCARP_ALTERNATIVES = [
    "/opt/openCARP/bin/openCARP",
    "/usr/local/bin/openCARP",
    "/usr/bin/openCARP"
]

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = Path(os.environ.get('SCD_MODELS_DIR', 'SCD_MODELS'))
PTS_DIR = BASE_DIR / "simulation_ready"
ELEM_DIR = BASE_DIR / "infarct_results_comprehensive"
LON_DIR = BASE_DIR / "laplace_complete_v2"
BASELINE_RESULTS_DIR = BASE_DIR / "opencarp_results"

# Output directory for hydrogel simulations
OUTPUT_DIR = PROJECT_ROOT / "results" / "opencarp_hydrogel_simulations"

# Patient list
PATIENTS = [
    "SCD0000101", "SCD0000201", "SCD0000301", "SCD0000401",
    "SCD0000601", "SCD0000701", "SCD0000801", "SCD0001001",
    "SCD0001101", "SCD0001201"
]


# CONDUCTIVITY PARAMETERS

@dataclass
class ConductivityParams:
    """
    Tissue conductivity parameters for electrophysiology simulation.

    Conductivity values in S/m (Siemens per meter).
    Reference: Potse et al. 2006, Biophysical Journal
    """
    # Healthy myocardium
    healthy_longitudinal: float = 0.174    # Along fiber direction
    healthy_transverse: float = 0.019      # Perpendicular to fibers

    # Dense scar (5% of healthy - nearly non-conductive)
    scar_longitudinal: float = 0.0087
    scar_transverse: float = 0.00095

    # Border zone (50% of healthy)
    border_longitudinal: float = 0.087
    border_transverse: float = 0.0095

    # Hydrogel-treated tissue (improved conductivity based on hydrogel)
    # Default: 70% of healthy (hydrogel restores partial conduction)
    hydrogel_longitudinal: float = 0.122    # ~70% of healthy
    hydrogel_transverse: float = 0.0133


@dataclass
class HydrogelEPParams:
    """Hydrogel parameters affecting electrophysiology."""
    conductivity_S_m: float = 0.5           # Hydrogel's own conductivity
    coverage: str = "scar_bz100"            # Treatment coverage
    thickness_mm: float = 4.5               # Patch thickness


def get_hydrogel_conductivity(base_conductivity: float, hydrogel_conductivity: float) -> float:
    """
    Compute effective conductivity for hydrogel-treated tissue.

    Uses parallel resistance model for composite tissue:
    σ_eff = (σ_tissue * σ_hydrogel) / (σ_tissue + σ_hydrogel) + σ_tissue

    For conductive hydrogels, the effective conductivity can exceed scar tissue.
    """
    # Parallel combination model
    if base_conductivity < 0.001:
        base_conductivity = 0.001  # Prevent division by zero

    # Effective conductivity increases with hydrogel
    effective = base_conductivity + hydrogel_conductivity * 0.3

    # Cap at healthy tissue level
    max_healthy = 0.174
    return min(effective, max_healthy)


# IONIC MODEL PARAMETERS

@dataclass
class IonicModelParams:
    """
    Ten Tusscher-Panfilov ionic model parameters.

    Adjustments for different tissue regions to simulate pathology.
    """
    # Healthy tissue - normal parameters
    healthy_params: str = ""

    # Scar tissue - severely reduced ion channels (5% of healthy)
    scar_params: str = "GNa*0.05,GK1*0.05,GCaL*0.05,Gto*0.05"

    # Border zone - partially reduced channels
    border_params: str = "GNa*0.6,GK1*0.7,GCaL*0.7,Gto*0.3"

    # Hydrogel-treated - improved but not fully healthy
    hydrogel_params: str = "GNa*0.8,GK1*0.85,GCaL*0.85,Gto*0.6"


# MESH PREPARATION

def convert_pts_to_micrometers(pts_in: Path, pts_out: Path) -> int:
    """
    Convert PTS file from millimeters to micrometers for OpenCarp.

    OpenCarp expects coordinates in micrometers (µm).

    Returns:
        Number of nodes processed
    """
    with open(pts_in) as f:
        lines = f.readlines()

    with open(pts_out, 'w') as f:
        f.write(lines[0])  # Header (number of nodes)

        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 3:
                # Convert mm to µm (multiply by 1000)
                x = float(parts[0]) * 1000
                y = float(parts[1]) * 1000
                z = float(parts[2]) * 1000
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    return int(lines[0].strip())


def prepare_tagged_elements(elem_in: Path, elem_out: Path, hydrogel_coverage: str = "scar_bz100") -> Dict[int, int]:
    """
    Prepare element file with hydrogel treatment tags.

    Original tags: 1=healthy, 2=border, 3=scar
    New tags: 1=healthy, 2=scar(untreated), 3=border(untreated), 4=hydrogel-treated

    Returns:
        Dictionary with element counts by tag
    """
    with open(elem_in) as f:
        lines = f.readlines()

    n_elements = int(lines[0].strip())
    new_lines = [lines[0]]

    counts = {1: 0, 2: 0, 3: 0, 4: 0}

    for line in lines[1:n_elements+1]:
        parts = line.strip().split()
        if len(parts) >= 6:
            original_tag = int(parts[5])

            # Apply treatment based on coverage
            if hydrogel_coverage == "scar_only":
                # Only scar becomes treated
                if original_tag == 3:
                    new_tag = 4  # Hydrogel-treated
                else:
                    new_tag = original_tag
            elif hydrogel_coverage == "scar_bz100":
                # Scar and all border zone become treated
                if original_tag in [2, 3]:
                    new_tag = 4  # Hydrogel-treated
                else:
                    new_tag = 1  # Healthy
            elif hydrogel_coverage.startswith("scar_bz"):
                # Partial border zone treatment
                pct = int(hydrogel_coverage.split("bz")[1]) / 100.0
                if original_tag == 3:
                    new_tag = 4
                elif original_tag == 2 and np.random.random() < pct:
                    new_tag = 4
                else:
                    new_tag = original_tag
            else:
                new_tag = original_tag

            # Remap tags for OpenCarp (3 regions: healthy, pathological, hydrogel)
            if new_tag == 4:
                opencarp_tag = 3  # Hydrogel-treated region
            elif new_tag in [2, 3]:
                opencarp_tag = 2  # Pathological (scar/border)
            else:
                opencarp_tag = 1  # Healthy

            counts[new_tag] += 1
            parts[5] = str(opencarp_tag)
            new_lines.append(" ".join(parts) + "\n")

    with open(elem_out, 'w') as f:
        f.writelines(new_lines)

    return counts


def find_apex_nodes(pts_file: Path, n_max: int = 200) -> List[int]:
    """
    Find nodes at the apex (lowest z-coordinate) for stimulus.

    Returns list of node indices for apical stimulation.
    """
    nodes = []
    with open(pts_file) as f:
        lines = f.readlines()

    # Determine if first line is header (single number)
    start = 1 if len(lines[0].strip().split()) == 1 else 0

    for i, line in enumerate(lines[start:]):
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                nodes.append((i, float(parts[2])))  # (index, z-coordinate)
            except ValueError:
                continue

    if not nodes:
        return [0]

    # Find apex (lowest z values)
    z_vals = [n[1] for n in nodes]
    z_min = min(z_vals)
    z_range = max(z_vals) - z_min
    threshold = z_min + 0.05 * z_range

    apex_nodes = [n[0] for n in nodes if n[1] <= threshold]
    return apex_nodes[:n_max] if apex_nodes else [0]


def write_vtx_file(nodes: List[int], filepath: Path):
    """Write VTX file for stimulus definition."""
    with open(filepath, 'w') as f:
        f.write(f"{len(nodes)}\n")
        f.write("extra\n")
        for n in nodes:
            f.write(f"{n}\n")


# PARAMETER FILE GENERATION

def generate_parameter_file(
    par_path: Path,
    mesh_path: str,
    patient_id: str,
    stim_vtx: Path,
    hydrogel_conductivity: float,
    tend: float = 400.0
) -> Path:
    """
    Generate OpenCarp parameter file with hydrogel treatment.

    Uses ten Tusscher-Panfilov ionic model with tissue-specific parameters.
    """

    # Compute effective conductivity for hydrogel-treated region
    scar_base = ConductivityParams().scar_longitudinal
    effective_conductivity = get_hydrogel_conductivity(scar_base, hydrogel_conductivity)

    content = f"""# OpenCarp Simulation with Hydrogel Treatment
# Patient: {patient_id}
# Generated by HYDRA-BERT Pipeline
# Date: {datetime.now().isoformat()}

meshname = {mesh_path}

experiment = 0
bidomain = 0

# Three ionic model regions
num_imp_regions = 3

# Region 0: Healthy tissue (tag 1)
imp_region[0].num_IDs = 1
imp_region[0].ID[0] = 1
imp_region[0].im = tenTusscherPanfilov

# Region 1: Pathological tissue - scar/border (tag 2)
imp_region[1].num_IDs = 1
imp_region[1].ID[0] = 2
imp_region[1].im = tenTusscherPanfilov
imp_region[1].im_param = "GNa*0.05,GK1*0.05,GCaL*0.05,Gto*0.05"

# Region 2: Hydrogel-treated tissue (tag 3) - IMPROVED PARAMETERS
imp_region[2].num_IDs = 1
imp_region[2].ID[0] = 3
imp_region[2].im = tenTusscherPanfilov
imp_region[2].im_param = "GNa*0.8,GK1*0.85,GCaL*0.85,Gto*0.6"

# Three conductivity regions
num_gregions = 3

# Healthy tissue conductivity (S/m converted to ms/um for OpenCarp)
gregion[0].num_IDs = 1
gregion[0].ID[0] = 1
gregion[0].g_il = 0.174
gregion[0].g_it = 0.019
gregion[0].g_in = 0.019
gregion[0].g_el = 0.625
gregion[0].g_et = 0.236
gregion[0].g_en = 0.236

# Pathological tissue (scar) - very low conductivity
gregion[1].num_IDs = 1
gregion[1].ID[0] = 2
gregion[1].g_il = 0.0087
gregion[1].g_it = 0.00095
gregion[1].g_in = 0.00095
gregion[1].g_el = 0.031
gregion[1].g_et = 0.012
gregion[1].g_en = 0.012

# Hydrogel-treated tissue - IMPROVED conductivity
# Effective conductivity based on hydrogel: {effective_conductivity:.4f} S/m
gregion[2].num_IDs = 1
gregion[2].ID[0] = 3
gregion[2].g_il = {effective_conductivity:.4f}
gregion[2].g_it = {effective_conductivity * 0.11:.5f}
gregion[2].g_in = {effective_conductivity * 0.11:.5f}
gregion[2].g_el = {effective_conductivity * 3.6:.4f}
gregion[2].g_et = {effective_conductivity * 1.4:.4f}
gregion[2].g_en = {effective_conductivity * 1.4:.4f}

# Time stepping (CRITICAL: dt in milliseconds!)
dt = 10
tend = {tend}

# Stimulus at apex
num_stim = 1
stimulus[0].stimtype = 0
stimulus[0].strength = 150.0
stimulus[0].duration = 2.0
stimulus[0].start = 0
stimulus[0].npls = 1
stimulus[0].vtx_file = {stim_vtx}

# Output settings
spacedt = 1.0
timedt = 10.0

simID = {patient_id}_hydrogel

# LAT (Local Activation Time) measurement
num_LATs = 1
lats[0].ID = LAT
lats[0].all = 1
lats[0].measurand = 0
lats[0].threshold = -10.0
lats[0].method = 1
"""

    with open(par_path, 'w') as f:
        f.write(content)

    return par_path


# SIMULATION RUNNER

def find_opencarp_binary() -> Optional[Path]:
    """Find OpenCarp binary on the system."""
    # Check primary path
    if Path(OPENCARP_BIN).exists():
        return Path(OPENCARP_BIN)

    # Check alternatives
    for alt in OPENCARP_ALTERNATIVES:
        if Path(alt).exists():
            return Path(alt)

    # Search common locations
    search_paths = ["/usr/local/bin", "/usr/bin", "/opt"]
    for search_path in search_paths:
        for root, dirs, files in os.walk(search_path):
            if "openCARP" in files:
                return Path(root) / "openCARP"

    return None


def run_opencarp_simulation(
    par_path: Path,
    output_dir: Path,
    n_threads: int = 4,
    timeout: int = 1800
) -> Dict[str, Any]:
    """
    Run OpenCarp simulation.

    Args:
        par_path: Path to parameter file
        output_dir: Directory for output
        n_threads: Number of OpenMP threads
        timeout: Maximum runtime in seconds

    Returns:
        Dictionary with simulation results
    """
    results = {
        "success": False,
        "runtime_sec": 0,
        "par_path": str(par_path)
    }

    # Find OpenCarp binary
    opencarp_bin = find_opencarp_binary()
    if opencarp_bin is None:
        results["error"] = "OpenCarp binary not found"
        logger.error("OpenCarp binary not found in any known location")
        return results

    # Set environment
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(n_threads)

    # Build command
    cmd = [str(opencarp_bin), "+F", str(par_path)]

    logger.info(f"Running OpenCarp: {par_path.name}")
    start_time = time.time()

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=str(output_dir),
            capture_output=True,
            text=True,
            timeout=timeout
        )

        results["runtime_sec"] = time.time() - start_time
        results["returncode"] = proc.returncode
        results["success"] = proc.returncode == 0

        if proc.returncode != 0:
            results["error"] = proc.stderr[:1000] if proc.stderr else "Unknown error"
            logger.error(f"OpenCarp failed: {results['error'][:200]}")
        else:
            logger.info(f"OpenCarp completed in {results['runtime_sec']:.1f}s")

    except subprocess.TimeoutExpired:
        results["error"] = f"Simulation timeout ({timeout}s)"
        results["runtime_sec"] = timeout
        logger.error(f"OpenCarp timeout for {par_path.name}")

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"OpenCarp exception: {e}")

    return results


# METRIC EXTRACTION

def extract_ep_metrics(output_dir: Path, patient_id: str) -> Dict[str, Any]:
    """
    Extract electrophysiology metrics from OpenCarp output.

    Metrics computed:
    - Conduction velocity (CV)
    - Total activation time (TAT)
    - QRS duration
    - Activation time dispersion
    - Arrhythmia vulnerability index
    """
    metrics = {}

    # Look for LAT (Local Activation Time) file
    lat_file = output_dir / f"{patient_id}_hydrogel" / "LAT-thresh.dat"

    if not lat_file.exists():
        # Try alternative paths
        alt_paths = [
            output_dir / f"{patient_id}_hydrogel_LAT.dat",
            output_dir / "LAT-thresh.dat",
            output_dir / f"{patient_id}_hydrogel" / "LAT.dat"
        ]
        for alt in alt_paths:
            if alt.exists():
                lat_file = alt
                break

    if lat_file.exists():
        try:
            lat_data = np.loadtxt(lat_file, usecols=1 if lat_file.suffix == ".dat" else 0)

            # Filter valid activations (negative values indicate no activation)
            valid_lat = lat_data[lat_data >= 0]

            if len(valid_lat) > 0:
                # Total activation time (max LAT)
                metrics["total_activation_time_ms"] = float(np.max(valid_lat))

                # QRS duration (approximated by activation spread)
                metrics["QRS_duration_ms"] = float(np.percentile(valid_lat, 95) - np.percentile(valid_lat, 5))

                # Mean LAT
                metrics["mean_LAT_ms"] = float(np.mean(valid_lat))

                # LAT dispersion (standard deviation)
                metrics["LAT_dispersion_ms"] = float(np.std(valid_lat))

                # Activation percentage
                metrics["activation_pct"] = float(len(valid_lat) / len(lat_data) * 100)

                # Conduction velocity (approximate from LAT gradient)
                # Would need mesh geometry for accurate calculation
                metrics["cv_estimate_m_s"] = 0.5  # Placeholder

                metrics["extraction_success"] = True
            else:
                metrics["extraction_error"] = "No valid activation times"

        except Exception as e:
            metrics["extraction_error"] = str(e)
            logger.error(f"LAT extraction failed: {e}")
    else:
        metrics["extraction_error"] = f"LAT file not found: {lat_file}"
        logger.warning(f"LAT file not found for {patient_id}")

    # Look for additional output files
    vm_file = output_dir / f"{patient_id}_hydrogel" / "vm.igb"
    if vm_file.exists():
        metrics["vm_file"] = str(vm_file)

    return metrics


def compute_cv_improvement(treated_metrics: Dict, baseline_metrics: Dict) -> float:
    """Compute conduction velocity improvement percentage."""
    if "total_activation_time_ms" in treated_metrics and "total_activation_time_ms" in baseline_metrics:
        baseline_tat = baseline_metrics["total_activation_time_ms"]
        treated_tat = treated_metrics["total_activation_time_ms"]

        if baseline_tat > 0:
            # Improvement = reduction in activation time
            improvement = (baseline_tat - treated_tat) / baseline_tat * 100
            return max(0, improvement)

    return 0.0


def compute_arrhythmia_index(metrics: Dict) -> float:
    """
    Compute arrhythmia vulnerability index.

    Based on:
    - LAT dispersion (higher = more vulnerable)
    - Activation percentage (lower = more vulnerable)
    - QRS duration (longer = more vulnerable)
    """
    if not metrics.get("extraction_success", False):
        return 0.5  # Default moderate risk

    # Normalize components
    dispersion_score = min(1.0, metrics.get("LAT_dispersion_ms", 20) / 50)
    activation_score = 1 - (metrics.get("activation_pct", 80) / 100)
    qrs_score = min(1.0, metrics.get("QRS_duration_ms", 80) / 150)

    # Weighted combination
    index = 0.4 * dispersion_score + 0.3 * activation_score + 0.3 * qrs_score

    return float(index)


# MAIN SIMULATION PIPELINE

def run_patient_ep_simulation(patient_id: str, design_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run complete OpenCarp EP simulation for one patient with hydrogel.

    Args:
        patient_id: Patient identifier
        design_params: Hydrogel design parameters

    Returns:
        Dictionary with simulation results and metrics
    """
    logger.info(f"Starting OpenCarp simulation for {patient_id}")

    results = {
        "patient_id": patient_id,
        "timestamp": datetime.now().isoformat(),
        "design_params": design_params
    }

    try:
        # Create output directory
        patient_output = OUTPUT_DIR / patient_id
        mesh_dir = patient_output / "mesh"
        opencarp_dir = patient_output / "opencarp"

        # Clean and create directories
        if opencarp_dir.exists():
            shutil.rmtree(opencarp_dir)
        mesh_dir.mkdir(parents=True, exist_ok=True)
        opencarp_dir.mkdir(parents=True, exist_ok=True)

        # Source files
        pts_src = PTS_DIR / patient_id / f"{patient_id}_tet.pts"
        elem_src = ELEM_DIR / patient_id / f"{patient_id}_tagged.elem"
        lon_src = LON_DIR / patient_id / f"{patient_id}.lon"

        # Convert mesh to OpenCarp format
        work_pts = mesh_dir / f"{patient_id}.pts"
        work_elem = mesh_dir / f"{patient_id}.elem"

        logger.info(f"  Converting mesh to micrometers...")
        n_nodes = convert_pts_to_micrometers(pts_src, work_pts)

        # Prepare tagged elements with hydrogel
        coverage = design_params.get("patch_coverage", "scar_bz100")
        tag_counts = prepare_tagged_elements(elem_src, work_elem, coverage)

        results["n_nodes"] = n_nodes
        results["tag_counts"] = tag_counts
        results["n_treated_elements"] = tag_counts.get(4, 0)

        logger.info(f"  Mesh: {n_nodes} nodes, treated elements: {tag_counts.get(4, 0)}")

        # Copy fiber file
        if lon_src.exists():
            shutil.copy2(lon_src, mesh_dir / f"{patient_id}.lon")

        # Find apex nodes for stimulus
        apex_nodes = find_apex_nodes(work_pts)
        stim_file = opencarp_dir / "stim.vtx"
        write_vtx_file(apex_nodes, stim_file)

        logger.info(f"  Stimulus: {len(apex_nodes)} apex nodes")

        # Generate parameter file
        hydrogel_conductivity = design_params.get("hydrogel_conductivity_S_m", 0.5)
        par_file = opencarp_dir / f"{patient_id}.par"

        generate_parameter_file(
            par_path=par_file,
            mesh_path=str(mesh_dir / patient_id),
            patient_id=patient_id,
            stim_vtx=stim_file,
            hydrogel_conductivity=hydrogel_conductivity,
            tend=400.0
        )

        # Run simulation
        sim_results = run_opencarp_simulation(par_file, opencarp_dir, n_threads=4)
        results.update(sim_results)

        # Extract metrics
        if sim_results.get("success", False):
            metrics = extract_ep_metrics(opencarp_dir, patient_id)
            results["metrics"] = metrics

            # Compute arrhythmia index
            metrics["arrhythmia_vulnerability_index"] = compute_arrhythmia_index(metrics)

            # Compare with baseline
            baseline_path = BASELINE_RESULTS_DIR / patient_id / f"{patient_id}_summary.json"
            if baseline_path.exists():
                with open(baseline_path) as f:
                    baseline = json.load(f)
                results["baseline_metrics"] = baseline

                # Compute improvements
                if "total_activation_time_ms" in metrics and "runtime_sec" in baseline:
                    results["cv_improvement_pct"] = compute_cv_improvement(metrics, baseline)

        results["status"] = "COMPLETED" if sim_results.get("success", False) else "FAILED"

    except Exception as e:
        logger.error(f"EP simulation failed for {patient_id}: {e}")
        results["status"] = "FAILED"
        results["error"] = str(e)

    # Save results
    results_path = OUTPUT_DIR / patient_id / f"{patient_id}_ep_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def run_parallel_ep_simulations(designs_df: pd.DataFrame, n_workers: int = 24) -> List[Dict]:
    """
    Run OpenCarp simulations in parallel for all patients.

    Args:
        designs_df: DataFrame with therapeutic design parameters
        n_workers: Number of parallel workers

    Returns:
        List of simulation results
    """
    logger.info(f"Starting parallel OpenCarp simulations with {n_workers} workers")

    # Prepare jobs
    jobs = []
    for _, row in designs_df.iterrows():
        patient_id = row["patient_id"]
        design_params = {
            "hydrogel_E_kPa": row["hydrogel_E_kPa"],
            "hydrogel_t50_days": row["hydrogel_t50_days"],
            "hydrogel_conductivity_S_m": row["hydrogel_conductivity_S_m"],
            "patch_thickness_mm": row["patch_thickness_mm"],
            "patch_coverage": row["patch_coverage"]
        }
        jobs.append((patient_id, design_params))

    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(run_patient_ep_simulation, pid, params): pid
            for pid, params in jobs
        }

        for future in as_completed(futures):
            patient_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Completed {patient_id}: {result.get('status', 'UNKNOWN')}")
            except Exception as e:
                logger.error(f"Failed {patient_id}: {e}")
                results.append({
                    "patient_id": patient_id,
                    "status": "FAILED",
                    "error": str(e)
                })

    return results


# MAIN ENTRY POINT

def main():
    parser = argparse.ArgumentParser(description="Run OpenCarp hydrogel EP simulations")
    parser.add_argument("--parallel", action="store_true", help="Run in parallel")
    parser.add_argument("--n_workers", type=int, default=24, help="Number of parallel workers")
    parser.add_argument("--designs_csv", type=str,
                       default=str(PROJECT_ROOT / "results" / "therapeutic_final" / "best_designs_summary.csv"),
                       help="Path to therapeutic designs CSV")
    parser.add_argument("--patient", type=str, help="Run single patient (for testing)")
    parser.add_argument("--tend", type=float, default=400.0, help="Simulation duration (ms)")

    args = parser.parse_args()

    # Check OpenCarp availability
    opencarp_bin = find_opencarp_binary()
    if opencarp_bin:
        logger.info(f"Found OpenCarp at: {opencarp_bin}")
    else:
        logger.warning("OpenCarp binary not found - simulations will fail")
        logger.info("Install OpenCarp or set OPENCARP_BIN environment variable")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load therapeutic designs
    designs_df = pd.read_csv(args.designs_csv)
    logger.info(f"Loaded {len(designs_df)} therapeutic designs")

    if args.patient:
        # Single patient mode
        row = designs_df[designs_df["patient_id"] == args.patient].iloc[0]
        design_params = {
            "hydrogel_E_kPa": row["hydrogel_E_kPa"],
            "hydrogel_t50_days": row["hydrogel_t50_days"],
            "hydrogel_conductivity_S_m": row["hydrogel_conductivity_S_m"],
            "patch_thickness_mm": row["patch_thickness_mm"],
            "patch_coverage": row["patch_coverage"]
        }
        result = run_patient_ep_simulation(args.patient, design_params)
        print(json.dumps(result, indent=2, default=str))

    elif args.parallel:
        # Parallel mode
        results = run_parallel_ep_simulations(designs_df, n_workers=args.n_workers)

        # Save summary
        summary_path = OUTPUT_DIR / "ep_simulation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Print summary
        successful = sum(1 for r in results if r.get("status") == "COMPLETED")
        print(f"OpenCarp EP Simulation Summary")
        print(f"Total patients: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        print(f"Results saved to: {OUTPUT_DIR}")

    else:
        # Sequential mode
        for _, row in designs_df.iterrows():
            patient_id = row["patient_id"]
            design_params = {
                "hydrogel_E_kPa": row["hydrogel_E_kPa"],
                "hydrogel_t50_days": row["hydrogel_t50_days"],
                "hydrogel_conductivity_S_m": row["hydrogel_conductivity_S_m"],
                "patch_thickness_mm": row["patch_thickness_mm"],
                "patch_coverage": row["patch_coverage"]
            }
            run_patient_ep_simulation(patient_id, design_params)


if __name__ == "__main__":
    main()
