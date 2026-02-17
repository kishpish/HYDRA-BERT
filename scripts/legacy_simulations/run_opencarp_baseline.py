#!/usr/bin/env python3
"""
OpenCARP Simulation v7 - verified & corrected


VERIFIED AGAINST DOCUMENTATION:
- dt is in MICROSECONDS (μs), not milliseconds
- dt=10 means 10μs = 0.01ms (40,000 timesteps for 400ms)
- tend is in milliseconds (ms)
- Mesh coordinates are in micrometers (μm)

FIXES FROM v6:
- Corrected CV calculation (proper unit handling)
- Verified solver is working correctly (1000 ODE steps per 10ms output)

METRICS COMPUTED:
- Local Activation Time (LAT)
- QRS duration (total activation time)
- Conduction Velocity (CV) - per region
- Per-region activation statistics
- Activation delays

"""

import os
import subprocess
import shutil
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
OPENCARP_BIN = "/usr/local/bin/openCARP"
BASE_DIR = Path(os.environ.get('SCD_MODELS_DIR', 'SCD_MODELS'))


@dataclass
class ConductivityParams:
    """
    Conductivity parameters (S/m) based on literature
    - Clerc 1976, Roberts & Scher 1982
    """
    # Healthy myocardium (Tag 1) - 100%
    healthy_il: float = 0.174
    healthy_it: float = 0.019
    healthy_in: float = 0.019
    healthy_el: float = 0.625
    healthy_et: float = 0.236
    healthy_en: float = 0.236

    # Dense scar (Tag 2) - 5% of healthy
    scar_il: float = 0.0087
    scar_it: float = 0.00095
    scar_in: float = 0.00095
    scar_el: float = 0.031
    scar_et: float = 0.012
    scar_en: float = 0.012

    # Border zone (Tag 3) - 50% of healthy
    border_il: float = 0.087
    border_it: float = 0.0095
    border_in: float = 0.0095
    border_el: float = 0.312
    border_et: float = 0.118
    border_en: float = 0.118


def get_num_cpus() -> int:
    """Get number of available CPUs"""
    try:
        return os.cpu_count() or 4
    except:
        return 4


def check_mesh_units(pts_file: Path) -> Tuple[str, float]:
    """Check if mesh is in micrometers or millimeters"""
    try:
        with open(pts_file, 'r') as f:
            lines = f.readlines()

        start = 1 if len(lines[0].strip().split()) == 1 else 0

        coords = []
        for line in lines[start:start+1000]:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except:
                    continue

        if not coords:
            return "unknown", 0.0

        coords = np.array(coords)
        ranges = coords.max(axis=0) - coords.min(axis=0)
        max_range = max(ranges)

        if max_range > 10000:
            return "um", max_range
        elif max_range > 10:
            return "mm", max_range
        else:
            return "other", max_range

    except Exception as e:
        logger.error(f"Error checking mesh: {e}")
        return "unknown", 0.0


def convert_pts_to_um(pts_file: Path, output_file: Path, scale: float = 1000.0):
    """Convert pts file from mm to micrometers"""
    with open(pts_file, 'r') as f:
        lines = f.readlines()

    start = 1 if len(lines[0].strip().split()) == 1 else 0

    with open(output_file, 'w') as f:
        if start == 1:
            f.write(lines[0])

        for line in lines[start:]:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    x = float(parts[0]) * scale
                    y = float(parts[1]) * scale
                    z = float(parts[2]) * scale
                    f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
                except:
                    f.write(line)


def get_mesh_info(pts_file: Path, elem_file: Path) -> Dict:
    """Get mesh statistics"""
    info = {"n_nodes": 0, "n_elements": 0, "tags": {}}

    try:
        with open(pts_file) as f:
            first = f.readline().strip().split()
            info["n_nodes"] = int(first[0]) if len(first) == 1 else 1 + sum(1 for _ in f)
    except:
        pass

    try:
        with open(elem_file) as f:
            first = f.readline().strip().split()
            if len(first) == 1:
                info["n_elements"] = int(first[0])
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            tag = int(parts[-1])
                            info["tags"][tag] = info["tags"].get(tag, 0) + 1
                        except:
                            pass
    except:
        pass

    return info


def find_apex_nodes(pts_file: Path, n_max: int = 200) -> List[int]:
    """Find lowest z-coordinate nodes for stimulus"""
    nodes = []
    try:
        with open(pts_file) as f:
            lines = f.readlines()
        start = 1 if len(lines[0].strip().split()) == 1 else 0
        for i, line in enumerate(lines[start:]):
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    nodes.append((i, float(parts[2])))
                except:
                    continue
    except:
        return [0]

    if not nodes:
        return [0]

    z_vals = [n[1] for n in nodes]
    z_min = min(z_vals)
    z_range = max(z_vals) - z_min

    threshold = z_min + 0.05 * z_range
    apex = [n[0] for n in nodes if n[1] <= threshold]

    if len(apex) < 10:
        threshold = z_min + 0.10 * z_range
        apex = [n[0] for n in nodes if n[1] <= threshold]

    return apex[:n_max]


def write_vtx_file(nodes: List[int], filepath: Path):
    """Write VTX file with 'extra' domain type"""
    with open(filepath, 'w') as f:
        f.write(f"{len(nodes)}\n")
        f.write("extra\n")
        for n in nodes:
            f.write(f"{n}\n")


def write_parameter_file(par_path: Path, mesh_path: str, patient_id: str,
                         apex_vtx: Path, tend: float = 400.0) -> Path:
    """
    Write OpenCARP v18.1 parameter file

    VERIFIED: dt is in MICROSECONDS
    - dt=10 means 10 microseconds = 0.01 ms
    - For 400ms simulation: 40,000 timesteps
    """
    c = ConductivityParams()

    with open(par_path, 'w') as f:
        f.write(f"# OpenCARP v7 Simulation - VERIFIED PARAMETERS\n")
        f.write(f"# Patient: {patient_id}\n")
        f.write(f"# NOTE: dt is in MICROSECONDS (us), tend is in MILLISECONDS (ms)\n\n")

        f.write(f"simID = {patient_id}_v7\n")
        f.write(f"meshname = {mesh_path}\n\n")

        # Time parameters - VERIFIED UNITS
        f.write(f"# Time stepping (dt in MICROSECONDS, tend in MILLISECONDS)\n")
        f.write(f"dt = 10\n")  # 10 microseconds = 0.01 ms
        f.write(f"tend = {tend}\n\n")

        # Solver settings
        f.write(f"# Solver settings (monodomain)\n")
        f.write(f"bidomain = 0\n")
        f.write(f"parab_solve = 1\n")
        f.write(f"mass_lumping = 1\n")
        f.write(f"cg_tol_parab = 1e-6\n")
        f.write(f"cg_maxit_parab = 500\n\n")

        # Conductivity regions
        f.write(f"num_gregions = 3\n\n")

        # Healthy (Tag 1)
        f.write(f"# Healthy myocardium (Tag 1) - 100%\n")
        f.write(f"gregion[0].num_IDs = 1\n")
        f.write(f"gregion[0].ID[0] = 1\n")
        f.write(f"gregion[0].g_il = {c.healthy_il}\n")
        f.write(f"gregion[0].g_it = {c.healthy_it}\n")
        f.write(f"gregion[0].g_in = {c.healthy_in}\n")
        f.write(f"gregion[0].g_el = {c.healthy_el}\n")
        f.write(f"gregion[0].g_et = {c.healthy_et}\n")
        f.write(f"gregion[0].g_en = {c.healthy_en}\n\n")

        # Scar (Tag 2) - 5%
        f.write(f"# Dense scar (Tag 2) - 5% of healthy\n")
        f.write(f"gregion[1].num_IDs = 1\n")
        f.write(f"gregion[1].ID[0] = 2\n")
        f.write(f"gregion[1].g_il = {c.scar_il}\n")
        f.write(f"gregion[1].g_it = {c.scar_it}\n")
        f.write(f"gregion[1].g_in = {c.scar_in}\n")
        f.write(f"gregion[1].g_el = {c.scar_el}\n")
        f.write(f"gregion[1].g_et = {c.scar_et}\n")
        f.write(f"gregion[1].g_en = {c.scar_en}\n\n")

        # Border (Tag 3) - 50%
        f.write(f"# Border zone (Tag 3) - 50% of healthy\n")
        f.write(f"gregion[2].num_IDs = 1\n")
        f.write(f"gregion[2].ID[0] = 3\n")
        f.write(f"gregion[2].g_il = {c.border_il}\n")
        f.write(f"gregion[2].g_it = {c.border_it}\n")
        f.write(f"gregion[2].g_in = {c.border_in}\n")
        f.write(f"gregion[2].g_el = {c.border_el}\n")
        f.write(f"gregion[2].g_et = {c.border_et}\n")
        f.write(f"gregion[2].g_en = {c.border_en}\n\n")

        # Ionic model regions
        f.write(f"num_imp_regions = 3\n\n")

        f.write(f"# Healthy - normal tenTusscherPanfilov\n")
        f.write(f"imp_region[0].num_IDs = 1\n")
        f.write(f"imp_region[0].ID[0] = 1\n")
        f.write(f"imp_region[0].im = tenTusscherPanfilov\n")
        f.write(f"imp_region[0].cellSurfVolRatio = 0.14\n\n")

        f.write(f"# Scar - 5% ionic currents (functionally inexcitable)\n")
        f.write(f"imp_region[1].num_IDs = 1\n")
        f.write(f"imp_region[1].ID[0] = 2\n")
        f.write(f"imp_region[1].im = tenTusscherPanfilov\n")
        f.write(f'imp_region[1].im_param = "GNa*0.05,GK1*0.05,GCaL*0.05,Gto*0.05"\n')
        f.write(f"imp_region[1].cellSurfVolRatio = 0.14\n\n")

        f.write(f"# Border zone - remodeled (60-70% currents)\n")
        f.write(f"imp_region[2].num_IDs = 1\n")
        f.write(f"imp_region[2].ID[0] = 3\n")
        f.write(f"imp_region[2].im = tenTusscherPanfilov\n")
        f.write(f'imp_region[2].im_param = "GNa*0.6,GK1*0.7,GCaL*0.7,Gto*0.3"\n')
        f.write(f"imp_region[2].cellSurfVolRatio = 0.14\n\n")

        # Stimulus
        f.write(f"# Stimulus at apex\n")
        f.write(f"num_stim = 1\n")
        f.write(f"stimulus[0].stimtype = 0\n")
        f.write(f"stimulus[0].strength = 150.0\n")
        f.write(f"stimulus[0].duration = 2.0\n")
        f.write(f"stimulus[0].start = 0\n")
        f.write(f"stimulus[0].npls = 1\n")
        f.write(f"stimulus[0].vtx_file = {apex_vtx}\n\n")

        # LAT measurement
        f.write(f"# LAT detection\n")
        f.write(f"num_LATs = 1\n")
        f.write(f"lats[0].ID = LAT\n")
        f.write(f"lats[0].all = 1\n")
        f.write(f"lats[0].measurand = 0\n")
        f.write(f"lats[0].threshold = -10.0\n")
        f.write(f"lats[0].method = 1\n\n")

        # APD computation
        f.write(f"# APD computation\n")
        f.write(f"compute_APD = 1\n")
        f.write(f"actthresh = -40.0\n")
        f.write(f"recovery_thresh = 0.9\n\n")

        # Output settings
        f.write(f"# Output intervals (in ms)\n")
        f.write(f"spacedt = 1.0\n")
        f.write(f"timedt = 10.0\n")

    return par_path


def run_opencarp_mpi(par_file: Path, work_dir: Path, n_procs: int = 4,
                     timeout_seconds: int = 1800) -> bool:
    """
    Run OpenCARP with MPI parallelization
    """
    cmd = [OPENCARP_BIN, "+F", str(par_file)]

    if n_procs > 1:
        cmd = ["mpirun", "-np", str(n_procs)] + cmd

    logger.info(f"Running with {n_procs} MPI processes...")

    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(work_dir),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        output = []
        last_t = -1

        while True:
            elapsed = time.time() - start
            if elapsed > timeout_seconds:
                logger.error(f"TIMEOUT after {elapsed:.0f}s")
                proc.kill()
                proc.wait()
                return False

            try:
                line = proc.stdout.readline()
            except:
                break

            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
                continue

            output.append(line)

            # Parse progress from IO_stats output
            if "time\t" in line.lower() or "|" in line:
                try:
                    parts = line.split()
                    if len(parts) > 0:
                        t = float(parts[0])
                        if t > last_t + 50:
                            logger.info(f"  [{int(elapsed)}s] t = {t:.0f} ms")
                            last_t = t
                except:
                    pass
            elif "Error" in line or "diverged" in line:
                logger.error(f"  {line.rstrip()}")

        proc.wait()
        elapsed = time.time() - start

        if proc.returncode != 0:
            logger.error(f"OpenCARP failed (code {proc.returncode})")
            for line in output[-20:]:
                logger.error(f"  {line.rstrip()}")
            return False

        logger.info(f"Completed in {elapsed:.1f}s")
        return True

    except Exception as e:
        logger.error(f"Exception: {e}")
        return False


def compute_cv_correct(lat: np.ndarray, nodes: np.ndarray,
                       elements: np.ndarray, tags: np.ndarray) -> Dict:
    """
    Compute conduction velocity with CORRECT unit handling.

    Mesh coordinates: micrometers (μm)
    LAT values: milliseconds (ms)

    CV = distance / time
    Convert μm to mm for CV in m/s: (μm / 1000) / ms = mm/ms = m/s
    """
    cv_by_tag = {1: [], 2: [], 3: []}

    for i, (elem, tag) in enumerate(zip(elements, tags)):
        # Skip if any node has invalid LAT
        if any(n >= len(lat) or lat[n] < 0 for n in elem):
            continue

        try:
            # Node coordinates: convert from μm to mm
            pts = nodes[elem] / 1000.0  # μm -> mm
            lat_vals = lat[elem]

            # Skip if LAT values are too similar (no gradient)
            lat_range = lat_vals.max() - lat_vals.min()
            if lat_range < 0.01:  # Less than 0.01 ms
                continue

            # Build gradient matrix for tetrahedron
            # Gradient of LAT: ∇LAT (units: ms/mm)
            A = np.array([
                pts[1] - pts[0],
                pts[2] - pts[0],
                pts[3] - pts[0]
            ])

            b = np.array([
                lat_vals[1] - lat_vals[0],
                lat_vals[2] - lat_vals[0],
                lat_vals[3] - lat_vals[0]
            ])

            # Check matrix conditioning
            if np.linalg.cond(A) > 1e10:
                continue

            # Solve for gradient (ms/mm)
            grad_lat = np.linalg.solve(A, b)
            grad_mag = np.linalg.norm(grad_lat)  # ms/mm

            if grad_mag > 0.1:  # At least 0.1 ms/mm (CV < 10 m/s)
                cv = 1.0 / grad_mag  # mm/ms = m/s
                cv = min(max(cv, 0.01), 5.0)  # Clamp to 0.01-5 m/s

                if tag in cv_by_tag:
                    cv_by_tag[tag].append(cv)
        except:
            continue

    result = {}
    for tag, name in [(1, "healthy"), (2, "scar"), (3, "border")]:
        vals = cv_by_tag.get(tag, [])
        if vals:
            vals = np.array(vals)
            result[f"CV_{name}_mean_m_s"] = float(np.mean(vals))
            result[f"CV_{name}_std_m_s"] = float(np.std(vals))
            result[f"CV_{name}_median_m_s"] = float(np.median(vals))
            result[f"CV_{name}_p10_m_s"] = float(np.percentile(vals, 10))
            result[f"CV_{name}_p90_m_s"] = float(np.percentile(vals, 90))
            result[f"CV_{name}_n_elements"] = len(vals)

    return result


def extract_all_metrics(sim_output_dir: Path, mesh_dir: Path,
                        patient_id: str) -> Dict:
    """
    Extract ALL electrophysiology metrics from simulation output
    """
    metrics = {"patient_id": patient_id}

    # Find output directory
    sim_out = sim_output_dir / f"{patient_id}_v7"
    if not sim_out.exists():
        sim_out = sim_output_dir

    # ==================== LOAD MESH ====================
    pts_file = mesh_dir / f"{patient_id}.pts"
    elem_file = mesh_dir / f"{patient_id}.elem"

    # Load nodes (in micrometers)
    nodes = None
    try:
        with open(pts_file, 'r') as f:
            lines = f.readlines()
        start = 1 if len(lines[0].strip().split()) == 1 else 0
        node_list = []
        for line in lines[start:]:
            parts = line.strip().split()
            if len(parts) >= 3:
                node_list.append([float(p) for p in parts[:3]])
        nodes = np.array(node_list)
        metrics["n_nodes"] = len(nodes)
    except Exception as e:
        logger.error(f"Error loading nodes: {e}")

    # Load elements with tags
    elements = None
    tags = None
    try:
        with open(elem_file, 'r') as f:
            lines = f.readlines()
        start = 1 if len(lines[0].strip().split()) == 1 else 0
        elem_list = []
        tag_list = []
        for line in lines[start:]:
            parts = line.strip().split()
            if parts[0] == "Tt" and len(parts) >= 6:
                elem_list.append([int(parts[i]) for i in range(1, 5)])
                tag_list.append(int(parts[-1]))
        elements = np.array(elem_list)
        tags = np.array(tag_list)
        metrics["n_elements"] = len(elements)
    except Exception as e:
        logger.error(f"Error loading elements: {e}")

    # ==================== LOAD LAT ====================
    lat = None
    lat_file = None

    for pattern in ["LAT-thresh.dat", "*LAT*.dat", "LAT.dat"]:
        for f in sim_out.rglob(pattern):
            try:
                lat_data = np.loadtxt(f)
                if lat_data.ndim == 2 and lat_data.shape[1] >= 2:
                    # Create full LAT array indexed by node
                    if nodes is not None:
                        lat = np.full(len(nodes), -1.0)
                        for row in lat_data:
                            idx = int(row[0])
                            if idx < len(lat):
                                lat[idx] = row[1]
                    else:
                        lat = lat_data[:, 1]
                else:
                    lat = lat_data
                lat_file = f
                logger.info(f"Loaded LAT from {f}")
                break
            except Exception as e:
                logger.warning(f"Could not load {f}: {e}")
        if lat is not None:
            break

    if lat is None:
        metrics["error"] = "no_lat_data"
        return metrics

    # Valid LAT (non-negative)
    valid_mask = lat >= 0
    valid_lat = lat[valid_mask]

    if len(valid_lat) == 0:
        metrics["error"] = "no_valid_lat"
        return metrics

    # ==================== BASIC LAT METRICS ====================
    metrics["total_activation_time_ms"] = float(valid_lat.max() - valid_lat.min())
    metrics["QRS_duration_ms"] = float(valid_lat.max())
    metrics["mean_activation_ms"] = float(np.mean(valid_lat))
    metrics["median_activation_ms"] = float(np.median(valid_lat))
    metrics["activation_std_ms"] = float(np.std(valid_lat))
    metrics["n_activated_nodes"] = int(np.sum(valid_mask))
    metrics["n_total_nodes"] = len(lat)
    metrics["activation_ratio"] = float(np.sum(valid_mask) / len(lat))

    # ==================== PER-REGION LAT METRICS ====================
    if elements is not None and tags is not None:
        lat_by_tag = {1: [], 2: [], 3: []}

        for elem, tag in zip(elements, tags):
            if any(n >= len(lat) for n in elem):
                continue
            elem_lats = lat[elem]
            valid_elem = elem_lats[elem_lats >= 0]
            if len(valid_elem) > 0:
                mean_lat = np.mean(valid_elem)
                if tag in lat_by_tag:
                    lat_by_tag[tag].append(mean_lat)

        for tag, name in [(1, "healthy"), (2, "scar"), (3, "border")]:
            vals = lat_by_tag.get(tag, [])
            if vals:
                vals = np.array(vals)
                metrics[f"activation_{name}_mean_ms"] = float(np.mean(vals))
                metrics[f"activation_{name}_std_ms"] = float(np.std(vals))
                metrics[f"activation_{name}_min_ms"] = float(np.min(vals))
                metrics[f"activation_{name}_max_ms"] = float(np.max(vals))
                metrics[f"n_elements_{name}"] = len(vals)

        # Activation delays
        if lat_by_tag[3] and lat_by_tag[1]:
            metrics["activation_delay_border_vs_healthy_ms"] = float(
                np.mean(lat_by_tag[3]) - np.mean(lat_by_tag[1]))

        if lat_by_tag[2] and lat_by_tag[1]:
            metrics["activation_delay_scar_vs_healthy_ms"] = float(
                np.mean(lat_by_tag[2]) - np.mean(lat_by_tag[1]))

    # ==================== CONDUCTION VELOCITY ====================
    if nodes is not None and elements is not None and tags is not None:
        cv_metrics = compute_cv_correct(lat, nodes, elements, tags)
        metrics.update(cv_metrics)

    # ==================== MESH GEOMETRY ====================
    if nodes is not None:
        extent = nodes.max(axis=0) - nodes.min(axis=0)
        metrics["mesh_extent_x_um"] = float(extent[0])
        metrics["mesh_extent_y_um"] = float(extent[1])
        metrics["mesh_extent_z_um"] = float(extent[2])
        metrics["mesh_diagonal_mm"] = float(np.linalg.norm(extent) / 1000)

    return metrics


def process_patient(patient_id: str, output_base: Path, tend: float = 400.0,
                    n_procs: int = 8, timeout: int = 1800) -> Dict:
    """
    Process one patient - full pipeline
    """
    logger.info(f"Processing patient: {patient_id}")

    # Source files
    pts_src = BASE_DIR / "simulation_ready" / patient_id / f"{patient_id}_tet.pts"
    elem_src = BASE_DIR / "infarct_results_comprehensive" / patient_id / f"{patient_id}_tagged.elem"
    lon_src = BASE_DIR / "laplace_complete_v2" / patient_id / f"{patient_id}.lon"

    # Check files exist
    if not pts_src.exists():
        return {"error": f"Missing pts file: {pts_src}"}
    if not elem_src.exists():
        return {"error": f"Missing elem file: {elem_src}"}

    # Output directories
    mesh_dir = output_base / patient_id / "mesh"
    opencarp_dir = output_base / patient_id / "opencarp"

    # Clean and create directories
    if opencarp_dir.exists():
        shutil.rmtree(opencarp_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    opencarp_dir.mkdir(parents=True, exist_ok=True)

    # Check mesh units and copy/convert
    work_pts = mesh_dir / f"{patient_id}.pts"
    work_elem = mesh_dir / f"{patient_id}.elem"

    unit, max_range = check_mesh_units(pts_src)
    logger.info(f"Mesh units: {unit} (max range: {max_range:.1f})")

    if unit == "mm":
        logger.info("Converting mesh from mm to micrometers...")
        convert_pts_to_um(pts_src, work_pts)
    else:
        shutil.copy2(pts_src, work_pts)

    shutil.copy2(elem_src, work_elem)

    if lon_src.exists():
        shutil.copy2(lon_src, mesh_dir / f"{patient_id}.lon")

    # Get mesh info
    info = get_mesh_info(work_pts, work_elem)
    logger.info(f"Mesh: {info['n_nodes']} nodes, {info['n_elements']} elements")
    logger.info(f"Tags: {info['tags']}")

    # Find apex and create stimulus file
    apex = find_apex_nodes(work_pts)
    logger.info(f"Found {len(apex)} apex nodes for stimulus")
    vtx_file = opencarp_dir / "stim_apex.vtx"
    write_vtx_file(apex, vtx_file)

    # Create parameter file
    mesh_path = str(mesh_dir / patient_id)
    par_file = opencarp_dir / "simulation.par"
    write_parameter_file(par_file, mesh_path, patient_id, vtx_file, tend)

    # Determine optimal MPI processes
    # Use all requested CPUs (no artificial cap)
    optimal_procs = min(n_procs, max(1, info["n_nodes"] // 1000))
    logger.info(f"Using {optimal_procs} MPI processes")

    # Run simulation
    start_time = time.time()
    success = run_opencarp_mpi(par_file, opencarp_dir, optimal_procs, timeout)
    runtime = time.time() - start_time

    if not success:
        return {
            "error": "simulation_failed",
            "runtime_s": runtime,
            "n_nodes": info["n_nodes"],
            "n_elements": info["n_elements"]
        }

    # Extract metrics
    metrics = extract_all_metrics(opencarp_dir, mesh_dir, patient_id)
    metrics["runtime_s"] = runtime
    metrics["mpi_processes"] = optimal_procs
    metrics.update({f"n_tag_{k}": v for k, v in info["tags"].items()})

    # Save metrics
    results_file = opencarp_dir / f"{patient_id}_metrics.json"
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {results_file}")

    return metrics


def run_all_patients(output_base: Path, tend: float = 400.0,
                     n_procs: int = None) -> Dict:
    """Run simulation for all 10 patients"""

    patients = [
        "SCD0000101", "SCD0000201", "SCD0000301", "SCD0000401", "SCD0000601",
        "SCD0000701", "SCD0000801", "SCD0001001", "SCD0001101", "SCD0001201"
    ]

    if n_procs is None:
        n_procs = get_num_cpus()  # Use ALL available CPUs

    logger.info(f"OpenCARP v7 Simulation Pipeline - VERIFIED")
    logger.info(f"Output: {output_base}")
    logger.info(f"Duration: {tend} ms")
    logger.info(f"MPI processes: {n_procs}")
    logger.info(f"Patients: {len(patients)}")

    results = {}
    total_start = time.time()

    for i, pid in enumerate(patients):
        logger.info(f"\n[{i+1}/{len(patients)}] Processing {pid}...")
        try:
            results[pid] = process_patient(pid, output_base, tend, n_procs)
        except Exception as e:
            logger.error(f"Error processing {pid}: {e}")
            results[pid] = {"error": str(e)}

    total_time = time.time() - total_start

    # Summary
    logger.info("SUMMARY")
    logger.info(f"Total time: {total_time/60:.1f} minutes")

    successful = [p for p, r in results.items() if "error" not in r]
    failed = [p for p, r in results.items() if "error" in r]

    logger.info(f"Successful: {len(successful)}/{len(patients)}")
    if failed:
        logger.info(f"Failed: {failed}")

    # Print key metrics
    logger.info("\nKey Metrics:")
    for pid in successful:
        r = results[pid]
        qrs = r.get("QRS_duration_ms", "N/A")
        cv_h = r.get("CV_healthy_mean_m_s", "N/A")
        cv_s = r.get("CV_scar_mean_m_s", "N/A")
        cv_b = r.get("CV_border_mean_m_s", "N/A")
        if isinstance(qrs, float):
            logger.info(f"  {pid}: QRS={qrs:.1f}ms, CV_healthy={cv_h:.3f}m/s, CV_scar={cv_s:.3f}m/s, CV_border={cv_b:.3f}m/s")

    # Save all results
    summary_file = output_base / "all_results.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved all results to {summary_file}")

    return results


# Convenience functions for Jupyter notebooks
def run_single_patient(patient_id: str = "SCD0000101", tend: float = 400.0,
                       n_procs: int = 16, output_dir: str = None) -> Dict:
    """
    Run simulation for a single patient (Jupyter-friendly).

    Example:
        from opencarp_simulation_v7_verified import run_single_patient
        result = run_single_patient("SCD0000101", tend=400, n_procs=16)
    """
    if output_dir is None:
        output_dir = str(Path(os.environ.get('SCD_MODELS_DIR', 'SCD_MODELS')) / "opencarp_results" / "v7")

    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    return process_patient(patient_id, output_base, tend, n_procs)


def run_all(tend: float = 400.0, n_procs: int = 16, output_dir: str = None) -> Dict:
    """
    Run all 10 patients (Jupyter-friendly).

    Example:
        from opencarp_simulation_v7_verified import run_all
        results = run_all(tend=400, n_procs=16)
    """
    if output_dir is None:
        output_dir = str(Path(os.environ.get('SCD_MODELS_DIR', 'SCD_MODELS')) / "opencarp_results" / "v7_all")

    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    return run_all_patients(output_base, tend, n_procs)


def _is_jupyter():
    """Check if running in Jupyter notebook"""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except:
        pass
    return False


if __name__ == "__main__":
    if _is_jupyter():
        print("Running in Jupyter. Use:")
        print("  from opencarp_simulation_v7_verified import run_single_patient, run_all")
        print('  result = run_single_patient("SCD0000101", tend=400, n_procs=16)')
        print("  results = run_all(tend=400, n_procs=16)")
    else:
        import argparse

        parser = argparse.ArgumentParser(description="OpenCARP v7 - Verified Pipeline")
        parser.add_argument("--patient", type=str, help="Single patient ID")
        parser.add_argument("--all", action="store_true", help="Run all patients")
        parser.add_argument("--output", type=str,
                           default=str(Path(os.environ.get('SCD_MODELS_DIR', 'SCD_MODELS')) / "opencarp_results" / "v7"),
                           help="Output directory")
        parser.add_argument("--tend", type=float, default=400.0, help="Simulation duration (ms)")
        parser.add_argument("--nprocs", type=int, default=16, help="MPI processes")

        args = parser.parse_args()

        output_base = Path(args.output)
        output_base.mkdir(parents=True, exist_ok=True)

        if args.all:
            results = run_all_patients(output_base, args.tend, args.nprocs)
        elif args.patient:
            result = process_patient(args.patient, output_base, args.tend, args.nprocs)
            print(json.dumps(result, indent=2))
        else:
            print("OpenCARP Simulation v7 - VERIFIED PIPELINE")
            print("VERIFIED: dt is in MICROSECONDS (10 μs = 0.01 ms)")
            print("FIXED: CV calculation with correct unit handling")
            print()

            result = process_patient("SCD0000101", output_base, args.tend, args.nprocs)
            print("\nResults:")
            print(json.dumps(result, indent=2))
