#!/usr/bin/env python3
"""
HYDRA-BERT: OpenCarp Metrics Extraction Script

Extracts cardiac electrophysiology metrics from OpenCarp simulation outputs.

Metrics extracted:
- Activation time maps
- Action potential duration (APD)
- Conduction velocity (CV)
- Arrhythmia vulnerability (from S1S2 protocol)
- Calcium transient characteristics

Usage:
    python extract_opencarp_metrics.py --patient SCD0000101
    python extract_opencarp_metrics.py --all

Author: HYDRA-BERT Pipeline
Date: 2026-02-09
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OPENCARP_RESULTS = Path(os.environ.get('SCD_MODELS_DIR', 'SCD_MODELS')) / "opencarp_results"
OUTPUT_DIR = PROJECT_ROOT / "results" / "extracted_metrics"


def read_igb_file(filepath: Path) -> Optional[np.ndarray]:
    """Read OpenCarp IGB (binary) format file."""
    if not filepath.exists():
        return None

    try:
        with open(filepath, 'rb') as f:
            # Read header
            header_line = f.readline().decode('utf-8').strip()

            # Parse dimensions from header
            dims = {}
            for item in header_line.split():
                if '=' in item:
                    key, val = item.split('=')
                    dims[key] = int(val)

            # Read binary data
            data = np.frombuffer(f.read(), dtype=np.float32)

            if 'x' in dims and 'y' in dims:
                data = data.reshape((dims.get('t', 1), dims['x'], dims.get('y', 1)))

            return data
    except Exception as e:
        logger.warning(f'Error reading {filepath}: {e}')
        return None


def extract_activation_metrics(patient_dir: Path) -> Dict:
    """Extract activation time metrics."""
    metrics = {}

    # Look for activation time files
    act_files = list(patient_dir.glob('**/activation*.dat')) + \
                list(patient_dir.glob('**/LAT*.dat'))

    if act_files:
        metrics['n_activation_files'] = len(act_files)

        # Try to read first file
        try:
            data = np.loadtxt(act_files[0])
            metrics['mean_activation_ms'] = float(np.mean(data))
            metrics['max_activation_ms'] = float(np.max(data))
            metrics['activation_dispersion_ms'] = float(np.std(data))
        except:
            pass

    return metrics


def extract_apd_metrics(patient_dir: Path) -> Dict:
    """Extract action potential duration metrics."""
    metrics = {}

    # Look for APD files
    apd_files = list(patient_dir.glob('**/APD*.dat')) + \
                list(patient_dir.glob('**/apd*.dat'))

    if apd_files:
        metrics['n_apd_files'] = len(apd_files)

        try:
            data = np.loadtxt(apd_files[0])
            metrics['mean_apd_ms'] = float(np.mean(data))
            metrics['apd_dispersion_ms'] = float(np.std(data))
            metrics['apd_range_ms'] = float(np.ptp(data))
        except:
            pass

    return metrics


def extract_cv_metrics(patient_dir: Path) -> Dict:
    """Extract conduction velocity metrics."""
    metrics = {}

    # Look for CV files or compute from activation
    cv_files = list(patient_dir.glob('**/cv*.dat')) + \
               list(patient_dir.glob('**/CV*.dat'))

    if cv_files:
        try:
            data = np.loadtxt(cv_files[0])
            metrics['mean_cv_m_s'] = float(np.mean(data))
            metrics['cv_heterogeneity'] = float(np.std(data) / np.mean(data))
        except:
            pass

    return metrics


def extract_s1s2_metrics(patient_dir: Path) -> Dict:
    """Extract S1S2 protocol metrics (arrhythmia vulnerability)."""
    metrics = {}

    s1s2_dir = patient_dir / 's1s2_sims'
    if s1s2_dir.exists():
        # Look for reentry or vulnerability markers
        result_files = list(s1s2_dir.glob('**/*.dat'))
        metrics['n_s1s2_files'] = len(result_files)

        # Check for reentry
        reentry_files = list(s1s2_dir.glob('**/reentry*'))
        metrics['reentry_detected'] = len(reentry_files) > 0

    return metrics


def extract_calcium_metrics(patient_dir: Path) -> Dict:
    """Extract calcium transient metrics."""
    metrics = {}

    ca_dir = patient_dir / 'calcium_sims'
    if ca_dir.exists():
        ca_files = list(ca_dir.glob('**/*.dat'))
        metrics['n_calcium_files'] = len(ca_files)

    return metrics


def extract_patient_ep_metrics(patient_id: str) -> Dict:
    """Extract all EP metrics for a patient."""
    patient_dir = OPENCARP_RESULTS / patient_id

    result = {
        'patient_id': patient_id,
        'timestamp': datetime.now().isoformat(),
        'source': 'OpenCarp EP simulation'
    }

    if not patient_dir.exists():
        result['error'] = 'Patient directory not found'
        result['path_checked'] = str(patient_dir)
        return result

    # Extract each metric type
    result['activation'] = extract_activation_metrics(patient_dir)
    result['apd'] = extract_apd_metrics(patient_dir)
    result['conduction_velocity'] = extract_cv_metrics(patient_dir)
    result['s1s2'] = extract_s1s2_metrics(patient_dir)
    result['calcium'] = extract_calcium_metrics(patient_dir)

    # Summary
    result['has_activation_data'] = bool(result['activation'])
    result['has_apd_data'] = bool(result['apd'])
    result['has_cv_data'] = bool(result['conduction_velocity'])

    return result


def compute_ep_summary(metrics: Dict) -> Dict:
    """Compute summary EP metrics."""
    summary = {}

    # Activation
    if metrics.get('activation', {}).get('mean_activation_ms'):
        act = metrics['activation']
        summary['total_activation_time_ms'] = act.get('max_activation_ms', 0)
        summary['activation_uniformity'] = 1.0 - (act.get('activation_dispersion_ms', 0) /
                                                   max(act.get('mean_activation_ms', 1), 1))

    # APD
    if metrics.get('apd', {}).get('mean_apd_ms'):
        apd = metrics['apd']
        summary['mean_apd_ms'] = apd.get('mean_apd_ms', 0)
        summary['apd_heterogeneity'] = apd.get('apd_dispersion_ms', 0) / max(apd.get('mean_apd_ms', 1), 1)

    # CV
    if metrics.get('conduction_velocity', {}).get('mean_cv_m_s'):
        cv = metrics['conduction_velocity']
        summary['mean_cv_m_s'] = cv.get('mean_cv_m_s', 0)
        summary['cv_heterogeneity'] = cv.get('cv_heterogeneity', 0)

    # Arrhythmia risk
    summary['arrhythmia_vulnerable'] = metrics.get('s1s2', {}).get('reentry_detected', False)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Extract OpenCarp EP metrics')
    parser.add_argument('--patient', type=str, help='Extract for single patient')
    parser.add_argument('--all', action='store_true', help='Extract for all patients')

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.patient:
        result = extract_patient_ep_metrics(args.patient)
        result['summary'] = compute_ep_summary(result)
        print(json.dumps(result, indent=2, default=str))

        # Save
        output_file = OUTPUT_DIR / f'{args.patient}_opencarp_metrics.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f'Saved: {output_file}')

    elif args.all:
        patients = [
            "SCD0000101", "SCD0000201", "SCD0000301", "SCD0000401",
            "SCD0000601", "SCD0000701", "SCD0000801", "SCD0001001",
            "SCD0001101", "SCD0001201"
        ]

        all_results = []
        for patient_id in patients:
            logger.info(f'Processing {patient_id}')
            result = extract_patient_ep_metrics(patient_id)
            result['summary'] = compute_ep_summary(result)
            all_results.append(result)

            # Save individual
            output_file = OUTPUT_DIR / f'{patient_id}_opencarp_metrics.json'
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)

        # Summary
        summary_file = OUTPUT_DIR / 'opencarp_metrics_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f'Summary saved: {summary_file}')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
