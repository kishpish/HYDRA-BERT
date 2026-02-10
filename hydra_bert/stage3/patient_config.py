#!/usr/bin/env python3
"""
Patient Configuration Module

Contains patient-specific data for all 10 real cardiac patients including:
- Baseline cardiac function (LVEF, GLS, EDV, ESV)
- Infarct characteristics (scar fraction, border zone fraction)
- Tissue mechanics (stress, transmurality, wall thickness)
- Injection site coordinates (based on infarct location)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class InjectionSite:
    """Represents a single injection site in the myocardium."""
    x: float  # mm from apex
    y: float  # mm circumferential
    z: float  # mm transmural depth
    volume_ml: float = 0.5  # injection volume
    region: str = "border_zone"  # scar, border_zone, or remote


@dataclass
class InfarctGeometry:
    """Defines the infarct and border zone geometry."""
    # Infarct location (AHA 17-segment model)
    segments: List[int] = field(default_factory=lambda: [7, 8, 13])  # Default: mid-anterior
    # Geometric parameters
    center_x: float = 40.0  # mm from base
    center_y: float = 0.0   # degrees circumferential
    extent_longitudinal: float = 30.0  # mm
    extent_circumferential: float = 90.0  # degrees
    transmural_extent: float = 0.5  # fraction (0-1)


@dataclass
class PatientConfig:
    """Complete patient configuration for design optimization."""

    # Patient identification
    patient_id: str
    is_real_patient: bool = True

    # Baseline cardiac function
    baseline_LVEF_pct: float = 35.0
    baseline_GLS_pct: float = -17.0  # Negative value
    baseline_EDV_mL: float = 200.0
    baseline_ESV_mL: float = 130.0

    # Infarct characteristics
    scar_fraction_pct: float = 8.0
    bz_fraction_pct: float = 20.0

    # Tissue mechanics
    bz_stress_kPa: float = 18.0
    healthy_stress_kPa: float = 8.0
    stress_concentration: float = 2.25
    transmurality: float = 0.46
    wall_thickness_mm: float = 20.0

    # Infarct geometry
    infarct_geometry: InfarctGeometry = field(default_factory=InfarctGeometry)

    # Injection sites (calculated based on infarct location)
    injection_sites: List[InjectionSite] = field(default_factory=list)

    # Patient-specific constraints
    max_injection_volume_ml: float = 5.0
    min_stiffness_kPa: float = 1.0
    max_stiffness_kPa: float = 50.0

    def __post_init__(self):
        """Calculate injection sites if not provided."""
        if not self.injection_sites:
            self.injection_sites = self._calculate_injection_sites()

    def _calculate_injection_sites(self) -> List[InjectionSite]:
        """Calculate optimal injection sites based on infarct geometry."""
        sites = []
        infarct = self.infarct_geometry

        # Create grid of injection sites around border zone
        n_longitudinal = 3
        n_circumferential = 4

        for i in range(n_longitudinal):
            for j in range(n_circumferential):
                # Longitudinal position (along infarct + BZ)
                x = infarct.center_x - infarct.extent_longitudinal/2 + \
                    (i + 0.5) * infarct.extent_longitudinal / n_longitudinal

                # Circumferential position
                y = infarct.center_y - infarct.extent_circumferential/2 + \
                    (j + 0.5) * infarct.extent_circumferential / n_circumferential

                # Transmural depth (mid-wall for better retention)
                z = self.wall_thickness_mm * 0.5

                # Determine region based on position relative to scar core
                dist_from_center = np.sqrt((i - n_longitudinal/2)**2 + (j - n_circumferential/2)**2)
                if dist_from_center < 0.5:
                    region = "scar"
                else:
                    region = "border_zone"

                sites.append(InjectionSite(
                    x=x, y=y, z=z,
                    volume_ml=0.4,
                    region=region
                ))

        return sites

    def get_patient_features(self) -> np.ndarray:
        """Get feature vector for model input."""
        return np.array([
            self.baseline_LVEF_pct / 100.0,
            self.baseline_GLS_pct / 100.0,
            self.baseline_EDV_mL / 500.0,  # Normalize
            self.baseline_ESV_mL / 500.0,
            self.scar_fraction_pct / 100.0,
            self.bz_fraction_pct / 100.0,
            self.bz_stress_kPa / 50.0,
            self.healthy_stress_kPa / 50.0,
            self.stress_concentration / 5.0,
            self.transmurality,
        ], dtype=np.float32)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "patient_id": self.patient_id,
            "is_real_patient": self.is_real_patient,
            "baseline_LVEF_pct": self.baseline_LVEF_pct,
            "baseline_GLS_pct": self.baseline_GLS_pct,
            "baseline_EDV_mL": self.baseline_EDV_mL,
            "baseline_ESV_mL": self.baseline_ESV_mL,
            "scar_fraction_pct": self.scar_fraction_pct,
            "bz_fraction_pct": self.bz_fraction_pct,
            "bz_stress_kPa": self.bz_stress_kPa,
            "healthy_stress_kPa": self.healthy_stress_kPa,
            "stress_concentration": self.stress_concentration,
            "transmurality": self.transmurality,
            "wall_thickness_mm": self.wall_thickness_mm,
            "num_injection_sites": len(self.injection_sites),
            "infarct_segments": self.infarct_geometry.segments,
        }


# Real patient configurations based on clinical data
REAL_PATIENTS: Dict[str, PatientConfig] = {
    "SCD0000101": PatientConfig(
        patient_id="SCD0000101",
        baseline_LVEF_pct=34.75,
        baseline_GLS_pct=-16.81,
        baseline_EDV_mL=112.66,
        baseline_ESV_mL=73.51,
        scar_fraction_pct=7.53,
        bz_fraction_pct=23.91,
        bz_stress_kPa=18.0,
        healthy_stress_kPa=8.0,
        stress_concentration=2.25,
        transmurality=0.463,
        wall_thickness_mm=17.42,
        infarct_geometry=InfarctGeometry(
            segments=[7, 8, 13],  # Mid-anterior
            center_x=45.0,
            center_y=0.0,
            extent_longitudinal=25.0,
            extent_circumferential=80.0,
            transmural_extent=0.463
        )
    ),

    "SCD0000201": PatientConfig(
        patient_id="SCD0000201",
        baseline_LVEF_pct=36.70,
        baseline_GLS_pct=-17.09,
        baseline_EDV_mL=100.84,
        baseline_ESV_mL=63.83,
        scar_fraction_pct=7.15,
        bz_fraction_pct=21.15,
        bz_stress_kPa=17.5,
        healthy_stress_kPa=8.0,
        stress_concentration=2.19,
        transmurality=0.454,
        wall_thickness_mm=15.01,
        infarct_geometry=InfarctGeometry(
            segments=[7, 8, 12, 13],  # Anteroseptal
            center_x=42.0,
            center_y=-30.0,
            extent_longitudinal=22.0,
            extent_circumferential=75.0,
            transmural_extent=0.454
        )
    ),

    "SCD0000301": PatientConfig(
        patient_id="SCD0000301",
        baseline_LVEF_pct=33.96,
        baseline_GLS_pct=-16.79,
        baseline_EDV_mL=563.66,
        baseline_ESV_mL=372.26,
        scar_fraction_pct=9.03,
        bz_fraction_pct=21.00,
        bz_stress_kPa=20.0,
        healthy_stress_kPa=8.0,
        stress_concentration=2.50,
        transmurality=0.501,
        wall_thickness_mm=24.03,
        infarct_geometry=InfarctGeometry(
            segments=[2, 3, 8, 9, 14],  # Large anterolateral
            center_x=50.0,
            center_y=45.0,
            extent_longitudinal=40.0,
            extent_circumferential=100.0,
            transmural_extent=0.501
        )
    ),

    "SCD0000401": PatientConfig(
        patient_id="SCD0000401",
        baseline_LVEF_pct=35.35,
        baseline_GLS_pct=-16.94,
        baseline_EDV_mL=363.63,
        baseline_ESV_mL=235.08,
        scar_fraction_pct=8.01,
        bz_fraction_pct=21.26,
        bz_stress_kPa=18.5,
        healthy_stress_kPa=8.0,
        stress_concentration=2.31,
        transmurality=0.475,
        wall_thickness_mm=21.39,
        infarct_geometry=InfarctGeometry(
            segments=[3, 4, 9, 10, 15],  # Inferolateral
            center_x=48.0,
            center_y=120.0,
            extent_longitudinal=35.0,
            extent_circumferential=90.0,
            transmural_extent=0.475
        )
    ),

    "SCD0000601": PatientConfig(
        patient_id="SCD0000601",
        baseline_LVEF_pct=35.15,
        baseline_GLS_pct=-16.90,
        baseline_EDV_mL=394.40,
        baseline_ESV_mL=255.77,
        scar_fraction_pct=7.90,
        bz_fraction_pct=21.99,
        bz_stress_kPa=18.3,
        healthy_stress_kPa=8.0,
        stress_concentration=2.29,
        transmurality=0.473,
        wall_thickness_mm=23.64,
        infarct_geometry=InfarctGeometry(
            segments=[4, 5, 10, 11],  # Inferior
            center_x=52.0,
            center_y=180.0,
            extent_longitudinal=32.0,
            extent_circumferential=85.0,
            transmural_extent=0.473
        )
    ),

    "SCD0000701": PatientConfig(
        patient_id="SCD0000701",
        baseline_LVEF_pct=36.68,
        baseline_GLS_pct=-17.14,
        baseline_EDV_mL=247.18,
        baseline_ESV_mL=156.51,
        scar_fraction_pct=7.90,
        bz_fraction_pct=18.93,
        bz_stress_kPa=17.8,
        healthy_stress_kPa=8.0,
        stress_concentration=2.23,
        transmurality=0.473,
        wall_thickness_mm=21.05,
        infarct_geometry=InfarctGeometry(
            segments=[6, 11, 12],  # Inferoseptal
            center_x=46.0,
            center_y=-120.0,
            extent_longitudinal=28.0,
            extent_circumferential=70.0,
            transmural_extent=0.473
        )
    ),

    "SCD0000801": PatientConfig(
        patient_id="SCD0000801",
        baseline_LVEF_pct=37.14,
        baseline_GLS_pct=-17.20,
        baseline_EDV_mL=216.50,
        baseline_ESV_mL=136.10,
        scar_fraction_pct=7.67,
        bz_fraction_pct=18.72,
        bz_stress_kPa=17.5,
        healthy_stress_kPa=8.0,
        stress_concentration=2.19,
        transmurality=0.467,
        wall_thickness_mm=21.80,
        infarct_geometry=InfarctGeometry(
            segments=[1, 7, 13],  # Anterior apex
            center_x=55.0,
            center_y=0.0,
            extent_longitudinal=30.0,
            extent_circumferential=75.0,
            transmural_extent=0.467
        )
    ),

    "SCD0001001": PatientConfig(
        patient_id="SCD0001001",
        baseline_LVEF_pct=39.89,
        baseline_GLS_pct=-17.59,
        baseline_EDV_mL=232.00,
        baseline_ESV_mL=139.46,
        scar_fraction_pct=7.00,
        bz_fraction_pct=15.23,
        bz_stress_kPa=16.5,
        healthy_stress_kPa=8.0,
        stress_concentration=2.06,
        transmurality=0.450,
        wall_thickness_mm=19.81,
        infarct_geometry=InfarctGeometry(
            segments=[7, 8],  # Small mid-anterior
            center_x=44.0,
            center_y=15.0,
            extent_longitudinal=20.0,
            extent_circumferential=60.0,
            transmural_extent=0.450
        )
    ),

    "SCD0001101": PatientConfig(
        patient_id="SCD0001101",
        baseline_LVEF_pct=37.57,
        baseline_GLS_pct=-17.24,
        baseline_EDV_mL=249.42,
        baseline_ESV_mL=155.71,
        scar_fraction_pct=7.25,
        bz_fraction_pct=19.09,
        bz_stress_kPa=17.6,
        healthy_stress_kPa=8.0,
        stress_concentration=2.20,
        transmurality=0.456,
        wall_thickness_mm=21.00,
        infarct_geometry=InfarctGeometry(
            segments=[2, 3, 8, 9],  # Anterolateral
            center_x=47.0,
            center_y=60.0,
            extent_longitudinal=28.0,
            extent_circumferential=80.0,
            transmural_extent=0.456
        )
    ),

    "SCD0001201": PatientConfig(
        patient_id="SCD0001201",
        baseline_LVEF_pct=37.06,
        baseline_GLS_pct=-17.17,
        baseline_EDV_mL=75.34,
        baseline_ESV_mL=47.43,
        scar_fraction_pct=7.44,
        bz_fraction_pct=19.55,
        bz_stress_kPa=17.7,
        healthy_stress_kPa=8.0,
        stress_concentration=2.21,
        transmurality=0.461,
        wall_thickness_mm=19.12,
        infarct_geometry=InfarctGeometry(
            segments=[13, 14],  # Apical anterior
            center_x=58.0,
            center_y=30.0,
            extent_longitudinal=18.0,
            extent_circumferential=55.0,
            transmural_extent=0.461
        )
    ),
}


def load_patient_configs() -> Dict[str, PatientConfig]:
    """Load all real patient configurations."""
    return REAL_PATIENTS.copy()


def get_patient_by_id(patient_id: str) -> Optional[PatientConfig]:
    """Get patient configuration by ID."""
    return REAL_PATIENTS.get(patient_id)


def get_patient_ids() -> List[str]:
    """Get list of all patient IDs."""
    return list(REAL_PATIENTS.keys())


if __name__ == "__main__":
    # Print summary of all patients
    print("HYDRA-BERT Real Patient Configurations")

    for pid, config in REAL_PATIENTS.items():
        print(f"\n{pid}:")
        print(f"  LVEF: {config.baseline_LVEF_pct:.1f}% | GLS: {config.baseline_GLS_pct:.2f}%")
        print(f"  EDV: {config.baseline_EDV_mL:.1f}mL | ESV: {config.baseline_ESV_mL:.1f}mL")
        print(f"  Scar: {config.scar_fraction_pct:.1f}% | BZ: {config.bz_fraction_pct:.1f}%")
        print(f"  Transmurality: {config.transmurality:.3f}")
        print(f"  Wall: {config.wall_thickness_mm:.1f}mm")
        print(f"  Injection sites: {len(config.injection_sites)}")
        print(f"  Infarct segments: {config.infarct_geometry.segments}")
