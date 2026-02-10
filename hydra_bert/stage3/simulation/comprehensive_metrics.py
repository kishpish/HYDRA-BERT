#!/usr/bin/env python3
"""
Comprehensive FEBio/OpenCarp Simulation Metrics

Calculates 50+ metrics from cardiac simulations for thorough validation
of hydrogel designs. Organized into categories:

1. Mechanical Metrics (FEBio) - 20 metrics
2. Electrical Metrics (OpenCarp) - 15 metrics
3. Functional Metrics - 10 metrics
4. Integration Metrics - 8 metrics

Total: 53 simulation-based metrics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class FEBioMechanicalMetrics:
    """
    Mechanical metrics from FEBio finite element analysis.
    20 metrics covering stress, strain, and material behavior.
    """

    # STRESS METRICS (8)

    # Wall stress
    peak_wall_stress_kPa: float = 0.0           # Maximum wall stress
    mean_wall_stress_kPa: float = 0.0           # Average wall stress
    wall_stress_reduction_pct: float = 0.0       # Reduction from baseline

    # Regional stress
    scar_stress_kPa: float = 0.0                # Stress in scar region
    border_zone_stress_kPa: float = 0.0          # Stress in border zone
    remote_stress_kPa: float = 0.0               # Stress in remote healthy tissue
    bz_to_remote_stress_ratio: float = 0.0       # Stress concentration indicator

    # Interface stress
    interface_stress_kPa: float = 0.0            # Hydrogel-tissue interface stress

    # STRAIN METRICS (7)

    # Principal strains
    max_principal_strain: float = 0.0            # E1 (circumferential)
    min_principal_strain: float = 0.0            # E3 (radial)
    fiber_strain: float = 0.0                    # Along fiber direction

    # Regional strains
    scar_strain: float = 0.0                     # Strain in scar
    border_zone_strain: float = 0.0              # Strain in BZ
    strain_heterogeneity: float = 0.0            # Variance across regions

    # Strain normalization
    strain_normalization_pct: float = 0.0        # Improvement toward normal

    # DEFORMATION METRICS (5)

    # Wall motion
    wall_thickening_pct: float = 0.0             # Systolic wall thickening
    wall_motion_score: float = 0.0               # Regional wall motion index

    # Geometry
    sphericity_index: float = 0.0                # LV sphericity (remodeling indicator)
    end_diastolic_dimension_mm: float = 0.0      # LV internal diameter
    wall_thickness_mm: float = 0.0               # Average wall thickness

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class OpenCarpElectricalMetrics:
    """
    Electrical metrics from OpenCarp electrophysiology simulation.
    15 metrics covering conduction, activation, and repolarization.
    """

    # CONDUCTION VELOCITY (4)

    # Regional CV
    scar_cv_m_s: float = 0.0                     # CV in scar (typically 0)
    border_zone_cv_m_s: float = 0.0              # CV in border zone
    remote_cv_m_s: float = 0.0                   # CV in remote tissue
    cv_improvement_pct: float = 0.0              # BZ CV improvement

    # ACTIVATION METRICS (5)

    # Activation times
    total_activation_time_ms: float = 0.0        # QRS-like duration
    activation_time_reduction_pct: float = 0.0   # Reduction from baseline
    latest_activation_site: str = ""             # Location of latest activation

    # Activation patterns
    activation_heterogeneity: float = 0.0        # Variance in activation
    conduction_block_present: bool = False       # Presence of block

    # REPOLARIZATION METRICS (4)

    # APD (Action Potential Duration)
    mean_apd_ms: float = 0.0                     # Average APD
    apd_dispersion_ms: float = 0.0               # APD variance (arrhythmia risk)
    apd_dispersion_reduction_pct: float = 0.0    # Improvement

    # Repolarization gradient
    repolarization_gradient: float = 0.0         # Spatial gradient

    # ARRHYTHMIA RISK (2)

    arrhythmia_vulnerability_index: float = 0.0  # Composite risk score
    reentry_circuit_probability: float = 0.0     # Probability of reentry

    def to_dict(self) -> Dict:
        result = {k: v for k, v in self.__dict__.items()}
        # Convert bool to int for compatibility
        result["conduction_block_present"] = int(result["conduction_block_present"])
        return result


@dataclass
class FunctionalMetrics:
    """
    Cardiac functional outcome metrics.
    10 metrics covering pump function and hemodynamics.
    """

    # PUMP FUNCTION (5)

    # Ejection fraction
    baseline_LVEF_pct: float = 0.0
    new_LVEF_pct: float = 0.0
    delta_EF_pct: float = 0.0

    # Stroke volume
    stroke_volume_mL: float = 0.0
    stroke_volume_improvement_pct: float = 0.0

    # VOLUMES (3)

    end_diastolic_volume_mL: float = 0.0
    end_systolic_volume_mL: float = 0.0
    delta_ESV_mL: float = 0.0                    # Negative = improvement

    # GLOBAL STRAIN (2)

    baseline_GLS_pct: float = 0.0
    new_GLS_pct: float = 0.0

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class IntegrationMetrics:
    """
    Hydrogel-tissue integration metrics.
    8 metrics covering material interaction and stability.
    """

    # MECHANICAL INTEGRATION (4)

    # Stiffness matching
    tissue_stiffness_kPa: float = 0.0
    hydrogel_stiffness_kPa: float = 0.0
    stiffness_mismatch_ratio: float = 0.0        # |Egap - Etissue| / Etissue

    # Interface
    interface_shear_stress_kPa: float = 0.0

    # COVERAGE AND RETENTION (4)

    # Coverage
    scar_coverage_pct: float = 0.0
    border_zone_coverage_pct: float = 0.0

    # Retention
    retention_fraction: float = 0.0              # At 30 days
    edge_effect_severity: float = 0.0            # Stress concentration at edges

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


class ComprehensiveSimulationCalculator:
    """
    Calculates all 53 simulation metrics for a hydrogel design.

    Uses FEBio for mechanical analysis and OpenCarp for electrical analysis.
    Falls back to physics-based surrogate models when simulators unavailable.
    """

    def __init__(
        self,
        febio_available: bool = False,
        opencarp_available: bool = False,
        use_high_fidelity: bool = True
    ):
        self.febio_available = febio_available
        self.opencarp_available = opencarp_available
        self.use_high_fidelity = use_high_fidelity

    def calculate_all_metrics(
        self,
        design,
        patient_config,
        febio_result: Optional[Dict] = None,
        opencarp_result: Optional[Dict] = None
    ) -> Dict:
        """
        Calculate all 53 simulation metrics.

        Args:
            design: DesignCandidate object
            patient_config: PatientConfig object
            febio_result: Raw FEBio simulation output (optional)
            opencarp_result: Raw OpenCarp simulation output (optional)

        Returns:
            Dict with all 53 metrics
        """
        # Calculate each category
        mechanical = self.calculate_mechanical_metrics(design, patient_config, febio_result)
        electrical = self.calculate_electrical_metrics(design, patient_config, opencarp_result)
        functional = self.calculate_functional_metrics(design, patient_config, mechanical)
        integration = self.calculate_integration_metrics(design, patient_config, mechanical)

        # Combine all metrics
        all_metrics = {}
        all_metrics.update({"mech_" + k: v for k, v in mechanical.to_dict().items()})
        all_metrics.update({"elec_" + k: v for k, v in electrical.to_dict().items()})
        all_metrics.update({"func_" + k: v for k, v in functional.to_dict().items()})
        all_metrics.update({"integ_" + k: v for k, v in integration.to_dict().items()})

        return all_metrics

    def calculate_mechanical_metrics(
        self,
        design,
        patient_config,
        febio_result: Optional[Dict] = None
    ) -> FEBioMechanicalMetrics:
        """Calculate 20 mechanical metrics from FEBio or surrogate model."""

        metrics = FEBioMechanicalMetrics()

        if febio_result and self.febio_available:
            # Extract from real FEBio results
            metrics = self._extract_febio_metrics(febio_result)
        else:
            # Use physics-based surrogate model
            metrics = self._surrogate_mechanical_metrics(design, patient_config)

        return metrics

    def _surrogate_mechanical_metrics(
        self,
        design,
        patient_config
    ) -> FEBioMechanicalMetrics:
        """Physics-based surrogate for mechanical metrics."""

        metrics = FEBioMechanicalMetrics()

        # Get parameters
        E_gel = design.stiffness_kPa
        E_healthy = 15.0  # kPa, healthy myocardium
        E_scar = 50.0  # kPa, scar tissue
        thickness = design.patch_thickness_mm
        coverage = {"scar_only": 0.5, "scar_bz25": 0.625, "scar_bz50": 0.75, "scar_bz100": 1.0}.get(design.patch_coverage, 0.5)

        # Baseline stresses (Laplace law estimates)
        R = 30  # mm, LV radius
        P = 15  # kPa, LV pressure
        h = patient_config.wall_thickness_mm

        baseline_wall_stress = (P * R) / (2 * h)  # Laplace

        # Stress reduction from hydrogel support
        support_factor = (E_gel / E_healthy) * coverage * (thickness / 3.0)
        stress_reduction = min(0.6, support_factor * 0.4)

        # Wall stress metrics
        metrics.peak_wall_stress_kPa = baseline_wall_stress * (1 - stress_reduction * 0.8)
        metrics.mean_wall_stress_kPa = baseline_wall_stress * (1 - stress_reduction)
        metrics.wall_stress_reduction_pct = stress_reduction * 100

        # Regional stresses
        metrics.scar_stress_kPa = baseline_wall_stress * 1.3 * (1 - stress_reduction * 0.9)
        metrics.border_zone_stress_kPa = baseline_wall_stress * 1.5 * (1 - stress_reduction * 0.7)
        metrics.remote_stress_kPa = baseline_wall_stress * 0.9
        metrics.bz_to_remote_stress_ratio = metrics.border_zone_stress_kPa / max(0.1, metrics.remote_stress_kPa)

        # Interface stress
        stiffness_diff = abs(E_gel - E_healthy) / E_healthy
        metrics.interface_stress_kPa = baseline_wall_stress * 0.3 * (1 + stiffness_diff)

        # Strain metrics
        baseline_strain = 0.15  # ~15% healthy strain
        scar_strain_baseline = 0.02  # Akinetic scar

        # Strain improvement from support
        strain_support = min(0.5, support_factor * 0.3)
        metrics.scar_strain = scar_strain_baseline + baseline_strain * strain_support
        metrics.border_zone_strain = 0.08 + baseline_strain * strain_support * 0.5
        metrics.max_principal_strain = baseline_strain * (1 + strain_support * 0.2)
        metrics.min_principal_strain = -baseline_strain * 0.4 * (1 + strain_support * 0.1)
        metrics.fiber_strain = baseline_strain * 0.9

        # Strain heterogeneity (lower is better)
        metrics.strain_heterogeneity = 0.05 * (1 - strain_support)
        metrics.strain_normalization_pct = strain_support * 100 * (1 + np.random.uniform(-0.1, 0.1))

        # Deformation metrics
        metrics.wall_thickening_pct = 30 + stress_reduction * 20  # Normal ~40%
        metrics.wall_motion_score = max(1.0, 2.5 - stress_reduction * 1.5)  # 1=normal, 4=dyskinetic
        metrics.sphericity_index = 0.6 - stress_reduction * 0.1  # Lower is better
        metrics.end_diastolic_dimension_mm = 55 - stress_reduction * 5
        metrics.wall_thickness_mm = h

        return metrics

    def calculate_electrical_metrics(
        self,
        design,
        patient_config,
        opencarp_result: Optional[Dict] = None
    ) -> OpenCarpElectricalMetrics:
        """Calculate 15 electrical metrics from OpenCarp or surrogate model."""

        metrics = OpenCarpElectricalMetrics()

        if opencarp_result and self.opencarp_available:
            # Extract from real OpenCarp results
            metrics = self._extract_opencarp_metrics(opencarp_result)
        else:
            # Use physics-based surrogate model
            metrics = self._surrogate_electrical_metrics(design, patient_config)

        return metrics

    def _surrogate_electrical_metrics(
        self,
        design,
        patient_config
    ) -> OpenCarpElectricalMetrics:
        """Physics-based surrogate for electrical metrics."""

        metrics = OpenCarpElectricalMetrics()

        # Conductivity parameters
        sigma_gel = design.conductivity_S_m
        sigma_healthy = 0.2  # S/m, healthy myocardium longitudinal
        sigma_scar = 0.01  # S/m, scar (nearly insulating)

        coverage = {"scar_only": 0.5, "scar_bz25": 0.625, "scar_bz50": 0.75, "scar_bz100": 1.0}.get(design.patch_coverage, 0.5)

        # Conduction velocity (cable equation based)
        # CV ∝ sqrt(σ)
        baseline_cv = 0.6  # m/s healthy
        scar_cv = 0.05  # m/s (nearly blocked)
        bz_cv_baseline = 0.3  # m/s (slowed)

        # Improvement from conductive hydrogel
        if sigma_gel > 0.1:
            cv_improvement_factor = min(2.0, 1 + np.log(1 + sigma_gel / sigma_healthy))
        else:
            cv_improvement_factor = 1.0

        metrics.scar_cv_m_s = scar_cv * (1 + sigma_gel * coverage * 5)
        metrics.border_zone_cv_m_s = bz_cv_baseline * cv_improvement_factor
        metrics.remote_cv_m_s = baseline_cv
        metrics.cv_improvement_pct = (cv_improvement_factor - 1) * 100

        # Activation metrics
        baseline_activation_time = 80  # ms
        delayed_activation = 40 * (1 - sigma_gel * coverage)  # Extra delay from scar
        metrics.total_activation_time_ms = baseline_activation_time + delayed_activation
        metrics.activation_time_reduction_pct = (delayed_activation / 40) * 100
        metrics.latest_activation_site = "border_zone" if sigma_gel < 0.2 else "remote"

        # Activation heterogeneity
        metrics.activation_heterogeneity = 20 * (1 - sigma_gel * coverage * 0.5)
        metrics.conduction_block_present = sigma_gel < 0.05 and coverage < 0.5

        # APD metrics
        baseline_apd = 250  # ms
        metrics.mean_apd_ms = baseline_apd
        metrics.apd_dispersion_ms = 30 * (1 - sigma_gel * coverage * 0.3)
        metrics.apd_dispersion_reduction_pct = sigma_gel * coverage * 30
        metrics.repolarization_gradient = 0.5 * (1 - sigma_gel * coverage * 0.4)

        # Arrhythmia risk
        # Lower CV, higher APD dispersion = higher risk
        cv_risk = max(0, 1 - metrics.border_zone_cv_m_s / baseline_cv)
        apd_risk = metrics.apd_dispersion_ms / 50
        metrics.arrhythmia_vulnerability_index = (cv_risk + apd_risk) / 2
        metrics.reentry_circuit_probability = metrics.arrhythmia_vulnerability_index * 0.3

        return metrics

    def calculate_functional_metrics(
        self,
        design,
        patient_config,
        mechanical: FEBioMechanicalMetrics
    ) -> FunctionalMetrics:
        """Calculate 10 functional cardiac metrics."""

        metrics = FunctionalMetrics()

        # Baseline values
        metrics.baseline_LVEF_pct = patient_config.baseline_LVEF_pct
        metrics.baseline_GLS_pct = patient_config.baseline_GLS_pct

        # EF improvement based on mechanical support
        # Correlation: stress reduction -> improved contractility -> higher EF
        stress_effect = mechanical.wall_stress_reduction_pct * 0.15
        strain_effect = mechanical.strain_normalization_pct * 0.08

        metrics.delta_EF_pct = stress_effect + strain_effect + np.random.uniform(-0.5, 0.5)
        metrics.new_LVEF_pct = patient_config.baseline_LVEF_pct + metrics.delta_EF_pct

        # Volumes
        baseline_EDV = patient_config.baseline_EDV_mL
        baseline_ESV = patient_config.baseline_ESV_mL

        metrics.end_diastolic_volume_mL = baseline_EDV * 0.98  # Slight reverse remodeling
        metrics.end_systolic_volume_mL = metrics.end_diastolic_volume_mL * (1 - metrics.new_LVEF_pct / 100)
        metrics.delta_ESV_mL = metrics.end_systolic_volume_mL - baseline_ESV

        # Stroke volume
        baseline_SV = baseline_EDV * (patient_config.baseline_LVEF_pct / 100)
        metrics.stroke_volume_mL = metrics.end_diastolic_volume_mL - metrics.end_systolic_volume_mL
        metrics.stroke_volume_improvement_pct = (metrics.stroke_volume_mL - baseline_SV) / baseline_SV * 100

        # GLS improvement
        gls_improvement = mechanical.strain_normalization_pct * 0.15
        metrics.new_GLS_pct = patient_config.baseline_GLS_pct - gls_improvement  # More negative = better

        return metrics

    def calculate_integration_metrics(
        self,
        design,
        patient_config,
        mechanical: FEBioMechanicalMetrics
    ) -> IntegrationMetrics:
        """Calculate 8 integration metrics."""

        metrics = IntegrationMetrics()

        # Stiffness
        metrics.tissue_stiffness_kPa = 15.0  # Average healthy myocardium
        metrics.hydrogel_stiffness_kPa = design.stiffness_kPa
        metrics.stiffness_mismatch_ratio = abs(design.stiffness_kPa - 15.0) / 15.0

        # Interface stress (from mechanical)
        metrics.interface_shear_stress_kPa = mechanical.interface_stress_kPa * 0.3

        # Coverage
        coverage_map = {
            "scar_only": (100, 0),
            "scar_bz25": (100, 25),
            "scar_bz50": (100, 50),
            "scar_bz100": (100, 100),
        }
        scar_cov, bz_cov = coverage_map.get(design.patch_coverage, (50, 0))
        metrics.scar_coverage_pct = scar_cov
        metrics.border_zone_coverage_pct = bz_cov

        # Retention (based on degradation)
        t50 = design.degradation_days
        metrics.retention_fraction = np.exp(-30 * np.log(2) / t50)

        # Edge effects (higher at stiffness mismatch and thin patches)
        metrics.edge_effect_severity = metrics.stiffness_mismatch_ratio * (3.0 / design.patch_thickness_mm)

        return metrics

    def _extract_febio_metrics(self, febio_result: Dict) -> FEBioMechanicalMetrics:
        """Extract metrics from real FEBio output."""
        # This would parse actual FEBio .xplt output files
        # For now, return placeholder
        return FEBioMechanicalMetrics()

    def _extract_opencarp_metrics(self, opencarp_result: Dict) -> OpenCarpElectricalMetrics:
        """Extract metrics from real OpenCarp output."""
        # This would parse actual OpenCarp output files
        # For now, return placeholder
        return OpenCarpElectricalMetrics()


def get_all_metric_names() -> List[str]:
    """Return list of all 53 metric names."""
    return (
        ["mech_" + k for k in FEBioMechanicalMetrics().__dict__.keys()] +
        ["elec_" + k for k in OpenCarpElectricalMetrics().__dict__.keys()] +
        ["func_" + k for k in FunctionalMetrics().__dict__.keys()] +
        ["integ_" + k for k in IntegrationMetrics().__dict__.keys()]
    )
