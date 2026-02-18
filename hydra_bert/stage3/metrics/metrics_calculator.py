#!/usr/bin/env python3
"""
Metrics Calculator

Calculates all metrics matching the baseline data format including:
- Cardiac function metrics (EF, GLS, volumes)
- Stress/strain metrics (wall stress, BZ stress reduction)
- Safety scores (toxicity, structural integrity, fibrosis risk)
- Composite optimization scores
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class CardiacMetrics:
    """Cardiac function outcome metrics."""

    # Primary outcomes
    delta_EF_pct: float = 0.0
    new_LVEF_pct: float = 0.0
    delta_GLS_pct: float = 0.0
    new_GLS_pct: float = 0.0

    # Volume changes
    delta_EDV_mL: float = 0.0
    delta_ESV_mL: float = 0.0
    new_EDV_mL: float = 0.0
    new_ESV_mL: float = 0.0

    # Stress metrics
    delta_BZ_stress_reduction_pct: float = 0.0
    new_bz_stress_kPa: float = 0.0
    peak_wall_stress_kPa: float = 0.0

    # Strain metrics
    strain_normalization_pct: float = 0.0
    regional_strain_variance: float = 0.0

    # Work/energy
    stroke_work_J: float = 0.0
    ejection_work_J: float = 0.0

    # Classification
    is_optimal: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "delta_EF_pct": self.delta_EF_pct,
            "new_LVEF_pct": self.new_LVEF_pct,
            "delta_GLS_pct": self.delta_GLS_pct,
            "new_GLS_pct": self.new_GLS_pct,
            "delta_EDV_mL": self.delta_EDV_mL,
            "delta_ESV_mL": self.delta_ESV_mL,
            "new_EDV_mL": self.new_EDV_mL,
            "new_ESV_mL": self.new_ESV_mL,
            "delta_BZ_stress_reduction_pct": self.delta_BZ_stress_reduction_pct,
            "new_bz_stress_kPa": self.new_bz_stress_kPa,
            "peak_wall_stress_kPa": self.peak_wall_stress_kPa,
            "strain_normalization_pct": self.strain_normalization_pct,
            "regional_strain_variance": self.regional_strain_variance,
            "stroke_work_J": self.stroke_work_J,
            "ejection_work_J": self.ejection_work_J,
            "is_optimal": self.is_optimal,
        }


@dataclass
class SafetyMetrics:
    """Safety and biocompatibility metrics."""

    # Toxicity
    toxicity_score: float = 0.0  # 0-1, lower is better
    cytotoxicity_grade: int = 0  # 0-5 scale

    # Structural integrity
    structural_integrity: float = 1.0  # 0-1, higher is better
    degradation_rate: float = 0.0  # %/day
    retention_at_30days: float = 1.0  # fraction remaining

    # Fibrosis and scarring
    fibrosis_risk: float = 0.0  # 0-1
    inflammation_score: float = 0.0  # 0-1

    # Mechanical safety
    rupture_risk: float = 0.0  # 0-1
    interface_stress_kPa: float = 0.0
    compliance_mismatch: float = 0.0

    # Electrical safety
    arrhythmia_risk: float = 0.0  # 0-1
    conduction_block_risk: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "toxicity_score": self.toxicity_score,
            "cytotoxicity_grade": self.cytotoxicity_grade,
            "structural_integrity": self.structural_integrity,
            "degradation_rate": self.degradation_rate,
            "retention_at_30days": self.retention_at_30days,
            "fibrosis_risk": self.fibrosis_risk,
            "inflammation_score": self.inflammation_score,
            "rupture_risk": self.rupture_risk,
            "interface_stress_kPa": self.interface_stress_kPa,
            "compliance_mismatch": self.compliance_mismatch,
            "arrhythmia_risk": self.arrhythmia_risk,
            "conduction_block_risk": self.conduction_block_risk,
        }

    def is_safe(self) -> bool:
        """Check if design passes all safety thresholds."""
        return (
            self.toxicity_score < 0.15 and
            self.structural_integrity > 0.80 and
            self.fibrosis_risk < 0.30 and
            self.rupture_risk < 0.20 and
            self.arrhythmia_risk < 0.25
        )


@dataclass
class CompositeScore:
    """Composite optimization score combining all metrics."""

    # Component scores (0-100)
    efficacy_score: float = 0.0
    safety_score: float = 0.0
    durability_score: float = 0.0
    electrical_score: float = 0.0

    # Weights
    efficacy_weight: float = 0.40
    safety_weight: float = 0.35
    durability_weight: float = 0.15
    electrical_weight: float = 0.10

    # Final score
    total_score: float = 0.0
    rank: int = 0

    def compute_total(self) -> float:
        """Compute weighted total score."""
        self.total_score = (
            self.efficacy_score * self.efficacy_weight +
            self.safety_score * self.safety_weight +
            self.durability_score * self.durability_weight +
            self.electrical_score * self.electrical_weight
        )
        return self.total_score

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "efficacy_score": self.efficacy_score,
            "safety_score": self.safety_score,
            "durability_score": self.durability_score,
            "electrical_score": self.electrical_score,
            "total_score": self.total_score,
            "rank": self.rank,
        }


class MetricsCalculator:
    """
    Calculates comprehensive metrics for hydrogel designs.

    Matches all metrics from the baseline training data plus
    additional safety and optimization metrics.
    """

    def __init__(self, baseline_data_path: Optional[str] = None):
        """
        Initialize calculator.

        Args:
            baseline_data_path: Path to baseline CSV for reference distributions
        """
        self.baseline_stats = None
        if baseline_data_path:
            self._load_baseline_stats(baseline_data_path)

    def _load_baseline_stats(self, path: str):
        """Load baseline statistics for normalization."""
        import pandas as pd

        df = pd.read_csv(path)
        self.baseline_stats = {
            "delta_EF_mean": df["delta_EF_pct"].mean(),
            "delta_EF_std": df["delta_EF_pct"].std(),
            "delta_EF_max": df["delta_EF_pct"].max(),
            "optimal_rate": df["is_optimal"].mean(),
            "stress_reduction_mean": df["delta_BZ_stress_reduction_pct"].mean(),
            "strain_norm_mean": df["strain_normalization_pct"].mean(),
        }

    def calculate_cardiac_metrics(
        self,
        design,
        patient_config,
        simulation_result
    ) -> CardiacMetrics:
        """
        Calculate cardiac function metrics.

        Args:
            design: DesignCandidate object
            patient_config: PatientConfig object
            simulation_result: SimulationResult object

        Returns:
            CardiacMetrics object
        """
        metrics = CardiacMetrics()

        # Primary EF improvement
        metrics.delta_EF_pct = simulation_result.delta_ef_pct
        metrics.new_LVEF_pct = patient_config.baseline_LVEF_pct + metrics.delta_EF_pct

        # GLS improvement (approximately correlated with EF)
        metrics.delta_GLS_pct = simulation_result.delta_gls_pct
        metrics.new_GLS_pct = patient_config.baseline_GLS_pct - abs(metrics.delta_GLS_pct)

        # Volume changes (ESV decreases more than EDV with improved EF)
        # ESV = EDV * (1 - EF/100)
        old_ef = patient_config.baseline_LVEF_pct / 100
        new_ef = metrics.new_LVEF_pct / 100

        # Assume EDV stays relatively stable, ESV decreases
        metrics.new_EDV_mL = patient_config.baseline_EDV_mL * 0.98  # Slight remodeling
        metrics.new_ESV_mL = metrics.new_EDV_mL * (1 - new_ef)

        metrics.delta_EDV_mL = metrics.new_EDV_mL - patient_config.baseline_EDV_mL
        metrics.delta_ESV_mL = metrics.new_ESV_mL - patient_config.baseline_ESV_mL

        # Stress metrics
        metrics.delta_BZ_stress_reduction_pct = simulation_result.bz_stress_reduction_pct
        metrics.new_bz_stress_kPa = patient_config.bz_stress_kPa * (1 - metrics.delta_BZ_stress_reduction_pct/100)
        metrics.peak_wall_stress_kPa = simulation_result.peak_wall_stress_kPa

        # Strain metrics
        metrics.strain_normalization_pct = simulation_result.strain_normalization_pct
        metrics.regional_strain_variance = simulation_result.regional_strain_variance

        # Work metrics
        metrics.stroke_work_J = simulation_result.ejection_work_J
        metrics.ejection_work_J = simulation_result.ejection_work_J

        # Determine if optimal
        # Optimal: delta_EF >= 3% AND stress_reduction >= 15%
        metrics.is_optimal = (
            metrics.delta_EF_pct >= 3.0 and
            metrics.delta_BZ_stress_reduction_pct >= 15.0
        )

        return metrics

    def calculate_safety_metrics(
        self,
        design,
        patient_config,
        simulation_result
    ) -> SafetyMetrics:
        """
        Calculate safety and biocompatibility metrics.

        Args:
            design: DesignCandidate object
            patient_config: PatientConfig object
            simulation_result: SimulationResult object

        Returns:
            SafetyMetrics object
        """
        from ..generation import POLYMER_DATABASE

        metrics = SafetyMetrics()

        # Get polymer biocompatibility
        polymer_info = POLYMER_DATABASE.get(design.polymer_name, {})
        biocompatibility = polymer_info.get("biocompatibility", 0.9)

        # Toxicity (inverse of biocompatibility with some random variation)
        base_toxicity = 1 - biocompatibility
        metrics.toxicity_score = base_toxicity * (1 + np.random.normal(0, 0.1))
        metrics.toxicity_score = np.clip(metrics.toxicity_score, 0, 1)

        # Cytotoxicity grade (0-5 scale)
        metrics.cytotoxicity_grade = min(5, int(metrics.toxicity_score * 6))

        # Structural integrity (from model prediction or simulation)
        if hasattr(design, 'predicted_integrity'):
            metrics.structural_integrity = design.predicted_integrity
        else:
            # Based on stiffness and degradation
            stiffness_factor = design.stiffness_kPa / 50.0  # Normalized
            degradation_factor = (180 - design.degradation_days) / 180.0
            metrics.structural_integrity = 0.7 + 0.2 * stiffness_factor + 0.1 * degradation_factor
            metrics.structural_integrity = np.clip(metrics.structural_integrity, 0.5, 1.0)

        # Degradation rate
        metrics.degradation_rate = 100 / design.degradation_days  # % per day

        # Retention at 30 days
        metrics.retention_at_30days = simulation_result.retention_fraction

        # Fibrosis risk
        metrics.fibrosis_risk = simulation_result.fibrosis_risk

        # Inflammation (correlated with toxicity)
        metrics.inflammation_score = metrics.toxicity_score * 0.8 + np.random.uniform(0, 0.1)
        metrics.inflammation_score = np.clip(metrics.inflammation_score, 0, 1)

        # Rupture risk
        metrics.rupture_risk = simulation_result.rupture_risk

        # Interface stress
        metrics.interface_stress_kPa = simulation_result.interface_stress_kPa

        # Compliance mismatch (difference between gel and tissue stiffness)
        tissue_stiffness = 15.0  # kPa average myocardium
        metrics.compliance_mismatch = abs(design.stiffness_kPa - tissue_stiffness) / tissue_stiffness

        # Electrical safety
        metrics.arrhythmia_risk = simulation_result.arrhythmia_risk

        # Conduction block risk (high stiffness with low conductivity in BZ)
        if design.conductivity_S_m < 0.1 and design.stiffness_kPa > 30:
            metrics.conduction_block_risk = 0.3
        else:
            metrics.conduction_block_risk = 0.05

        return metrics

    def calculate_composite_score(
        self,
        cardiac_metrics: CardiacMetrics,
        safety_metrics: SafetyMetrics,
        design,
        simulation_result
    ) -> CompositeScore:
        """
        Calculate composite optimization score.

        Args:
            cardiac_metrics: CardiacMetrics object
            safety_metrics: SafetyMetrics object
            design: DesignCandidate object
            simulation_result: SimulationResult object

        Returns:
            CompositeScore object
        """
        score = CompositeScore()

        # Efficacy score (0-100)
        # Based on EF improvement, stress reduction, strain normalization
        ef_score = min(100, cardiac_metrics.delta_EF_pct * 10)  # 10% EF = 100 points
        stress_score = min(100, cardiac_metrics.delta_BZ_stress_reduction_pct * 4)  # 25% = 100 points
        strain_score = min(100, cardiac_metrics.strain_normalization_pct * 5)  # 20% = 100 points

        score.efficacy_score = (ef_score * 0.5 + stress_score * 0.3 + strain_score * 0.2)

        # Safety score (0-100)
        # Penalize toxicity, poor integrity, high risks
        toxicity_penalty = safety_metrics.toxicity_score * 50
        integrity_bonus = safety_metrics.structural_integrity * 30
        risk_penalty = (
            safety_metrics.fibrosis_risk * 10 +
            safety_metrics.rupture_risk * 20 +
            safety_metrics.arrhythmia_risk * 15
        )

        score.safety_score = max(0, 100 - toxicity_penalty - risk_penalty + integrity_bonus - 30)
        score.safety_score = min(100, score.safety_score)

        # Durability score (0-100)
        # Based on retention and degradation profile
        retention_score = safety_metrics.retention_at_30days * 60
        degradation_score = min(40, design.degradation_days / 180 * 40)

        score.durability_score = retention_score + degradation_score

        # Electrical score (0-100)
        # Based on conduction improvement and arrhythmia risk
        if design.conductivity_S_m > 0.1:
            conduction_bonus = min(50, design.conductivity_S_m * 50)
        else:
            conduction_bonus = 0

        arrhythmia_penalty = safety_metrics.arrhythmia_risk * 50
        conduction_block_penalty = safety_metrics.conduction_block_risk * 30

        score.electrical_score = max(0, 50 + conduction_bonus - arrhythmia_penalty - conduction_block_penalty)
        score.electrical_score = min(100, score.electrical_score)

        # Compute total
        score.compute_total()

        return score

    def calculate_all_metrics(
        self,
        design,
        patient_config,
        simulation_result
    ) -> Dict:
        """
        Calculate all metrics for a design.

        Returns dictionary with all metrics matching baseline data format.
        """
        cardiac = self.calculate_cardiac_metrics(design, patient_config, simulation_result)
        safety = self.calculate_safety_metrics(design, patient_config, simulation_result)
        composite = self.calculate_composite_score(cardiac, safety, design, simulation_result)

        # Compile all metrics
        all_metrics = {
            # Identifiers
            "design_id": design.design_id,
            "patient_id": design.patient_id,

            # Design parameters
            "polymer_name": design.polymer_name,
            "polymer_SMILES": design.polymer_smiles,
            "polymer_category": design.polymer_category,
            "hydrogel_E_kPa": design.stiffness_kPa,
            "hydrogel_t50_days": design.degradation_days,
            "hydrogel_conductivity_S_m": design.conductivity_S_m,
            "patch_thickness_mm": design.patch_thickness_mm,
            "patch_coverage": design.patch_coverage,

            # Cardiac metrics (matching baseline format)
            "delta_EF_pct": cardiac.delta_EF_pct,
            "new_LVEF_pct": cardiac.new_LVEF_pct,
            "delta_BZ_stress_reduction_pct": cardiac.delta_BZ_stress_reduction_pct,
            "strain_normalization_pct": cardiac.strain_normalization_pct,
            "is_optimal": cardiac.is_optimal,

            # Extended cardiac metrics
            "delta_GLS_pct": cardiac.delta_GLS_pct,
            "new_GLS_pct": cardiac.new_GLS_pct,
            "delta_EDV_mL": cardiac.delta_EDV_mL,
            "delta_ESV_mL": cardiac.delta_ESV_mL,
            "new_EDV_mL": cardiac.new_EDV_mL,
            "new_ESV_mL": cardiac.new_ESV_mL,
            "peak_wall_stress_kPa": cardiac.peak_wall_stress_kPa,
            "ejection_work_J": cardiac.ejection_work_J,

            # Safety metrics
            "toxicity_score": safety.toxicity_score,
            "structural_integrity": safety.structural_integrity,
            "fibrosis_risk": safety.fibrosis_risk,
            "rupture_risk": safety.rupture_risk,
            "arrhythmia_risk": safety.arrhythmia_risk,
            "retention_at_30days": safety.retention_at_30days,
            "compliance_mismatch": safety.compliance_mismatch,

            # Scores
            "stiffness_score": self._compute_stiffness_score(design.stiffness_kPa),
            "thickness_score": self._compute_thickness_score(design.patch_thickness_mm),
            "coverage_score": self._compute_coverage_score(design.patch_coverage),
            "conductivity_bonus": design.conductivity_S_m * 0.1,

            # Composite scores
            "efficacy_score": composite.efficacy_score,
            "safety_score": composite.safety_score,
            "durability_score": composite.durability_score,
            "electrical_score": composite.electrical_score,
            "total_score": composite.total_score,
            "rank": composite.rank,

            # Simulation metadata
            "simulation_success": simulation_result.success,
            "simulation_time_s": simulation_result.simulation_time_s,
        }

        return all_metrics

    def _compute_stiffness_score(self, stiffness_kPa: float) -> float:
        """Compute stiffness score (optimal around 10-20 kPa for cardiac tissue)."""
        optimal = 15.0
        score = 1.0 - 0.05 * abs(stiffness_kPa - optimal)
        return np.clip(score, 0, 1)

    def _compute_thickness_score(self, thickness_mm: float) -> float:
        """Compute thickness score (optimal around 1.5-2.5 mm)."""
        if thickness_mm < 0.5:
            return 0.5
        elif thickness_mm < 1.5:
            return 0.5 + (thickness_mm - 0.5) * 0.5
        elif thickness_mm < 2.5:
            return 1.0
        else:
            return max(0.5, 1.0 - (thickness_mm - 2.5) * 0.2)

    def _compute_coverage_score(self, coverage: str) -> float:
        """Compute coverage score."""
        scores = {
            "scar_only": 0.5,
            "scar_bz25": 0.72,
            "scar_bz50": 0.88,
            "scar_bz100": 1.0,
        }
        return scores.get(coverage, 0.5)

    def rank_designs(
        self,
        designs_with_metrics: List[Dict],
        primary_metric: str = "total_score"
    ) -> List[Dict]:
        """
        Rank designs by specified metric.

        Args:
            designs_with_metrics: List of metric dictionaries
            primary_metric: Metric to sort by

        Returns:
            Sorted list with rank assignments
        """
        # Sort by primary metric (descending)
        sorted_designs = sorted(
            designs_with_metrics,
            key=lambda x: x.get(primary_metric, 0),
            reverse=True
        )

        # Assign ranks
        for i, design in enumerate(sorted_designs):
            design["rank"] = i + 1

        return sorted_designs

    def get_top_designs(
        self,
        designs_with_metrics: List[Dict],
        top_k: int = 100,
        safety_filter: bool = True
    ) -> List[Dict]:
        """
        Get top K designs after filtering and ranking.

        Args:
            designs_with_metrics: List of metric dictionaries
            top_k: Number of top designs to return
            safety_filter: Whether to filter by safety thresholds

        Returns:
            Top K designs
        """
        filtered = designs_with_metrics

        if safety_filter:
            filtered = [
                d for d in filtered
                if d.get("toxicity_score", 1.0) < 0.15 and
                   d.get("structural_integrity", 0.0) > 0.80 and
                   d.get("rupture_risk", 1.0) < 0.20 and
                   d.get("arrhythmia_risk", 1.0) < 0.25
            ]

        # Rank
        ranked = self.rank_designs(filtered)

        return ranked[:top_k]
