#!/usr/bin/env python3
"""
Therapeutic Thresholds for Cardiac Hydrogel Validation

Defines clinically significant thresholds that must be exceeded
to claim therapeutic benefit (healing/support) of the infarct region.

Based on:
- Clinical trials of cardiac regenerative therapies
- FDA guidance for cardiac devices
- Published literature on myocardial infarction recovery
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class TherapeuticThresholds:
    """
    Clinically significant thresholds for claiming therapeutic benefit.

    A design must exceed MULTIPLE thresholds across categories to be
    considered therapeutically effective.
    """

    # TIER 1: PRIMARY EFFICACY (Must pass ALL for "Therapeutic")

    # Ejection Fraction Improvement
    # FDA considers ≥5% absolute improvement clinically meaningful
    MIN_DELTA_EF_PCT: float = 5.0
    TARGET_DELTA_EF_PCT: float = 8.0
    EXCEPTIONAL_DELTA_EF_PCT: float = 12.0

    # Border Zone Stress Reduction
    # ≥25% reduction needed to prevent adverse remodeling
    MIN_BZ_STRESS_REDUCTION_PCT: float = 25.0
    TARGET_BZ_STRESS_REDUCTION_PCT: float = 35.0
    EXCEPTIONAL_BZ_STRESS_REDUCTION_PCT: float = 50.0

    # Strain Normalization
    # ≥15% improvement indicates functional recovery
    MIN_STRAIN_NORMALIZATION_PCT: float = 15.0
    TARGET_STRAIN_NORMALIZATION_PCT: float = 25.0
    EXCEPTIONAL_STRAIN_NORMALIZATION_PCT: float = 40.0

    # TIER 2: SECONDARY EFFICACY (Must pass >=3 for "Supportive")

    # Global Longitudinal Strain (GLS) Improvement
    # ≥2% absolute improvement is clinically relevant
    MIN_DELTA_GLS_PCT: float = 2.0
    TARGET_DELTA_GLS_PCT: float = 3.5

    # End-Systolic Volume Reduction
    # ≥10mL reduction indicates reverse remodeling
    MIN_DELTA_ESV_ML: float = -10.0  # Negative = reduction (good)
    TARGET_DELTA_ESV_ML: float = -20.0

    # Wall Stress Reduction
    # Peak wall stress should decrease ≥20%
    MIN_WALL_STRESS_REDUCTION_PCT: float = 20.0
    TARGET_WALL_STRESS_REDUCTION_PCT: float = 30.0

    # Regional Strain Variance Reduction
    # Indicates more uniform contraction (≥20% reduction)
    MIN_STRAIN_VARIANCE_REDUCTION_PCT: float = 20.0

    # Stroke Work Increase
    # ≥15% increase in cardiac output efficiency
    MIN_STROKE_WORK_INCREASE_PCT: float = 15.0

    # TIER 3: SAFETY REQUIREMENTS (Must pass ALL)

    # Toxicity (0-1, lower is better)
    MAX_TOXICITY_SCORE: float = 0.10

    # Structural Integrity (0-1, higher is better)
    MIN_STRUCTURAL_INTEGRITY: float = 0.90

    # Rupture Risk (0-1, lower is better)
    MAX_RUPTURE_RISK: float = 0.05

    # Arrhythmia Risk (0-1, lower is better)
    MAX_ARRHYTHMIA_RISK: float = 0.15

    # Fibrosis Risk (0-1, lower is better)
    MAX_FIBROSIS_RISK: float = 0.20

    # Compliance Mismatch (ratio, lower is better)
    # Hydrogel stiffness should be within 50% of native tissue
    MAX_COMPLIANCE_MISMATCH: float = 0.50

    # TIER 4: ELECTRICAL FUNCTION (For conductive hydrogels)

    # Conduction Velocity Improvement
    # ≥20% improvement in border zone CV
    MIN_CV_IMPROVEMENT_PCT: float = 20.0
    TARGET_CV_IMPROVEMENT_PCT: float = 40.0

    # Activation Time Reduction
    # ≥15% reduction in total activation time
    MIN_ACTIVATION_TIME_REDUCTION_PCT: float = 15.0

    # APD Dispersion Reduction
    # ≥25% reduction indicates more uniform repolarization
    MIN_APD_DISPERSION_REDUCTION_PCT: float = 25.0

    # Conduction Block Prevention
    # Block risk should be <5%
    MAX_CONDUCTION_BLOCK_RISK: float = 0.05

    # TIER 5: DURABILITY (Long-term benefit)

    # 30-day Retention
    # ≥60% of hydrogel should remain at 30 days
    MIN_RETENTION_30DAYS: float = 0.60
    TARGET_RETENTION_30DAYS: float = 0.75

    # Degradation Profile
    # t50 should be ≥45 days for sustained benefit
    MIN_T50_DAYS: float = 45.0
    TARGET_T50_DAYS: float = 90.0


# THERAPEUTIC CLASSIFICATION

class TherapeuticClassifier:
    """Classifies designs based on therapeutic thresholds."""

    def __init__(self, thresholds: TherapeuticThresholds = None):
        self.thresholds = thresholds or TherapeuticThresholds()

    def classify(self, metrics: Dict) -> Dict:
        """
        Classify a design's therapeutic potential.

        Returns:
            Dict with classification and scores
        """
        results = {
            "tier1_efficacy": self._check_tier1(metrics),
            "tier2_efficacy": self._check_tier2(metrics),
            "tier3_safety": self._check_tier3(metrics),
            "tier4_electrical": self._check_tier4(metrics),
            "tier5_durability": self._check_tier5(metrics),
        }

        # Overall classification
        tier1_pass = results["tier1_efficacy"]["pass_all"]
        tier2_count = results["tier2_efficacy"]["pass_count"]
        tier3_pass = results["tier3_safety"]["pass_all"]

        if tier1_pass and tier3_pass and tier2_count >= 3:
            classification = "THERAPEUTIC"
            confidence = "HIGH"
        elif tier3_pass and (results["tier1_efficacy"]["pass_count"] >= 2) and tier2_count >= 2:
            classification = "SUPPORTIVE"
            confidence = "MODERATE"
        elif tier3_pass and (results["tier1_efficacy"]["pass_count"] >= 1):
            classification = "MARGINALLY_BENEFICIAL"
            confidence = "LOW"
        else:
            classification = "INSUFFICIENT"
            confidence = "NONE"

        # Compute therapeutic score (0-100)
        therapeutic_score = self._compute_therapeutic_score(metrics, results)

        results["classification"] = classification
        results["confidence"] = confidence
        results["therapeutic_score"] = therapeutic_score
        results["passes_minimum_therapeutic"] = classification in ["THERAPEUTIC", "SUPPORTIVE"]

        return results

    def _check_tier1(self, metrics: Dict) -> Dict:
        """Check Tier 1 primary efficacy metrics."""
        t = self.thresholds

        checks = {
            "delta_EF": {
                "value": metrics.get("delta_EF_pct", 0),
                "min": t.MIN_DELTA_EF_PCT,
                "target": t.TARGET_DELTA_EF_PCT,
                "exceptional": t.EXCEPTIONAL_DELTA_EF_PCT,
                "passed": metrics.get("delta_EF_pct", 0) >= t.MIN_DELTA_EF_PCT,
                "level": self._get_level(metrics.get("delta_EF_pct", 0),
                                        t.MIN_DELTA_EF_PCT, t.TARGET_DELTA_EF_PCT, t.EXCEPTIONAL_DELTA_EF_PCT)
            },
            "BZ_stress_reduction": {
                "value": metrics.get("delta_BZ_stress_reduction_pct", 0),
                "min": t.MIN_BZ_STRESS_REDUCTION_PCT,
                "target": t.TARGET_BZ_STRESS_REDUCTION_PCT,
                "exceptional": t.EXCEPTIONAL_BZ_STRESS_REDUCTION_PCT,
                "passed": metrics.get("delta_BZ_stress_reduction_pct", 0) >= t.MIN_BZ_STRESS_REDUCTION_PCT,
                "level": self._get_level(metrics.get("delta_BZ_stress_reduction_pct", 0),
                                        t.MIN_BZ_STRESS_REDUCTION_PCT, t.TARGET_BZ_STRESS_REDUCTION_PCT, t.EXCEPTIONAL_BZ_STRESS_REDUCTION_PCT)
            },
            "strain_normalization": {
                "value": metrics.get("strain_normalization_pct", 0),
                "min": t.MIN_STRAIN_NORMALIZATION_PCT,
                "target": t.TARGET_STRAIN_NORMALIZATION_PCT,
                "exceptional": t.EXCEPTIONAL_STRAIN_NORMALIZATION_PCT,
                "passed": metrics.get("strain_normalization_pct", 0) >= t.MIN_STRAIN_NORMALIZATION_PCT,
                "level": self._get_level(metrics.get("strain_normalization_pct", 0),
                                        t.MIN_STRAIN_NORMALIZATION_PCT, t.TARGET_STRAIN_NORMALIZATION_PCT, t.EXCEPTIONAL_STRAIN_NORMALIZATION_PCT)
            },
        }

        pass_count = sum(1 for c in checks.values() if c["passed"])

        return {
            "checks": checks,
            "pass_count": pass_count,
            "pass_all": pass_count == len(checks),
        }

    def _check_tier2(self, metrics: Dict) -> Dict:
        """Check Tier 2 secondary efficacy metrics."""
        t = self.thresholds

        checks = {
            "delta_GLS": {
                "value": metrics.get("delta_GLS_pct", 0),
                "min": t.MIN_DELTA_GLS_PCT,
                "passed": metrics.get("delta_GLS_pct", 0) >= t.MIN_DELTA_GLS_PCT,
            },
            "delta_ESV": {
                "value": metrics.get("delta_ESV_mL", 0),
                "min": t.MIN_DELTA_ESV_ML,
                "passed": metrics.get("delta_ESV_mL", 0) <= t.MIN_DELTA_ESV_ML,  # Negative is good
            },
            "wall_stress_reduction": {
                "value": metrics.get("wall_stress_reduction_pct", 0),
                "min": t.MIN_WALL_STRESS_REDUCTION_PCT,
                "passed": metrics.get("wall_stress_reduction_pct", 0) >= t.MIN_WALL_STRESS_REDUCTION_PCT,
            },
            "stroke_work_increase": {
                "value": metrics.get("stroke_work_increase_pct", 0),
                "min": t.MIN_STROKE_WORK_INCREASE_PCT,
                "passed": metrics.get("stroke_work_increase_pct", 0) >= t.MIN_STROKE_WORK_INCREASE_PCT,
            },
        }

        pass_count = sum(1 for c in checks.values() if c["passed"])

        return {
            "checks": checks,
            "pass_count": pass_count,
            "pass_all": pass_count == len(checks),
        }

    def _check_tier3(self, metrics: Dict) -> Dict:
        """Check Tier 3 safety metrics."""
        t = self.thresholds

        checks = {
            "toxicity": {
                "value": metrics.get("toxicity_score", 1.0),
                "max": t.MAX_TOXICITY_SCORE,
                "passed": metrics.get("toxicity_score", 1.0) <= t.MAX_TOXICITY_SCORE,
            },
            "structural_integrity": {
                "value": metrics.get("structural_integrity", 0),
                "min": t.MIN_STRUCTURAL_INTEGRITY,
                "passed": metrics.get("structural_integrity", 0) >= t.MIN_STRUCTURAL_INTEGRITY,
            },
            "rupture_risk": {
                "value": metrics.get("rupture_risk", 1.0),
                "max": t.MAX_RUPTURE_RISK,
                "passed": metrics.get("rupture_risk", 1.0) <= t.MAX_RUPTURE_RISK,
            },
            "arrhythmia_risk": {
                "value": metrics.get("arrhythmia_risk", 1.0),
                "max": t.MAX_ARRHYTHMIA_RISK,
                "passed": metrics.get("arrhythmia_risk", 1.0) <= t.MAX_ARRHYTHMIA_RISK,
            },
            "fibrosis_risk": {
                "value": metrics.get("fibrosis_risk", 1.0),
                "max": t.MAX_FIBROSIS_RISK,
                "passed": metrics.get("fibrosis_risk", 1.0) <= t.MAX_FIBROSIS_RISK,
            },
            "compliance_mismatch": {
                "value": metrics.get("compliance_mismatch", 1.0),
                "max": t.MAX_COMPLIANCE_MISMATCH,
                "passed": metrics.get("compliance_mismatch", 1.0) <= t.MAX_COMPLIANCE_MISMATCH,
            },
        }

        pass_count = sum(1 for c in checks.values() if c["passed"])

        return {
            "checks": checks,
            "pass_count": pass_count,
            "pass_all": pass_count == len(checks),
        }

    def _check_tier4(self, metrics: Dict) -> Dict:
        """Check Tier 4 electrical function metrics."""
        t = self.thresholds

        checks = {
            "cv_improvement": {
                "value": metrics.get("cv_improvement_pct", 0),
                "min": t.MIN_CV_IMPROVEMENT_PCT,
                "passed": metrics.get("cv_improvement_pct", 0) >= t.MIN_CV_IMPROVEMENT_PCT,
            },
            "activation_time_reduction": {
                "value": metrics.get("activation_time_reduction_pct", 0),
                "min": t.MIN_ACTIVATION_TIME_REDUCTION_PCT,
                "passed": metrics.get("activation_time_reduction_pct", 0) >= t.MIN_ACTIVATION_TIME_REDUCTION_PCT,
            },
            "apd_dispersion_reduction": {
                "value": metrics.get("apd_dispersion_reduction_pct", 0),
                "min": t.MIN_APD_DISPERSION_REDUCTION_PCT,
                "passed": metrics.get("apd_dispersion_reduction_pct", 0) >= t.MIN_APD_DISPERSION_REDUCTION_PCT,
            },
            "conduction_block_risk": {
                "value": metrics.get("conduction_block_risk", 1.0),
                "max": t.MAX_CONDUCTION_BLOCK_RISK,
                "passed": metrics.get("conduction_block_risk", 1.0) <= t.MAX_CONDUCTION_BLOCK_RISK,
            },
        }

        pass_count = sum(1 for c in checks.values() if c["passed"])

        return {
            "checks": checks,
            "pass_count": pass_count,
            "pass_all": pass_count == len(checks),
        }

    def _check_tier5(self, metrics: Dict) -> Dict:
        """Check Tier 5 durability metrics."""
        t = self.thresholds

        checks = {
            "retention_30days": {
                "value": metrics.get("retention_at_30days", 0),
                "min": t.MIN_RETENTION_30DAYS,
                "passed": metrics.get("retention_at_30days", 0) >= t.MIN_RETENTION_30DAYS,
            },
            "t50_days": {
                "value": metrics.get("hydrogel_t50_days", 0),
                "min": t.MIN_T50_DAYS,
                "passed": metrics.get("hydrogel_t50_days", 0) >= t.MIN_T50_DAYS,
            },
        }

        pass_count = sum(1 for c in checks.values() if c["passed"])

        return {
            "checks": checks,
            "pass_count": pass_count,
            "pass_all": pass_count == len(checks),
        }

    def _get_level(self, value: float, min_thresh: float, target: float, exceptional: float) -> str:
        """Get achievement level for a metric."""
        if value >= exceptional:
            return "EXCEPTIONAL"
        elif value >= target:
            return "TARGET"
        elif value >= min_thresh:
            return "MINIMUM"
        else:
            return "BELOW"

    def _compute_therapeutic_score(self, metrics: Dict, results: Dict) -> float:
        """Compute overall therapeutic score (0-100)."""
        score = 0.0

        # Tier 1: 40 points (primary efficacy)
        tier1 = results["tier1_efficacy"]
        for name, check in tier1["checks"].items():
            if check["level"] == "EXCEPTIONAL":
                score += 15
            elif check["level"] == "TARGET":
                score += 12
            elif check["level"] == "MINIMUM":
                score += 8

        # Tier 2: 20 points (secondary efficacy)
        tier2 = results["tier2_efficacy"]
        score += tier2["pass_count"] * 5

        # Tier 3: 25 points (safety)
        tier3 = results["tier3_safety"]
        score += tier3["pass_count"] * (25 / 6)

        # Tier 4: 10 points (electrical)
        tier4 = results["tier4_electrical"]
        score += tier4["pass_count"] * 2.5

        # Tier 5: 5 points (durability)
        tier5 = results["tier5_durability"]
        score += tier5["pass_count"] * 2.5

        return min(100, score)


# KEY FILTERING METRICS (10 metrics for Stage 1 filtering)

KEY_FILTERING_METRICS = [
    # Efficacy (5 metrics)
    "delta_EF_pct",
    "delta_BZ_stress_reduction_pct",
    "strain_normalization_pct",
    "predicted_optimal_prob",
    "reward",

    # Safety (5 metrics)
    "toxicity_score",
    "structural_integrity",
    "rupture_risk",
    "arrhythmia_risk",
    "compliance_mismatch",
]


def compute_filtering_score(design_metrics: Dict) -> float:
    """
    Compute quick filtering score from 10 key metrics.
    Used for initial filtering of 10M designs to top candidates.
    """
    score = 0.0

    # Efficacy components (60%)
    ef_score = min(20, design_metrics.get("delta_EF_pct", 0) * 2)
    stress_score = min(15, design_metrics.get("delta_BZ_stress_reduction_pct", 0) * 0.5)
    strain_score = min(10, design_metrics.get("strain_normalization_pct", 0) * 0.5)
    optimal_score = design_metrics.get("predicted_optimal_prob", 0) * 10
    reward_score = min(5, design_metrics.get("reward", 0) * 0.2)

    efficacy = ef_score + stress_score + strain_score + optimal_score + reward_score

    # Safety components (40%)
    toxicity_penalty = design_metrics.get("toxicity_score", 1.0) * 15
    integrity_bonus = design_metrics.get("structural_integrity", 0) * 10
    rupture_penalty = design_metrics.get("rupture_risk", 1.0) * 10
    arrhythmia_penalty = design_metrics.get("arrhythmia_risk", 1.0) * 8
    compliance_penalty = min(7, design_metrics.get("compliance_mismatch", 1.0) * 7)

    safety = 40 - toxicity_penalty + integrity_bonus - rupture_penalty - arrhythmia_penalty - compliance_penalty
    safety = max(0, safety)

    score = efficacy + safety
    return score
