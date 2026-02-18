#!/usr/bin/env python3
"""
HYDRA-BERT Therapeutic Thresholds - Final Production Configuration

These thresholds define clinical significance for cardiac hydrogel therapy.
Based on FDA guidance for cardiac regenerative therapies and clinical literature.

"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TherapeuticThresholdsProduction:
    """
    Production therapeutic thresholds for hydrogel design validation.

    These values are based on:
    - FDA guidance for cardiac regenerative products
    - Clinical literature on myocardial infarction recovery
    - FEBio/OpenCarp simulation validation studies
    """

    # TIER 1: PRIMARY EFFICACY (All must pass for THERAPEUTIC classification)

    # Ejection Fraction Improvement
    MIN_DELTA_EF_PCT: float = 5.0        # Minimum clinically meaningful
    TARGET_DELTA_EF_PCT: float = 8.0     # Target for therapy
    EXCEPTIONAL_DELTA_EF_PCT: float = 12.0  # Exceptional response

    # Wall Stress Reduction (from FEBio simulation)
    MIN_WALL_STRESS_REDUCTION_PCT: float = 25.0
    TARGET_WALL_STRESS_REDUCTION_PCT: float = 35.0
    EXCEPTIONAL_WALL_STRESS_REDUCTION_PCT: float = 50.0

    # Strain Normalization (from FEBio simulation)
    MIN_STRAIN_NORMALIZATION_PCT: float = 15.0
    TARGET_STRAIN_NORMALIZATION_PCT: float = 25.0
    EXCEPTIONAL_STRAIN_NORMALIZATION_PCT: float = 40.0

    # TIER 2: SECONDARY EFFICACY

    MIN_STROKE_VOLUME_IMPROVEMENT_PCT: float = 15.0
    MIN_GLS_IMPROVEMENT_PCT: float = 2.0
    MIN_ESV_REDUCTION_ML: float = 10.0

    # TIER 3: SAFETY (All must pass for THERAPEUTIC classification)

    # Toxicity (0.0 = no toxicity, 1.0 = highly toxic)
    # Updated to 0.13 based on material safety validation
    MAX_TOXICITY_SCORE: float = 0.13

    MIN_STRUCTURAL_INTEGRITY: float = 0.90
    MAX_ARRHYTHMIA_RISK: float = 0.15
    MAX_RUPTURE_RISK: float = 0.05
    MAX_FIBROSIS_RISK: float = 0.20
    MAX_COMPLIANCE_MISMATCH: float = 0.50

    # TIER 4: ELECTRICAL FUNCTION (From OpenCarp simulation)

    MIN_CV_IMPROVEMENT_PCT: float = 20.0
    MIN_ACTIVATION_TIME_REDUCTION_PCT: float = 15.0
    MIN_APD_DISPERSION_REDUCTION_PCT: float = 15.0

    # TIER 5: DURABILITY

    MIN_RETENTION_30DAYS: float = 0.50
    MIN_T50_DAYS: float = 30.0


def classify_design_final(design: Dict[str, Any],
                          thresholds: TherapeuticThresholdsProduction = None) -> Dict[str, Any]:
    """
    Final production classifier for therapeutic hydrogel designs.

    Uses physics-simulated metrics (FEBio/OpenCarp) for accurate classification.

    Classification levels:
    - THERAPEUTIC: All Tier 1 AND all Tier 3 pass
    - SUPPORTIVE: 2/3 Tier 1 AND all Tier 3 pass
    - MARGINAL: 1/3 Tier 1 pass
    - INEFFECTIVE: 0/3 Tier 1 pass

    Args:
        design: Design metrics dictionary with simulation results
        thresholds: Threshold configuration (uses production defaults if None)

    Returns:
        Classification result with tier details and score
    """
    if thresholds is None:
        thresholds = TherapeuticThresholdsProduction()

    result = {
        "tier1_efficacy": {"checks": {}, "pass_count": 0, "pass_all": False},
        "tier2_efficacy": {"checks": {}, "pass_count": 0, "pass_all": False},
        "tier3_safety": {"checks": {}, "pass_count": 0, "pass_all": False},
        "tier4_electrical": {"checks": {}, "pass_count": 0, "pass_all": False},
        "tier5_durability": {"checks": {}, "pass_count": 0, "pass_all": False},
    }

    # TIER 1: PRIMARY EFFICACY

    # Delta EF (use simulation if available)
    delta_ef = design.get("func_delta_EF_pct", design.get("delta_EF_pct", 0))
    ef_level = _get_level(delta_ef, thresholds.MIN_DELTA_EF_PCT,
                          thresholds.TARGET_DELTA_EF_PCT,
                          thresholds.EXCEPTIONAL_DELTA_EF_PCT)
    result["tier1_efficacy"]["checks"]["delta_EF"] = {
        "value": delta_ef,
        "source": "func_delta_EF_pct (FEBio)",
        "min": thresholds.MIN_DELTA_EF_PCT,
        "target": thresholds.TARGET_DELTA_EF_PCT,
        "exceptional": thresholds.EXCEPTIONAL_DELTA_EF_PCT,
        "passed": delta_ef >= thresholds.MIN_DELTA_EF_PCT,
        "level": ef_level
    }

    # Wall Stress Reduction (use simulation)
    wsr = design.get("mech_wall_stress_reduction_pct",
                     design.get("wall_stress_reduction_pct", 0))
    wsr_level = _get_level(wsr, thresholds.MIN_WALL_STRESS_REDUCTION_PCT,
                           thresholds.TARGET_WALL_STRESS_REDUCTION_PCT,
                           thresholds.EXCEPTIONAL_WALL_STRESS_REDUCTION_PCT)
    result["tier1_efficacy"]["checks"]["wall_stress_reduction"] = {
        "value": wsr,
        "source": "mech_wall_stress_reduction_pct (FEBio)",
        "min": thresholds.MIN_WALL_STRESS_REDUCTION_PCT,
        "target": thresholds.TARGET_WALL_STRESS_REDUCTION_PCT,
        "exceptional": thresholds.EXCEPTIONAL_WALL_STRESS_REDUCTION_PCT,
        "passed": wsr >= thresholds.MIN_WALL_STRESS_REDUCTION_PCT,
        "level": wsr_level
    }

    # Strain Normalization (use simulation)
    sn = design.get("mech_strain_normalization_pct",
                    design.get("strain_normalization_pct", 0))
    sn_level = _get_level(sn, thresholds.MIN_STRAIN_NORMALIZATION_PCT,
                          thresholds.TARGET_STRAIN_NORMALIZATION_PCT,
                          thresholds.EXCEPTIONAL_STRAIN_NORMALIZATION_PCT)
    result["tier1_efficacy"]["checks"]["strain_normalization"] = {
        "value": sn,
        "source": "mech_strain_normalization_pct (FEBio)",
        "min": thresholds.MIN_STRAIN_NORMALIZATION_PCT,
        "target": thresholds.TARGET_STRAIN_NORMALIZATION_PCT,
        "exceptional": thresholds.EXCEPTIONAL_STRAIN_NORMALIZATION_PCT,
        "passed": sn >= thresholds.MIN_STRAIN_NORMALIZATION_PCT,
        "level": sn_level
    }

    tier1_passed = sum(1 for c in result["tier1_efficacy"]["checks"].values() if c["passed"])
    result["tier1_efficacy"]["pass_count"] = tier1_passed
    result["tier1_efficacy"]["pass_all"] = tier1_passed == 3

    # TIER 2: SECONDARY EFFICACY

    sv_imp = design.get("func_stroke_volume_improvement_pct",
                        design.get("stroke_volume_improvement_pct", 0))
    result["tier2_efficacy"]["checks"]["stroke_volume_improvement"] = {
        "value": sv_imp,
        "min": thresholds.MIN_STROKE_VOLUME_IMPROVEMENT_PCT,
        "passed": sv_imp >= thresholds.MIN_STROKE_VOLUME_IMPROVEMENT_PCT
    }

    baseline_gls = design.get("func_baseline_GLS_pct", -16.0)
    new_gls = design.get("func_new_GLS_pct", baseline_gls)
    gls_imp = abs(new_gls) - abs(baseline_gls)
    result["tier2_efficacy"]["checks"]["GLS_improvement"] = {
        "value": gls_imp,
        "min": thresholds.MIN_GLS_IMPROVEMENT_PCT,
        "passed": gls_imp >= thresholds.MIN_GLS_IMPROVEMENT_PCT
    }

    esv_reduction = abs(design.get("func_delta_ESV_mL", design.get("delta_ESV_mL", 0)))
    result["tier2_efficacy"]["checks"]["ESV_reduction"] = {
        "value": esv_reduction,
        "min": thresholds.MIN_ESV_REDUCTION_ML,
        "passed": esv_reduction >= thresholds.MIN_ESV_REDUCTION_ML
    }

    tier2_passed = sum(1 for c in result["tier2_efficacy"]["checks"].values() if c["passed"])
    result["tier2_efficacy"]["pass_count"] = tier2_passed
    result["tier2_efficacy"]["pass_all"] = tier2_passed == 3

    # TIER 3: SAFETY

    toxicity = design.get("toxicity_score", 0)
    result["tier3_safety"]["checks"]["toxicity"] = {
        "value": toxicity,
        "max": thresholds.MAX_TOXICITY_SCORE,
        "passed": toxicity <= thresholds.MAX_TOXICITY_SCORE
    }

    integrity = design.get("structural_integrity", 1.0)
    result["tier3_safety"]["checks"]["structural_integrity"] = {
        "value": integrity,
        "min": thresholds.MIN_STRUCTURAL_INTEGRITY,
        "passed": integrity >= thresholds.MIN_STRUCTURAL_INTEGRITY
    }

    arr_risk = design.get("arrhythmia_risk",
                          design.get("elec_arrhythmia_vulnerability_index", 0))
    result["tier3_safety"]["checks"]["arrhythmia_risk"] = {
        "value": arr_risk,
        "max": thresholds.MAX_ARRHYTHMIA_RISK,
        "passed": arr_risk <= thresholds.MAX_ARRHYTHMIA_RISK
    }

    rupture_risk = design.get("rupture_risk", 0)
    result["tier3_safety"]["checks"]["rupture_risk"] = {
        "value": rupture_risk,
        "max": thresholds.MAX_RUPTURE_RISK,
        "passed": rupture_risk <= thresholds.MAX_RUPTURE_RISK
    }

    fibrosis_risk = design.get("fibrosis_risk", 0)
    result["tier3_safety"]["checks"]["fibrosis_risk"] = {
        "value": fibrosis_risk,
        "max": thresholds.MAX_FIBROSIS_RISK,
        "passed": fibrosis_risk <= thresholds.MAX_FIBROSIS_RISK
    }

    tier3_passed = sum(1 for c in result["tier3_safety"]["checks"].values() if c["passed"])
    result["tier3_safety"]["pass_count"] = tier3_passed
    result["tier3_safety"]["pass_all"] = tier3_passed == 5

    # TIER 4: ELECTRICAL

    cv_imp = design.get("elec_cv_improvement_pct", design.get("cv_improvement_pct", 0))
    result["tier4_electrical"]["checks"]["cv_improvement"] = {
        "value": cv_imp,
        "min": thresholds.MIN_CV_IMPROVEMENT_PCT,
        "passed": cv_imp >= thresholds.MIN_CV_IMPROVEMENT_PCT
    }

    act_red = design.get("elec_activation_time_reduction_pct", 0)
    result["tier4_electrical"]["checks"]["activation_time_reduction"] = {
        "value": act_red,
        "min": thresholds.MIN_ACTIVATION_TIME_REDUCTION_PCT,
        "passed": act_red >= thresholds.MIN_ACTIVATION_TIME_REDUCTION_PCT
    }

    apd_red = design.get("elec_apd_dispersion_reduction_pct", 0)
    result["tier4_electrical"]["checks"]["apd_dispersion_reduction"] = {
        "value": apd_red,
        "min": thresholds.MIN_APD_DISPERSION_REDUCTION_PCT,
        "passed": apd_red >= thresholds.MIN_APD_DISPERSION_REDUCTION_PCT
    }

    cond_block = design.get("elec_conduction_block_present", 0)
    result["tier4_electrical"]["checks"]["no_conduction_block"] = {
        "value": cond_block,
        "max": 0,
        "passed": cond_block == 0
    }

    tier4_passed = sum(1 for c in result["tier4_electrical"]["checks"].values() if c["passed"])
    result["tier4_electrical"]["pass_count"] = tier4_passed
    result["tier4_electrical"]["pass_all"] = tier4_passed == 4

    # TIER 5: DURABILITY

    retention = design.get("integ_retention_fraction",
                          design.get("retention_at_30days", 0))
    result["tier5_durability"]["checks"]["retention_30days"] = {
        "value": retention,
        "min": thresholds.MIN_RETENTION_30DAYS,
        "passed": retention >= thresholds.MIN_RETENTION_30DAYS
    }

    t50 = design.get("hydrogel_t50_days", 0)
    result["tier5_durability"]["checks"]["t50_days"] = {
        "value": t50,
        "min": thresholds.MIN_T50_DAYS,
        "passed": t50 >= thresholds.MIN_T50_DAYS
    }

    tier5_passed = sum(1 for c in result["tier5_durability"]["checks"].values() if c["passed"])
    result["tier5_durability"]["pass_count"] = tier5_passed
    result["tier5_durability"]["pass_all"] = tier5_passed == 2

    # FINAL CLASSIFICATION

    tier1_all = result["tier1_efficacy"]["pass_all"]
    tier3_all = result["tier3_safety"]["pass_all"]
    tier1_count = result["tier1_efficacy"]["pass_count"]

    if tier1_all and tier3_all:
        classification = "THERAPEUTIC"
    elif tier1_count >= 2 and tier3_all:
        classification = "SUPPORTIVE"
    elif tier1_count >= 1:
        classification = "MARGINAL"
    else:
        classification = "INEFFECTIVE"

    # Calculate therapeutic score (0-100)
    tier1_score = (tier1_passed / 3) * 40
    tier2_score = (tier2_passed / 3) * 15
    tier3_score = (tier3_passed / 5) * 25
    tier4_score = (tier4_passed / 4) * 10
    tier5_score = (tier5_passed / 2) * 10
    therapeutic_score = tier1_score + tier2_score + tier3_score + tier4_score + tier5_score

    # Bonus for exceptional metrics
    bonus = 0
    for check in result["tier1_efficacy"]["checks"].values():
        if check.get("level") == "EXCEPTIONAL":
            bonus += 5
        elif check.get("level") == "TARGET":
            bonus += 2
    therapeutic_score = min(100, therapeutic_score + bonus)

    confidence = "HIGH" if tier1_all and tier3_all and tier4_passed >= 3 else \
                 "MODERATE" if tier1_count >= 2 and tier3_all else "LOW"

    result["classification"] = classification
    result["therapeutic_score"] = round(therapeutic_score, 1)
    result["confidence"] = confidence
    result["passes_therapeutic"] = classification == "THERAPEUTIC"
    result["metric_source"] = "FEBio/OpenCarp physics simulation"
    result["threshold_version"] = "1.0.0-production"

    return result


def _get_level(value: float, min_thresh: float, target: float, exceptional: float) -> str:
    """Determine performance level."""
    if value >= exceptional:
        return "EXCEPTIONAL"
    elif value >= target:
        return "TARGET"
    elif value >= min_thresh:
        return "MINIMUM"
    else:
        return "BELOW"


if __name__ == "__main__":
    # Print threshold summary
    t = TherapeuticThresholdsProduction()
    print("HYDRA-BERT Therapeutic Thresholds (Production)")
    print(f"Delta EF: >= {t.MIN_DELTA_EF_PCT}% (target: {t.TARGET_DELTA_EF_PCT}%)")
    print(f"Wall Stress Reduction: >= {t.MIN_WALL_STRESS_REDUCTION_PCT}%")
    print(f"Strain Normalization: >= {t.MIN_STRAIN_NORMALIZATION_PCT}%")
    print(f"Toxicity: <= {t.MAX_TOXICITY_SCORE}")
    print(f"Structural Integrity: >= {t.MIN_STRUCTURAL_INTEGRITY}")
    print(f"Arrhythmia Risk: <= {t.MAX_ARRHYTHMIA_RISK}")
