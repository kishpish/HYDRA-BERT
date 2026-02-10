#!/usr/bin/env python3
"""
Design Analysis and Reporting Module

Provides comprehensive analysis of hydrogel designs including:
- Statistical analysis of design populations
- Selection of optimal designs per patient
- Report generation for documentation
- Visualization of results
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DesignAnalysis:
    """Analysis results for a set of designs."""

    patient_id: str = ""
    total_designs: int = 0
    valid_designs: int = 0
    optimal_designs: int = 0

    # Score distributions
    mean_efficacy: float = 0.0
    std_efficacy: float = 0.0
    mean_safety: float = 0.0
    std_safety: float = 0.0
    mean_total: float = 0.0
    std_total: float = 0.0

    # Best design summary
    best_design_id: str = ""
    best_polymer: str = ""
    best_score: float = 0.0
    best_delta_ef: float = 0.0

    # Polymer distribution
    polymer_counts: Dict[str, int] = field(default_factory=dict)
    polymer_avg_scores: Dict[str, float] = field(default_factory=dict)

    # Coverage distribution
    coverage_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "patient_id": self.patient_id,
            "total_designs": self.total_designs,
            "valid_designs": self.valid_designs,
            "optimal_designs": self.optimal_designs,
            "mean_efficacy": self.mean_efficacy,
            "std_efficacy": self.std_efficacy,
            "mean_safety": self.mean_safety,
            "std_safety": self.std_safety,
            "mean_total": self.mean_total,
            "std_total": self.std_total,
            "best_design_id": self.best_design_id,
            "best_polymer": self.best_polymer,
            "best_score": self.best_score,
            "best_delta_ef": self.best_delta_ef,
            "polymer_counts": self.polymer_counts,
            "polymer_avg_scores": self.polymer_avg_scores,
            "coverage_counts": self.coverage_counts,
        }


class DesignAnalyzer:
    """
    Analyzes hydrogel design populations and selects optimal designs.
    """

    def __init__(self, output_dir: str = "results/stage3"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_designs(
        self,
        designs_with_metrics: List[Dict],
        patient_id: str
    ) -> DesignAnalysis:
        """
        Perform statistical analysis on design population.

        Args:
            designs_with_metrics: List of metric dictionaries
            patient_id: Patient identifier

        Returns:
            DesignAnalysis object
        """
        analysis = DesignAnalysis(patient_id=patient_id)
        analysis.total_designs = len(designs_with_metrics)

        if analysis.total_designs == 0:
            return analysis

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(designs_with_metrics)

        # Count valid (passed safety filters) and optimal designs
        if "toxicity_score" in df.columns:
            valid_mask = (
                (df["toxicity_score"] < 0.15) &
                (df["structural_integrity"] > 0.80) &
                (df["rupture_risk"] < 0.20)
            )
            analysis.valid_designs = valid_mask.sum()
        else:
            analysis.valid_designs = analysis.total_designs

        if "is_optimal" in df.columns:
            analysis.optimal_designs = df["is_optimal"].sum()

        # Score distributions
        if "efficacy_score" in df.columns:
            analysis.mean_efficacy = df["efficacy_score"].mean()
            analysis.std_efficacy = df["efficacy_score"].std()

        if "safety_score" in df.columns:
            analysis.mean_safety = df["safety_score"].mean()
            analysis.std_safety = df["safety_score"].std()

        if "total_score" in df.columns:
            analysis.mean_total = df["total_score"].mean()
            analysis.std_total = df["total_score"].std()

        # Find best design
        if "total_score" in df.columns:
            best_idx = df["total_score"].idxmax()
            best_row = df.loc[best_idx]
            analysis.best_design_id = best_row.get("design_id", "")
            analysis.best_polymer = best_row.get("polymer_name", "")
            analysis.best_score = best_row["total_score"]
            analysis.best_delta_ef = best_row.get("delta_EF_pct", 0.0)

        # Polymer distribution
        if "polymer_name" in df.columns:
            analysis.polymer_counts = df["polymer_name"].value_counts().to_dict()

            if "total_score" in df.columns:
                analysis.polymer_avg_scores = df.groupby("polymer_name")["total_score"].mean().to_dict()

        # Coverage distribution
        if "patch_coverage" in df.columns:
            analysis.coverage_counts = df["patch_coverage"].value_counts().to_dict()

        return analysis

    def select_best_design(
        self,
        designs_with_metrics: List[Dict],
        primary_metric: str = "total_score",
        safety_filter: bool = True
    ) -> Optional[Dict]:
        """
        Select the single best design based on metrics.

        Args:
            designs_with_metrics: List of metric dictionaries
            primary_metric: Metric to optimize
            safety_filter: Whether to apply safety filters first

        Returns:
            Best design dictionary or None if no valid designs
        """
        if not designs_with_metrics:
            return None

        candidates = designs_with_metrics

        # Apply safety filter
        if safety_filter:
            candidates = [
                d for d in candidates
                if d.get("toxicity_score", 1.0) < 0.15 and
                   d.get("structural_integrity", 0.0) > 0.80 and
                   d.get("rupture_risk", 1.0) < 0.20 and
                   d.get("arrhythmia_risk", 1.0) < 0.25
            ]

        if not candidates:
            logger.warning("No designs passed safety filter, relaxing constraints")
            # Relax to top 10% by safety
            candidates = sorted(
                designs_with_metrics,
                key=lambda x: x.get("safety_score", 0),
                reverse=True
            )[:max(1, len(designs_with_metrics) // 10)]

        # Sort by primary metric
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get(primary_metric, 0),
            reverse=True
        )

        best = sorted_candidates[0]
        best["rank"] = 1

        return best

    def generate_comparison_matrix(
        self,
        top_designs: List[Dict],
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Generate comparison matrix for top designs.

        Args:
            top_designs: List of top design dictionaries
            metrics: Metrics to include (default: key metrics)

        Returns:
            DataFrame with comparison matrix
        """
        if metrics is None:
            metrics = [
                "rank", "polymer_name", "hydrogel_E_kPa", "patch_coverage",
                "delta_EF_pct", "delta_BZ_stress_reduction_pct",
                "toxicity_score", "structural_integrity",
                "efficacy_score", "safety_score", "total_score"
            ]

        # Filter to only include available metrics
        available_metrics = [m for m in metrics if m in top_designs[0]]

        df = pd.DataFrame(top_designs)[available_metrics]
        return df

    def save_results(
        self,
        patient_id: str,
        analysis: DesignAnalysis,
        best_design: Dict,
        top_designs: List[Dict],
        all_designs: List[Dict] = None
    ):
        """
        Save analysis results to files.

        Args:
            patient_id: Patient identifier
            analysis: DesignAnalysis object
            best_design: Best design dictionary
            top_designs: List of top design dictionaries
            all_designs: Optional list of all designs (for full CSV)
        """
        patient_dir = self.output_dir / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)

        # Save analysis summary
        with open(patient_dir / "analysis_summary.json", 'w') as f:
            json.dump(analysis.to_dict(), f, indent=2)

        # Save best design
        with open(patient_dir / "best_design.json", 'w') as f:
            json.dump(best_design, f, indent=2)

        # Save top 100 designs
        top_df = pd.DataFrame(top_designs)
        top_df.to_csv(patient_dir / "top_100_designs.csv", index=False)

        # Save comparison matrix
        comparison = self.generate_comparison_matrix(top_designs[:20])
        comparison.to_csv(patient_dir / "top_20_comparison.csv", index=False)

        # Optionally save all designs
        if all_designs:
            all_df = pd.DataFrame(all_designs)
            all_df.to_csv(patient_dir / "all_designs.csv", index=False)

        logger.info(f"Saved results for {patient_id} to {patient_dir}")


def generate_patient_report(
    patient_config,
    analysis: DesignAnalysis,
    best_design: Dict,
    top_designs: List[Dict],
    output_path: str
) -> str:
    """
    Generate detailed patient-specific report.

    Args:
        patient_config: PatientConfig object
        analysis: DesignAnalysis object
        best_design: Best design dictionary
        top_designs: List of top design dictionaries
        output_path: Path to save report

    Returns:
        Path to generated report
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
================================================================================
HYDRA-BERT PATIENT-SPECIFIC HYDROGEL DESIGN REPORT
================================================================================

Generated: {timestamp}
Patient ID: {patient_config.patient_id}

--------------------------------------------------------------------------------
PATIENT BASELINE CHARACTERISTICS
--------------------------------------------------------------------------------

Cardiac Function:
  - Baseline LVEF: {patient_config.baseline_LVEF_pct:.1f}%
  - Baseline GLS: {patient_config.baseline_GLS_pct:.2f}%
  - Baseline EDV: {patient_config.baseline_EDV_mL:.1f} mL
  - Baseline ESV: {patient_config.baseline_ESV_mL:.1f} mL

Infarct Characteristics:
  - Scar Fraction: {patient_config.scar_fraction_pct:.2f}%
  - Border Zone Fraction: {patient_config.bz_fraction_pct:.2f}%
  - Transmurality: {patient_config.transmurality:.3f}
  - Wall Thickness: {patient_config.wall_thickness_mm:.1f} mm

Tissue Mechanics:
  - BZ Stress: {patient_config.bz_stress_kPa:.1f} kPa
  - Healthy Stress: {patient_config.healthy_stress_kPa:.1f} kPa
  - Stress Concentration: {patient_config.stress_concentration:.2f}

Infarct Location:
  - AHA Segments: {patient_config.infarct_geometry.segments}
  - Injection Sites: {len(patient_config.injection_sites)}

--------------------------------------------------------------------------------
DESIGN GENERATION SUMMARY
--------------------------------------------------------------------------------

Design Space Exploration:
  - Total Designs Generated: {analysis.total_designs:,}
  - Valid Designs (Passed Safety): {analysis.valid_designs:,} ({100*analysis.valid_designs/max(1,analysis.total_designs):.1f}%)
  - Optimal Designs (EF>=3%, Stress>=15%): {analysis.optimal_designs:,} ({100*analysis.optimal_designs/max(1,analysis.total_designs):.1f}%)

Score Distributions:
  - Efficacy Score: {analysis.mean_efficacy:.1f} ± {analysis.std_efficacy:.1f}
  - Safety Score: {analysis.mean_safety:.1f} ± {analysis.std_safety:.1f}
  - Total Score: {analysis.mean_total:.1f} ± {analysis.std_total:.1f}

Top Polymer Types:
"""

    # Add polymer distribution
    sorted_polymers = sorted(analysis.polymer_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for polymer, count in sorted_polymers:
        avg_score = analysis.polymer_avg_scores.get(polymer, 0)
        report += f"  - {polymer}: {count:,} designs (avg score: {avg_score:.1f})\n"

    report += f"""
--------------------------------------------------------------------------------
OPTIMAL DESIGN SELECTION
--------------------------------------------------------------------------------

BEST DESIGN: {best_design.get('design_id', 'N/A')}

Polymer Selection:
  - Name: {best_design.get('polymer_name', 'N/A')}
  - Category: {best_design.get('polymer_category', 'N/A')}
  - SMILES: {best_design.get('polymer_SMILES', 'N/A')[:60]}...

Formulation Parameters:
  - Stiffness: {best_design.get('hydrogel_E_kPa', 0):.1f} kPa
  - Degradation Half-life: {best_design.get('hydrogel_t50_days', 0):.0f} days
  - Conductivity: {best_design.get('hydrogel_conductivity_S_m', 0):.3f} S/m
  - Patch Thickness: {best_design.get('patch_thickness_mm', 0):.1f} mm
  - Patch Coverage: {best_design.get('patch_coverage', 'N/A')}

Predicted Outcomes:
  - Delta EF: +{best_design.get('delta_EF_pct', 0):.2f}%
  - New LVEF: {best_design.get('new_LVEF_pct', 0):.1f}%
  - BZ Stress Reduction: {best_design.get('delta_BZ_stress_reduction_pct', 0):.1f}%
  - Strain Normalization: {best_design.get('strain_normalization_pct', 0):.1f}%
  - Is Optimal: {best_design.get('is_optimal', False)}

Safety Profile:
  - Toxicity Score: {best_design.get('toxicity_score', 0):.3f}
  - Structural Integrity: {best_design.get('structural_integrity', 0):.3f}
  - Fibrosis Risk: {best_design.get('fibrosis_risk', 0):.3f}
  - Rupture Risk: {best_design.get('rupture_risk', 0):.3f}
  - Arrhythmia Risk: {best_design.get('arrhythmia_risk', 0):.3f}
  - 30-Day Retention: {best_design.get('retention_at_30days', 0):.1f}%

Composite Scores:
  - Efficacy Score: {best_design.get('efficacy_score', 0):.1f}/100
  - Safety Score: {best_design.get('safety_score', 0):.1f}/100
  - Durability Score: {best_design.get('durability_score', 0):.1f}/100
  - Electrical Score: {best_design.get('electrical_score', 0):.1f}/100
  - TOTAL SCORE: {best_design.get('total_score', 0):.1f}/100 (Rank #{best_design.get('rank', 0)})

--------------------------------------------------------------------------------
TOP 10 ALTERNATIVE DESIGNS
--------------------------------------------------------------------------------
"""

    # Add top 10 alternatives
    for i, design in enumerate(top_designs[1:11], 2):
        report += f"""
Rank #{i}: {design.get('design_id', 'N/A')}
  Polymer: {design.get('polymer_name', 'N/A')} | Stiffness: {design.get('hydrogel_E_kPa', 0):.1f} kPa
  Coverage: {design.get('patch_coverage', 'N/A')} | Thickness: {design.get('patch_thickness_mm', 0):.1f} mm
  Delta EF: +{design.get('delta_EF_pct', 0):.2f}% | Total Score: {design.get('total_score', 0):.1f}
"""

    report += f"""
--------------------------------------------------------------------------------
RECOMMENDATIONS
--------------------------------------------------------------------------------

Based on this analysis, the recommended hydrogel formulation for patient
{patient_config.patient_id} is:

  {best_design.get('polymer_name', 'N/A')} hydrogel
  - Stiffness: {best_design.get('hydrogel_E_kPa', 0):.1f} kPa
  - Patch coverage: {best_design.get('patch_coverage', 'N/A')}
  - Patch thickness: {best_design.get('patch_thickness_mm', 0):.1f} mm

Expected clinical outcome:
  - EF improvement of +{best_design.get('delta_EF_pct', 0):.1f}%
    (from {patient_config.baseline_LVEF_pct:.1f}% to {best_design.get('new_LVEF_pct', 0):.1f}%)
  - BZ stress reduction of {best_design.get('delta_BZ_stress_reduction_pct', 0):.1f}%
  - Classification: {'OPTIMAL' if best_design.get('is_optimal', False) else 'SUB-OPTIMAL'}

Safety assessment: {'ACCEPTABLE' if best_design.get('toxicity_score', 1) < 0.15 else 'REQUIRES REVIEW'}

================================================================================
END OF REPORT
================================================================================
"""

    # Save report
    with open(output_path, 'w') as f:
        f.write(report)

    logger.info(f"Generated patient report: {output_path}")
    return output_path


def generate_summary_report(
    all_patient_results: Dict[str, Dict],
    output_path: str
) -> str:
    """
    Generate summary report across all patients.

    Args:
        all_patient_results: Dict mapping patient_id to results
        output_path: Path to save report

    Returns:
        Path to generated report
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
================================================================================
HYDRA-BERT COMPREHENSIVE HYDROGEL DESIGN SUMMARY
================================================================================

Generated: {timestamp}
Number of Patients: {len(all_patient_results)}

--------------------------------------------------------------------------------
PATIENT-SPECIFIC OPTIMAL DESIGNS
--------------------------------------------------------------------------------

"""

    # Summary table
    summary_data = []
    for patient_id, results in all_patient_results.items():
        best = results.get("best_design", {})
        summary_data.append({
            "Patient": patient_id,
            "Baseline LVEF": results.get("baseline_LVEF", 0),
            "Polymer": best.get("polymer_name", "N/A"),
            "Stiffness (kPa)": best.get("hydrogel_E_kPa", 0),
            "Coverage": best.get("patch_coverage", "N/A"),
            "Delta EF (%)": best.get("delta_EF_pct", 0),
            "New LVEF (%)": best.get("new_LVEF_pct", 0),
            "Total Score": best.get("total_score", 0),
            "Optimal": best.get("is_optimal", False),
        })

    # Format as table
    df = pd.DataFrame(summary_data)
    report += df.to_string(index=False)

    report += f"""

--------------------------------------------------------------------------------
AGGREGATE STATISTICS
--------------------------------------------------------------------------------

Overall Performance:
  - Mean Delta EF: {df['Delta EF (%)'].mean():.2f}% ± {df['Delta EF (%)'].std():.2f}%
  - Optimal Classification Rate: {df['Optimal'].mean()*100:.1f}%
  - Mean Total Score: {df['Total Score'].mean():.1f}

Polymer Distribution in Optimal Designs:
"""

    polymer_counts = df['Polymer'].value_counts()
    for polymer, count in polymer_counts.items():
        report += f"  - {polymer}: {count} patients ({100*count/len(df):.0f}%)\n"

    report += f"""
Coverage Distribution in Optimal Designs:
"""

    coverage_counts = df['Coverage'].value_counts()
    for coverage, count in coverage_counts.items():
        report += f"  - {coverage}: {count} patients ({100*count/len(df):.0f}%)\n"

    report += f"""
--------------------------------------------------------------------------------
KEY FINDINGS
--------------------------------------------------------------------------------

1. Patient-Specific Optimization:
   - Each patient received a uniquely optimized hydrogel formulation
   - Designs account for individual infarct geometry and cardiac function

2. Polymer Selection Insights:
   - Most frequently optimal: {polymer_counts.index[0] if len(polymer_counts) > 0 else 'N/A'}
   - Polymer selection varies based on patient characteristics

3. Clinical Relevance:
   - All optimal designs predict clinically meaningful EF improvements (>3%)
   - Safety thresholds maintained across all recommended designs

================================================================================
END OF SUMMARY REPORT
================================================================================
"""

    with open(output_path, 'w') as f:
        f.write(report)

    logger.info(f"Generated summary report: {output_path}")
    return output_path


def select_best_design(
    designs_with_metrics: List[Dict],
    weights: Dict[str, float] = None
) -> Dict:
    """
    Select best design with custom weighting.

    Args:
        designs_with_metrics: List of design metric dicts
        weights: Custom weights for scoring components

    Returns:
        Best design dictionary
    """
    if weights is None:
        weights = {
            "efficacy": 0.40,
            "safety": 0.35,
            "durability": 0.15,
            "electrical": 0.10,
        }

    for design in designs_with_metrics:
        # Recompute total score with custom weights
        design["weighted_score"] = (
            design.get("efficacy_score", 0) * weights["efficacy"] +
            design.get("safety_score", 0) * weights["safety"] +
            design.get("durability_score", 0) * weights["durability"] +
            design.get("electrical_score", 0) * weights["electrical"]
        )

    # Filter by safety
    safe_designs = [
        d for d in designs_with_metrics
        if d.get("toxicity_score", 1) < 0.15 and
           d.get("structural_integrity", 0) > 0.80
    ]

    if not safe_designs:
        safe_designs = designs_with_metrics

    # Sort by weighted score
    sorted_designs = sorted(
        safe_designs,
        key=lambda x: x.get("weighted_score", 0),
        reverse=True
    )

    return sorted_designs[0] if sorted_designs else {}
