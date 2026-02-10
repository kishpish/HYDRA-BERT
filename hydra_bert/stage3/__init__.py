"""
HYDRA-BERT Stage 3: Patient-Specific Hydrogel Design Generation

This module generates, simulates, and ranks patient-specific hydrogel designs
using the trained HYDRA-BERT model and FEBio/OpenCarp cardiac simulations.

Pipeline:
1. Generate 1M+ candidate designs per patient
2. Filter using HYDRA-BERT predictions (top 10,000)
3. Run FEBio/OpenCarp simulations (top 100)
4. Calculate comprehensive metrics
5. Select optimal design per patient
"""

from .patient_config import (
    PatientConfig,
    load_patient_configs,
    REAL_PATIENTS,
    InjectionSite,
    InfarctGeometry,
    get_patient_by_id,
    get_patient_ids,
)

from .generation import (
    DesignGenerator,
    DesignCandidate,
    POLYMER_DATABASE,
    save_designs_to_csv,
    load_designs_from_csv,
)

from .simulation import (
    SimulationRunner,
    FEBioSimulator,
    OpenCarpSimulator,
    SimulationConfig,
    SimulationResult,
)

from .metrics import (
    MetricsCalculator,
    CardiacMetrics,
    SafetyMetrics,
    CompositeScore,
)

from .analysis import (
    DesignAnalyzer,
    generate_patient_report,
    generate_summary_report,
    select_best_design,
)

from .therapeutic_thresholds import (
    TherapeuticThresholds,
    TherapeuticClassifier,
    KEY_FILTERING_METRICS,
    compute_filtering_score,
)

__all__ = [
    # Patient configuration
    "PatientConfig",
    "load_patient_configs",
    "REAL_PATIENTS",
    "InjectionSite",
    "InfarctGeometry",
    "get_patient_by_id",
    "get_patient_ids",
    # Design generation
    "DesignGenerator",
    "DesignCandidate",
    "POLYMER_DATABASE",
    "save_designs_to_csv",
    "load_designs_from_csv",
    # Simulation
    "SimulationRunner",
    "FEBioSimulator",
    "OpenCarpSimulator",
    "SimulationConfig",
    "SimulationResult",
    # Metrics
    "MetricsCalculator",
    "CardiacMetrics",
    "SafetyMetrics",
    "CompositeScore",
    # Analysis
    "DesignAnalyzer",
    "generate_patient_report",
    "generate_summary_report",
    "select_best_design",
]
