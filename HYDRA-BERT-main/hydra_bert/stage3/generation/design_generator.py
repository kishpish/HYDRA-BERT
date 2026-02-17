#!/usr/bin/env python3
"""
Design Generator Module

Generates 1M+ hydrogel design candidates per patient using:
1. PPO-trained policy for formulation optimization
2. Systematic parameter space exploration
3. Polymer-specific constraint satisfaction
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json

logger = logging.getLogger(__name__)


# Valid polymer configurations
POLYMER_DATABASE = {
    "GelMA_3pct": {
        "smiles": "CC(=C)C(=O)NCCCC[C@H](NC(=O)C)C(=O)O",
        "category": "protein_modified",
        "stiffness_range": (1.0, 10.0),
        "degradation_range": (7.0, 60.0),
        "conductivity_range": (0.0, 0.0),
        "biocompatibility": 0.95,
    },
    "GelMA_5pct": {
        "smiles": "CC(=C)C(=O)NCCCC[C@H](NC(=O)C)C(=O)O",
        "category": "protein_modified",
        "stiffness_range": (2.0, 25.0),
        "degradation_range": (14.0, 90.0),
        "conductivity_range": (0.0, 0.0),
        "biocompatibility": 0.94,
    },
    "GelMA_7pct": {
        "smiles": "CC(=C)C(=O)NCCCC[C@H](NC(=O)C)C(=O)O",
        "category": "protein_modified",
        "stiffness_range": (5.0, 50.0),
        "degradation_range": (21.0, 120.0),
        "conductivity_range": (0.0, 0.0),
        "biocompatibility": 0.93,
    },
    "GelMA_10pct": {
        "smiles": "CC(=C)C(=O)NCCCC[C@H](NC(=O)C)C(=O)O",
        "category": "protein_modified",
        "stiffness_range": (10.0, 100.0),
        "degradation_range": (30.0, 180.0),
        "conductivity_range": (0.0, 0.0),
        "biocompatibility": 0.92,
    },
    "GelMA_Polypyrrole": {
        "smiles": "CC(=C)C(=O)NCCCC[C@H](NC(=O)C)C(=O)O.c1cc[nH]c1",
        "category": "conductive_hydrogel",
        "stiffness_range": (5.0, 50.0),
        "degradation_range": (14.0, 90.0),
        "conductivity_range": (0.1, 1.0),
        "biocompatibility": 0.88,
    },
    "GelMA_rGO": {
        "smiles": "CC(=C)C(=O)NCCCC[C@H](NC(=O)C)C(=O)O.O=C(O)c1cc(O)c(O)c(C(=O)O)c1",
        "category": "conductive_hydrogel",
        "stiffness_range": (10.0, 80.0),
        "degradation_range": (30.0, 120.0),
        "conductivity_range": (0.05, 0.5),
        "biocompatibility": 0.85,
    },
    "GelMA_BioIL": {
        "smiles": "CC(=C)C(=O)NCCCC[C@H](NC(=O)C)C(=O)O.[N+](C)(C)(C)CCCC",
        "category": "conductive_hydrogel",
        "stiffness_range": (3.0, 30.0),
        "degradation_range": (14.0, 60.0),
        "conductivity_range": (0.2, 0.8),
        "biocompatibility": 0.87,
    },
    "GelMA_MXene": {
        "smiles": "CC(=C)C(=O)NCCCC[C@H](NC(=O)C)C(=O)O.[Ti]",
        "category": "conductive_hydrogel",
        "stiffness_range": (20.0, 100.0),
        "degradation_range": (60.0, 180.0),
        "conductivity_range": (0.5, 1.0),
        "biocompatibility": 0.82,
    },
    "PEGDA_575": {
        "smiles": "C=CC(=O)OCCOCCOCCOC(=O)C=C",
        "category": "synthetic",
        "stiffness_range": (1.0, 30.0),
        "degradation_range": (30.0, 180.0),
        "conductivity_range": (0.0, 0.0),
        "biocompatibility": 0.90,
    },
    "PEGDA_700": {
        "smiles": "C=CC(=O)OCCOCCOCCOCCOC(=O)C=C",
        "category": "synthetic",
        "stiffness_range": (1.0, 25.0),
        "degradation_range": (30.0, 180.0),
        "conductivity_range": (0.0, 0.0),
        "biocompatibility": 0.91,
    },
    "PEGDA_3400": {
        "smiles": "C=CC(=O)OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOC(=O)C=C",
        "category": "synthetic",
        "stiffness_range": (0.5, 15.0),
        "degradation_range": (60.0, 180.0),
        "conductivity_range": (0.0, 0.0),
        "biocompatibility": 0.93,
    },
    "Alginate_CaCl2": {
        "smiles": "O[C@H]1O[C@H](C(=O)[O-])[C@@H](O)[C@H](O)[C@H]1O.[Ca+2]",
        "category": "polysaccharide",
        "stiffness_range": (1.0, 20.0),
        "degradation_range": (14.0, 90.0),
        "conductivity_range": (0.0, 0.0),
        "biocompatibility": 0.95,
    },
    "Alginate_RGD": {
        "smiles": "O[C@H]1O[C@H](C(=O)[O-])[C@@H](O)[C@H](O)[C@H]1O.NCC(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)O",
        "category": "polysaccharide",
        "stiffness_range": (2.0, 25.0),
        "degradation_range": (21.0, 120.0),
        "conductivity_range": (0.0, 0.0),
        "biocompatibility": 0.96,
    },
    "Chitosan_thermogel": {
        "smiles": "N[C@H]1[C@H](O)O[C@H](CO)[C@@H](O)[C@@H]1O",
        "category": "polysaccharide",
        "stiffness_range": (1.0, 15.0),
        "degradation_range": (14.0, 60.0),
        "conductivity_range": (0.0, 0.0),
        "biocompatibility": 0.94,
    },
    "Chitosan_EGCG": {
        "smiles": "N[C@H]1[C@H](O)O[C@H](CO)[C@@H](O)[C@@H]1O.Oc1cc(O)c2c(c1)OC(c1cc(O)c(O)c(O)c1)(O2)c1cc(O)c(O)c(O)c1",
        "category": "polysaccharide",
        "stiffness_range": (2.0, 20.0),
        "degradation_range": (21.0, 90.0),
        "conductivity_range": (0.0, 0.0),
        "biocompatibility": 0.93,
    },
    "Chitosan_HA": {
        "smiles": "N[C@H]1[C@H](O)O[C@H](CO)[C@@H](O)[C@@H]1O.CC(=O)N[C@H]1[C@H](O)O[C@H](CO)[C@@H](O)[C@@H]1O[C@H]2O[C@H](C(=O)[O-])[C@@H](O)[C@H](O)[C@H]2O",
        "category": "polysaccharide",
        "stiffness_range": (3.0, 30.0),
        "degradation_range": (30.0, 120.0),
        "conductivity_range": (0.0, 0.0),
        "biocompatibility": 0.95,
    },
    "HA_acellular": {
        "smiles": "CC(=O)N[C@H]1[C@H](O)O[C@H](CO)[C@@H](O)[C@@H]1O[C@H]2O[C@H](C(=O)[O-])[C@@H](O)[C@H](O)[C@H]2O",
        "category": "glycosaminoglycan",
        "stiffness_range": (0.5, 10.0),
        "degradation_range": (7.0, 45.0),
        "conductivity_range": (0.0, 0.0),
        "biocompatibility": 0.97,
    },
    "HA_ECM": {
        "smiles": "CC(=O)N[C@H]1[C@H](O)O[C@H](CO)[C@@H](O)[C@@H]1O[C@H]2O[C@H](C(=O)[O-])[C@@H](O)[C@H](O)[C@H]2O",
        "category": "glycosaminoglycan",
        "stiffness_range": (1.0, 20.0),
        "degradation_range": (14.0, 90.0),
        "conductivity_range": (0.0, 0.0),
        "biocompatibility": 0.98,
    },
    "MeHA_photocrosslink": {
        "smiles": "C=C(C)C(=O)OC[C@H]1O[C@@H](O[C@H]2[C@H](O)[C@@H](NC(C)=O)[C@@H](O[C@H]3O[C@H](C(=O)[O-])[C@@H](O)[C@H](O)[C@H]3O)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@@H]1O",
        "category": "glycosaminoglycan",
        "stiffness_range": (2.0, 40.0),
        "degradation_range": (21.0, 120.0),
        "conductivity_range": (0.0, 0.0),
        "biocompatibility": 0.94,
    },
    "dECM_VentriGel": {
        "smiles": "NCC(=O)N1CCC[C@H]1C(=O)N[C@@H](CCCNC(=N)N)C(=O)NCC(=O)O",
        "category": "decellularized",
        "stiffness_range": (1.0, 15.0),
        "degradation_range": (14.0, 60.0),
        "conductivity_range": (0.0, 0.0),
        "biocompatibility": 0.99,
    },
    "dECM_cardiac": {
        "smiles": "NCC(=O)N1CCC[C@H]1C(=O)NCC(=O)O",
        "category": "decellularized",
        "stiffness_range": (2.0, 20.0),
        "degradation_range": (21.0, 90.0),
        "conductivity_range": (0.0, 0.0),
        "biocompatibility": 0.98,
    },
    "Gelatin_crosslinked": {
        "smiles": "NCC(=O)N1CCC[C@H]1C(=O)NCC(=O)O.OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
        "category": "protein",
        "stiffness_range": (1.0, 30.0),
        "degradation_range": (14.0, 90.0),
        "conductivity_range": (0.0, 0.0),
        "biocompatibility": 0.94,
    },
    "Fibrin_thrombin": {
        "smiles": "NCC(=O)N[C@@H](CCCCN)C(=O)NCC(=O)O",
        "category": "protein",
        "stiffness_range": (0.5, 10.0),
        "degradation_range": (7.0, 30.0),
        "conductivity_range": (0.0, 0.0),
        "biocompatibility": 0.97,
    },
    "PEDOT_PSS": {
        "smiles": "c1sc(c2OCCO2)cc1.c1ccc(S(=O)(=O)[O-])cc1",
        "category": "conductive",
        "stiffness_range": (10.0, 100.0),
        "degradation_range": (90.0, 180.0),
        "conductivity_range": (0.5, 1.0),
        "biocompatibility": 0.80,
    },
}

# Patch coverage options
PATCH_COVERAGES = ["scar_only", "scar_bz25", "scar_bz50", "scar_bz100"]
COVERAGE_TO_INDEX = {cov: i for i, cov in enumerate(PATCH_COVERAGES)}


@dataclass
class DesignCandidate:
    """Represents a single hydrogel design candidate."""

    # Identifiers
    design_id: str = ""
    patient_id: str = ""

    # Polymer selection
    polymer_name: str = ""
    polymer_smiles: str = ""
    polymer_category: str = ""

    # Formulation parameters
    stiffness_kPa: float = 10.0
    degradation_days: float = 30.0
    conductivity_S_m: float = 0.0
    patch_thickness_mm: float = 1.5
    patch_coverage: str = "scar_bz50"

    # Model predictions (from HYDRA-BERT)
    predicted_gcs: float = 0.0
    predicted_delta_ef: float = 0.0
    predicted_optimal_prob: float = 0.0
    predicted_toxicity: float = 0.0
    predicted_integrity: float = 1.0
    predicted_fibrosis_risk: float = 0.0

    # Reward from PPO
    reward: float = 0.0

    # Simulation results (filled after FEBio/OpenCarp)
    simulated: bool = False
    sim_delta_ef: float = 0.0
    sim_stress_reduction: float = 0.0
    sim_strain_normalization: float = 0.0
    sim_wall_stress_kPa: float = 0.0
    sim_ejection_work_J: float = 0.0

    # Computed metrics
    composite_score: float = 0.0
    rank: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "design_id": self.design_id,
            "patient_id": self.patient_id,
            "polymer_name": self.polymer_name,
            "polymer_smiles": self.polymer_smiles,
            "polymer_category": self.polymer_category,
            "stiffness_kPa": self.stiffness_kPa,
            "degradation_days": self.degradation_days,
            "conductivity_S_m": self.conductivity_S_m,
            "patch_thickness_mm": self.patch_thickness_mm,
            "patch_coverage": self.patch_coverage,
            "predicted_gcs": self.predicted_gcs,
            "predicted_delta_ef": self.predicted_delta_ef,
            "predicted_optimal_prob": self.predicted_optimal_prob,
            "predicted_toxicity": self.predicted_toxicity,
            "predicted_integrity": self.predicted_integrity,
            "predicted_fibrosis_risk": self.predicted_fibrosis_risk,
            "reward": self.reward,
            "simulated": self.simulated,
            "sim_delta_ef": self.sim_delta_ef,
            "sim_stress_reduction": self.sim_stress_reduction,
            "sim_strain_normalization": self.sim_strain_normalization,
            "sim_wall_stress_kPa": self.sim_wall_stress_kPa,
            "sim_ejection_work_J": self.sim_ejection_work_J,
            "composite_score": self.composite_score,
            "rank": self.rank,
        }

    def get_formulation_vector(self) -> np.ndarray:
        """Get normalized formulation vector for model input."""
        return np.array([
            self.stiffness_kPa / 100.0,
            self.degradation_days / 180.0,
            self.conductivity_S_m,
            self.patch_thickness_mm / 5.0,
            float(COVERAGE_TO_INDEX.get(self.patch_coverage, 2)) / 3.0,
            1.0,  # Literature validated placeholder
        ], dtype=np.float32)


class DesignGenerator:
    """
    Generates hydrogel design candidates for patient-specific optimization.

    Uses a combination of:
    - Systematic grid sampling over parameter space
    - PPO-trained policy for intelligent sampling
    - Constraint satisfaction for valid designs
    """

    def __init__(
        self,
        model=None,
        ppo_policy=None,
        num_designs_per_patient: int = 1_000_000,
        batch_size: int = 10000,
        device: str = "cuda",
        seed: int = 42,
    ):
        self.model = model
        self.ppo_policy = ppo_policy
        self.num_designs = num_designs_per_patient
        self.batch_size = batch_size
        self.device = device
        self.rng = np.random.RandomState(seed)

        # Parameter ranges for systematic sampling
        self.stiffness_values = np.linspace(1.0, 100.0, 50)
        self.degradation_values = np.linspace(7.0, 180.0, 30)
        self.thickness_values = np.linspace(0.5, 5.0, 20)

    def generate_designs(
        self,
        patient_config,
        use_ppo: bool = True,
        progress_callback=None
    ) -> List[DesignCandidate]:
        """
        Generate design candidates for a specific patient.

        Args:
            patient_config: PatientConfig object
            use_ppo: Whether to use PPO policy for sampling
            progress_callback: Optional callback for progress updates

        Returns:
            List of DesignCandidate objects
        """
        logger.info(f"Generating {self.num_designs:,} designs for {patient_config.patient_id}")

        designs = []
        design_counter = 0

        # Phase 1: Systematic grid sampling (~20% of designs)
        systematic_count = int(self.num_designs * 0.2)
        systematic_designs = self._generate_systematic(
            patient_config, systematic_count, design_counter
        )
        designs.extend(systematic_designs)
        design_counter += len(systematic_designs)

        if progress_callback:
            progress_callback(len(designs) / self.num_designs)

        # Phase 2: PPO-guided sampling (~60% of designs)
        if use_ppo and self.ppo_policy is not None:
            ppo_count = int(self.num_designs * 0.6)
            ppo_designs = self._generate_ppo_guided(
                patient_config, ppo_count, design_counter
            )
            designs.extend(ppo_designs)
            design_counter += len(ppo_designs)

            if progress_callback:
                progress_callback(len(designs) / self.num_designs)

        # Phase 3: Random exploration (~20% of designs)
        random_count = self.num_designs - len(designs)
        random_designs = self._generate_random(
            patient_config, random_count, design_counter
        )
        designs.extend(random_designs)

        if progress_callback:
            progress_callback(1.0)

        logger.info(f"Generated {len(designs):,} designs for {patient_config.patient_id}")
        return designs

    def _generate_systematic(
        self,
        patient_config,
        count: int,
        start_id: int
    ) -> List[DesignCandidate]:
        """Generate designs via systematic grid sampling."""
        designs = []

        polymers = list(POLYMER_DATABASE.keys())
        generated = 0

        for polymer_name in polymers:
            polymer_info = POLYMER_DATABASE[polymer_name]

            # Get valid ranges for this polymer
            stiff_min, stiff_max = polymer_info["stiffness_range"]
            deg_min, deg_max = polymer_info["degradation_range"]
            cond_min, cond_max = polymer_info["conductivity_range"]

            # Sample within polymer-specific ranges
            stiff_vals = np.linspace(stiff_min, stiff_max, 10)
            deg_vals = np.linspace(deg_min, deg_max, 8)

            for stiff in stiff_vals:
                for deg in deg_vals:
                    for thick in [1.0, 1.5, 2.0, 2.5]:
                        for coverage in PATCH_COVERAGES:
                            if generated >= count:
                                return designs

                            # Sample conductivity if applicable
                            cond = self.rng.uniform(cond_min, cond_max)

                            design = DesignCandidate(
                                design_id=f"{patient_config.patient_id}_sys_{start_id + generated}",
                                patient_id=patient_config.patient_id,
                                polymer_name=polymer_name,
                                polymer_smiles=polymer_info["smiles"],
                                polymer_category=polymer_info["category"],
                                stiffness_kPa=stiff,
                                degradation_days=deg,
                                conductivity_S_m=cond,
                                patch_thickness_mm=thick,
                                patch_coverage=coverage,
                            )
                            designs.append(design)
                            generated += 1

        return designs

    def _generate_ppo_guided(
        self,
        patient_config,
        count: int,
        start_id: int
    ) -> List[DesignCandidate]:
        """Generate designs using PPO policy."""
        designs = []

        if self.ppo_policy is None:
            # Fallback to smart random sampling
            return self._generate_smart_random(patient_config, count, start_id)

        patient_features = torch.tensor(
            patient_config.get_patient_features(),
            device=self.device
        ).unsqueeze(0)

        generated = 0
        batch_num = 0

        while generated < count:
            batch_size = min(self.batch_size, count - generated)

            # Expand patient features for batch
            patient_batch = patient_features.expand(batch_size, -1)

            # Sample from PPO policy
            with torch.no_grad():
                actions, _ = self.ppo_policy.sample_action(patient_batch)

            # Convert actions to design parameters
            for i in range(batch_size):
                action = actions[i].cpu().numpy()

                # Decode action to formulation parameters
                polymer_idx = int(action[0] * len(POLYMER_DATABASE)) % len(POLYMER_DATABASE)
                polymer_name = list(POLYMER_DATABASE.keys())[polymer_idx]
                polymer_info = POLYMER_DATABASE[polymer_name]

                stiff_min, stiff_max = polymer_info["stiffness_range"]
                deg_min, deg_max = polymer_info["degradation_range"]
                cond_min, cond_max = polymer_info["conductivity_range"]

                design = DesignCandidate(
                    design_id=f"{patient_config.patient_id}_ppo_{start_id + generated}",
                    patient_id=patient_config.patient_id,
                    polymer_name=polymer_name,
                    polymer_smiles=polymer_info["smiles"],
                    polymer_category=polymer_info["category"],
                    stiffness_kPa=stiff_min + action[1] * (stiff_max - stiff_min),
                    degradation_days=deg_min + action[2] * (deg_max - deg_min),
                    conductivity_S_m=cond_min + action[3] * (cond_max - cond_min),
                    patch_thickness_mm=0.5 + action[4] * 4.5,
                    patch_coverage=PATCH_COVERAGES[int(action[5] * 4) % 4],
                )
                designs.append(design)
                generated += 1

            # Clear GPU cache periodically
            if torch.cuda.is_available() and batch_num % 10 == 0:
                torch.cuda.empty_cache()

            batch_num += 1

        return designs

    def _generate_smart_random(
        self,
        patient_config,
        count: int,
        start_id: int
    ) -> List[DesignCandidate]:
        """Smart random sampling with patient-aware biases."""
        designs = []

        # Weight polymers by biocompatibility and relevance
        polymer_weights = []
        for name, info in POLYMER_DATABASE.items():
            weight = info["biocompatibility"]
            # Boost conductive hydrogels for larger infarcts
            if patient_config.scar_fraction_pct > 8.0:
                if "conductive" in info["category"]:
                    weight *= 1.5
            polymer_weights.append(weight)

        polymer_weights = np.array(polymer_weights)
        polymer_weights /= polymer_weights.sum()
        polymer_names = list(POLYMER_DATABASE.keys())

        for i in range(count):
            # Sample polymer weighted by biocompatibility
            polymer_name = self.rng.choice(polymer_names, p=polymer_weights)
            polymer_info = POLYMER_DATABASE[polymer_name]

            stiff_min, stiff_max = polymer_info["stiffness_range"]
            deg_min, deg_max = polymer_info["degradation_range"]
            cond_min, cond_max = polymer_info["conductivity_range"]

            # Bias stiffness based on wall thickness
            target_stiffness = patient_config.wall_thickness_mm * 0.8
            stiff = np.clip(
                self.rng.normal(target_stiffness, 10.0),
                stiff_min, stiff_max
            )

            # Bias degradation based on scar size
            target_degradation = 30 + patient_config.scar_fraction_pct * 5
            deg = np.clip(
                self.rng.normal(target_degradation, 20.0),
                deg_min, deg_max
            )

            # Sample other parameters
            cond = self.rng.uniform(cond_min, cond_max)
            thick = self.rng.uniform(0.5, 3.5)
            coverage = self.rng.choice(PATCH_COVERAGES)

            design = DesignCandidate(
                design_id=f"{patient_config.patient_id}_smart_{start_id + i}",
                patient_id=patient_config.patient_id,
                polymer_name=polymer_name,
                polymer_smiles=polymer_info["smiles"],
                polymer_category=polymer_info["category"],
                stiffness_kPa=stiff,
                degradation_days=deg,
                conductivity_S_m=cond,
                patch_thickness_mm=thick,
                patch_coverage=coverage,
            )
            designs.append(design)

        return designs

    def _generate_random(
        self,
        patient_config,
        count: int,
        start_id: int
    ) -> List[DesignCandidate]:
        """Pure random sampling for exploration."""
        designs = []
        polymer_names = list(POLYMER_DATABASE.keys())

        for i in range(count):
            polymer_name = self.rng.choice(polymer_names)
            polymer_info = POLYMER_DATABASE[polymer_name]

            stiff_min, stiff_max = polymer_info["stiffness_range"]
            deg_min, deg_max = polymer_info["degradation_range"]
            cond_min, cond_max = polymer_info["conductivity_range"]

            design = DesignCandidate(
                design_id=f"{patient_config.patient_id}_rand_{start_id + i}",
                patient_id=patient_config.patient_id,
                polymer_name=polymer_name,
                polymer_smiles=polymer_info["smiles"],
                polymer_category=polymer_info["category"],
                stiffness_kPa=self.rng.uniform(stiff_min, stiff_max),
                degradation_days=self.rng.uniform(deg_min, deg_max),
                conductivity_S_m=self.rng.uniform(cond_min, cond_max),
                patch_thickness_mm=self.rng.uniform(0.5, 5.0),
                patch_coverage=self.rng.choice(PATCH_COVERAGES),
            )
            designs.append(design)

        return designs

    def score_designs_batch(
        self,
        designs: List[DesignCandidate],
        patient_config,
    ) -> List[DesignCandidate]:
        """Score designs using HYDRA-BERT model in batches."""
        if self.model is None:
            logger.warning("No model provided, skipping scoring")
            return designs

        patient_features = patient_config.get_patient_features()

        for batch_start in range(0, len(designs), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(designs))
            batch = designs[batch_start:batch_end]

            # Prepare batch inputs
            smiles_list = [d.polymer_smiles for d in batch]
            formulation_batch = np.stack([d.get_formulation_vector() for d in batch])
            patient_batch = np.tile(patient_features, (len(batch), 1))

            # Convert to tensors
            formulation_tensor = torch.tensor(
                formulation_batch, device=self.device, dtype=torch.float32
            )
            patient_tensor = torch.tensor(
                patient_batch, device=self.device, dtype=torch.float32
            )

            # Run model inference
            with torch.no_grad():
                predictions = self.model.forward_from_smiles(
                    smiles_list=smiles_list,
                    formulation_params=formulation_tensor,
                    patient_features=patient_tensor
                )

            # Update designs with predictions
            for i, design in enumerate(batch):
                design.predicted_gcs = predictions["gcs_prediction"][i].item()
                design.predicted_delta_ef = design.predicted_gcs * 1.5  # Approximate
                design.predicted_optimal_prob = predictions["outcome_probs"][i, 2].item()
                design.predicted_toxicity = predictions["toxicity"][i].item()
                design.predicted_integrity = predictions["structural_integrity"][i].item()
                design.predicted_fibrosis_risk = predictions["fibrosis_risk"][i].item()

                # Compute reward
                design.reward = (
                    design.predicted_gcs +
                    design.predicted_optimal_prob * 10 -
                    design.predicted_toxicity * 20 -
                    (1 - design.predicted_integrity) * 20
                )

            # Clear GPU cache to avoid memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return designs


def save_designs_to_csv(designs: List[DesignCandidate], filepath: str):
    """Save designs to CSV file."""
    import pandas as pd

    data = [d.to_dict() for d in designs]
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(designs)} designs to {filepath}")


def load_designs_from_csv(filepath: str) -> List[DesignCandidate]:
    """Load designs from CSV file."""
    import pandas as pd

    df = pd.read_csv(filepath)
    designs = []

    for _, row in df.iterrows():
        design = DesignCandidate(
            design_id=row["design_id"],
            patient_id=row["patient_id"],
            polymer_name=row["polymer_name"],
            polymer_smiles=row["polymer_smiles"],
            polymer_category=row["polymer_category"],
            stiffness_kPa=row["stiffness_kPa"],
            degradation_days=row["degradation_days"],
            conductivity_S_m=row["conductivity_S_m"],
            patch_thickness_mm=row["patch_thickness_mm"],
            patch_coverage=row["patch_coverage"],
            predicted_gcs=row.get("predicted_gcs", 0.0),
            predicted_delta_ef=row.get("predicted_delta_ef", 0.0),
            predicted_optimal_prob=row.get("predicted_optimal_prob", 0.0),
            predicted_toxicity=row.get("predicted_toxicity", 0.0),
            predicted_integrity=row.get("predicted_integrity", 1.0),
            reward=row.get("reward", 0.0),
            simulated=row.get("simulated", False),
        )
        designs.append(design)

    return designs
