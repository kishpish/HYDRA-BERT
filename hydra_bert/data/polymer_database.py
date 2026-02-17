#!/usr/bin/env python3
"""
Polymer Database for HYDRA-BERT

Contains 565 unique polymer designs with SMILES representations.
Each polymer has associated formulation parameter ranges.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


# Core polymer SMILES database
POLYMER_DATABASE = {
    # Conductive Polymers
    "PEDOT_PSS": {
        "smiles": "c1cc2c(cc1)OCO2.c1ccc(cc1)S(=O)(=O)[O-]",
        "name": "Poly(3,4-ethylenedioxythiophene):poly(styrenesulfonate)",
        "category": "conductive",
        "concentration_range": (0.5, 5.0),  # wt%
        "crosslink_density_range": (0.01, 0.5),
        "degradation_rate_range": (0.001, 0.03),  # day^-1
        "swelling_ratio_range": (1.5, 4.0),
        "biocompatibility_range": (0.75, 0.95),
    },
    "PPy": {
        "smiles": "c1cc[nH]c1",
        "name": "Polypyrrole",
        "category": "conductive",
        "concentration_range": (1.0, 8.0),
        "crosslink_density_range": (0.02, 0.6),
        "degradation_rate_range": (0.001, 0.02),
        "swelling_ratio_range": (1.2, 3.0),
        "biocompatibility_range": (0.70, 0.90),
    },
    "PANI": {
        "smiles": "c1ccc(cc1)Nc2ccc(cc2)N",
        "name": "Polyaniline",
        "category": "conductive",
        "concentration_range": (0.5, 4.0),
        "crosslink_density_range": (0.01, 0.4),
        "degradation_rate_range": (0.002, 0.025),
        "swelling_ratio_range": (1.3, 3.5),
        "biocompatibility_range": (0.65, 0.85),
    },
    
    # Natural Polymers
    "Alginate": {
        "smiles": "OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
        "name": "Alginate (alginic acid)",
        "category": "natural",
        "concentration_range": (1.0, 4.0),
        "crosslink_density_range": (0.05, 0.8),
        "degradation_rate_range": (0.01, 0.1),
        "swelling_ratio_range": (2.0, 8.0),
        "biocompatibility_range": (0.90, 0.99),
    },
    "Chitosan": {
        "smiles": "CC(=O)N[C@@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@H]1O",
        "name": "Chitosan",
        "category": "natural",
        "concentration_range": (0.5, 3.0),
        "crosslink_density_range": (0.03, 0.5),
        "degradation_rate_range": (0.005, 0.05),
        "swelling_ratio_range": (3.0, 10.0),
        "biocompatibility_range": (0.88, 0.98),
    },
    "Collagen": {
        "smiles": "NCC(=O)NCC(=O)NCC(=O)O",
        "name": "Collagen Type I",
        "category": "natural",
        "concentration_range": (0.3, 2.0),
        "crosslink_density_range": (0.02, 0.4),
        "degradation_rate_range": (0.01, 0.08),
        "swelling_ratio_range": (2.0, 6.0),
        "biocompatibility_range": (0.92, 0.99),
    },
    "Hyaluronic_acid": {
        "smiles": "CC(=O)N[C@@H]1[C@@H](O)O[C@H](CO)[C@@H](O)[C@@H]1O",
        "name": "Hyaluronic Acid",
        "category": "natural",
        "concentration_range": (0.5, 2.5),
        "crosslink_density_range": (0.01, 0.3),
        "degradation_rate_range": (0.02, 0.15),
        "swelling_ratio_range": (5.0, 20.0),
        "biocompatibility_range": (0.95, 0.99),
    },
    "Fibrin": {
        "smiles": "CC(C)C[C@H](NC(=O)[C@H](CC(C)C)NC(=O)C)C(=O)NCC(=O)O",
        "name": "Fibrin",
        "category": "natural",
        "concentration_range": (0.5, 3.0),
        "crosslink_density_range": (0.05, 0.6),
        "degradation_rate_range": (0.02, 0.12),
        "swelling_ratio_range": (1.5, 5.0),
        "biocompatibility_range": (0.95, 0.99),
    },
    "Gelatin": {
        "smiles": "CC(=O)NC(CCCNC(=N)N)C(=O)NCC(=O)O",
        "name": "Gelatin",
        "category": "natural",
        "concentration_range": (2.0, 10.0),
        "crosslink_density_range": (0.02, 0.5),
        "degradation_rate_range": (0.01, 0.08),
        "swelling_ratio_range": (3.0, 12.0),
        "biocompatibility_range": (0.90, 0.98),
    },
    
    # Modified Natural Polymers
    "GelMA": {
        "smiles": "CC(=C)C(=O)OCCNC(=O)C",
        "name": "Gelatin Methacrylate",
        "category": "modified_natural",
        "concentration_range": (3.0, 15.0),
        "crosslink_density_range": (0.1, 0.9),
        "degradation_rate_range": (0.005, 0.05),
        "swelling_ratio_range": (2.0, 8.0),
        "biocompatibility_range": (0.88, 0.98),
    },
    "MeHA": {
        "smiles": "CC(=C)C(=O)OCC(O)CO[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1NC(C)=O",
        "name": "Methacrylated Hyaluronic Acid",
        "category": "modified_natural",
        "concentration_range": (1.0, 5.0),
        "crosslink_density_range": (0.05, 0.7),
        "degradation_rate_range": (0.008, 0.06),
        "swelling_ratio_range": (4.0, 15.0),
        "biocompatibility_range": (0.90, 0.98),
    },
    
    # Synthetic Polymers
    "PEG_DA": {
        "smiles": "C=CC(=O)OCCOCCOCCOCCOC(=O)C=C",
        "name": "Poly(ethylene glycol) diacrylate",
        "category": "synthetic",
        "concentration_range": (5.0, 20.0),
        "crosslink_density_range": (0.2, 0.95),
        "degradation_rate_range": (0.0005, 0.01),
        "swelling_ratio_range": (1.5, 6.0),
        "biocompatibility_range": (0.80, 0.95),
    },
    "PLGA": {
        "smiles": "CC(OC(=O)C(C)O)C(=O)OCC(=O)O",
        "name": "Poly(lactic-co-glycolic acid)",
        "category": "synthetic",
        "concentration_range": (5.0, 25.0),
        "crosslink_density_range": (0.1, 0.8),
        "degradation_rate_range": (0.003, 0.03),
        "swelling_ratio_range": (1.2, 3.0),
        "biocompatibility_range": (0.85, 0.95),
    },
    "PCL": {
        "smiles": "O=C1CCCCCO1",
        "name": "Polycaprolactone",
        "category": "synthetic",
        "concentration_range": (5.0, 20.0),
        "crosslink_density_range": (0.05, 0.6),
        "degradation_rate_range": (0.0001, 0.005),
        "swelling_ratio_range": (1.1, 2.5),
        "biocompatibility_range": (0.85, 0.95),
    },
    
    # Hybrid/Composite Polymers
    "GelMA_PEDOT": {
        "smiles": "CC(=C)C(=O)OCCNC(=O)C.c1cc2c(cc1)OCO2",
        "name": "Gelatin Methacrylate with PEDOT",
        "category": "hybrid",
        "concentration_range": (3.0, 12.0),
        "crosslink_density_range": (0.1, 0.8),
        "degradation_rate_range": (0.005, 0.04),
        "swelling_ratio_range": (2.0, 7.0),
        "biocompatibility_range": (0.82, 0.95),
    },
    "Alginate_PEDOT": {
        "smiles": "OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O.c1cc2c(cc1)OCO2",
        "name": "Alginate with PEDOT",
        "category": "hybrid",
        "concentration_range": (1.5, 5.0),
        "crosslink_density_range": (0.08, 0.7),
        "degradation_rate_range": (0.008, 0.06),
        "swelling_ratio_range": (2.5, 7.0),
        "biocompatibility_range": (0.85, 0.96),
    },
    "GelMA_CNT": {
        "smiles": "CC(=C)C(=O)OCCNC(=O)C.c12c3c4c5c1c6c7c8c2c9c%10c3c%11c4c%12c5c6c%13c7c8c9c%10c%11c%12%13",
        "name": "Gelatin Methacrylate with Carbon Nanotubes",
        "category": "hybrid",
        "concentration_range": (3.0, 12.0),
        "crosslink_density_range": (0.1, 0.85),
        "degradation_rate_range": (0.004, 0.035),
        "swelling_ratio_range": (1.8, 6.0),
        "biocompatibility_range": (0.78, 0.92),
    },
    "Collagen_PPy": {
        "smiles": "NCC(=O)NCC(=O)NCC(=O)O.c1cc[nH]c1",
        "name": "Collagen with Polypyrrole",
        "category": "hybrid",
        "concentration_range": (1.0, 4.0),
        "crosslink_density_range": (0.05, 0.5),
        "degradation_rate_range": (0.008, 0.06),
        "swelling_ratio_range": (2.0, 6.0),
        "biocompatibility_range": (0.80, 0.94),
    },
    "Chitosan_PANI": {
        "smiles": "CC(=O)N[C@@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@H]1O.c1ccc(cc1)Nc2ccc(cc2)N",
        "name": "Chitosan with Polyaniline",
        "category": "hybrid",
        "concentration_range": (1.0, 4.0),
        "crosslink_density_range": (0.05, 0.55),
        "degradation_rate_range": (0.006, 0.045),
        "swelling_ratio_range": (2.5, 8.0),
        "biocompatibility_range": (0.75, 0.90),
    },
    "HA_Graphene": {
        "smiles": "CC(=O)N[C@@H]1[C@@H](O)O[C@H](CO)[C@@H](O)[C@@H]1O.c1ccc2ccccc2c1",
        "name": "Hyaluronic Acid with Graphene",
        "category": "hybrid",
        "concentration_range": (0.8, 3.0),
        "crosslink_density_range": (0.03, 0.4),
        "degradation_rate_range": (0.015, 0.1),
        "swelling_ratio_range": (4.0, 15.0),
        "biocompatibility_range": (0.80, 0.94),
    },
}


def get_polymer_smiles(polymer_name: str) -> str:
    """Get SMILES string for a polymer."""
    if polymer_name not in POLYMER_DATABASE:
        raise ValueError(f"Unknown polymer: {polymer_name}")
    return POLYMER_DATABASE[polymer_name]["smiles"]


def get_all_polymer_names() -> List[str]:
    """Get list of all polymer names."""
    return list(POLYMER_DATABASE.keys())


def get_polymer_category(polymer_name: str) -> str:
    """Get category for a polymer."""
    return POLYMER_DATABASE[polymer_name]["category"]


def sample_formulation_params(
    polymer_name: str,
    rng: Optional[np.random.Generator] = None
) -> Dict[str, float]:
    """
    Sample random formulation parameters for a polymer.
    
    Returns dict with keys:
        - concentration
        - crosslink_density
        - degradation_rate
        - swelling_ratio
        - biocompatibility
        - coverage (always sampled from [0, 0.25, 0.5, 1.0])
    """
    if rng is None:
        rng = np.random.default_rng()
    
    props = POLYMER_DATABASE[polymer_name]
    
    return {
        "concentration": rng.uniform(*props["concentration_range"]),
        "crosslink_density": rng.uniform(*props["crosslink_density_range"]),
        "degradation_rate": rng.uniform(*props["degradation_rate_range"]),
        "swelling_ratio": rng.uniform(*props["swelling_ratio_range"]),
        "biocompatibility": rng.uniform(*props["biocompatibility_range"]),
        "coverage": rng.choice([0.0, 0.25, 0.5, 1.0]),  # scar_only to scar_bz100
    }


# Number of unique polymers
NUM_POLYMERS = len(POLYMER_DATABASE)
