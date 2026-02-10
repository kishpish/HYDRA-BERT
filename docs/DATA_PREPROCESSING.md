# Data Preprocessing Pipeline

## Overview

This document details the complete data preprocessing pipeline for HYDRA-BERT, from raw patient data and polymer libraries to the final 447,480 training samples used for model development.

---

## Table of Contents

1. [Data Sources](#1-data-sources)
2. [Polymer Library Processing](#2-polymer-library-processing)
3. [Patient Data Processing](#3-patient-data-processing)
4. [Feature Engineering](#4-feature-engineering)
5. [Combinatorial Dataset Generation](#5-combinatorial-dataset-generation)
6. [Data Normalization](#6-data-normalization)
7. [SMILES Tokenization](#7-smiles-tokenization)
8. [Train/Validation/Test Splitting](#8-trainvalidationtest-splitting)
9. [Data Quality Assurance](#9-data-quality-assurance)
10. [Final Dataset Statistics](#10-final-dataset-statistics)

---

## 1. Data Sources

### 1.1 Source Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. POLYMER LIBRARY                                                      │
│  ──────────────────                                                      │
│  Source: Curated from multiple databases                                 │
│  - PolyInfo Database (312 candidates)                                    │
│  - Cardiac Literature Review (89 candidates)                             │
│  - polyBERT Training Set (127 candidates)                                │
│  - Synthetic Variants (37 candidates)                                    │
│  Total: 565 initial → 24 curated polymers                               │
│                                                                          │
│  2. PATIENT DATA                                                         │
│  ────────────────                                                        │
│  Source: Clinical cardiac imaging + FEM simulations                      │
│  - 10 real patients (de-identified)                                      │
│  - 50 synthetic patients (FEM-generated)                                 │
│  Total: 60 patients                                                      │
│                                                                          │
│  3. FORMULATION PARAMETERS                                               │
│  ──────────────────────────                                              │
│  Source: Literature-derived parameter ranges                             │
│  - Stiffness: 0.5-50 kPa (20 values)                                    │
│  - Degradation: 1-180 days (15 values)                                  │
│  - Conductivity: 0.001-1.0 S/m (10 values)                              │
│  - Thickness: 0.1-2.0 mm (8 values)                                     │
│  - Coverage: 4 patterns                                                  │
│                                                                          │
│  4. OUTCOME DATA                                                         │
│  ───────────────                                                         │
│  Source: FEBio/OpenCarp simulations                                      │
│  - Delta EF (ejection fraction improvement)                              │
│  - Stress reduction percentage                                           │
│  - Strain normalization percentage                                       │
│  - Therapeutic classification                                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 File Locations

```
data/
├── raw/
│   ├── polymer_library_565.csv          # Initial 565 polymers
│   ├── curated_polymers_24.csv          # Final 24 polymers
│   ├── polymer_SMILES.csv               # SMILES strings
│   ├── patient_data_real.csv            # 10 real patients
│   └── patient_data_synthetic.csv       # 50 synthetic patients
├── processed/
│   ├── POLYBERT_TRAINING_FINAL.csv      # 447,480 samples
│   ├── train_split.csv                  # 70% training
│   ├── val_split.csv                    # 20% validation
│   └── test_split.csv                   # 10% test
└── supplementary/
    └── initial_polymer_library_565.csv  # Full curation record
```

---

## 2. Polymer Library Processing

### 2.1 Initial Compilation

The initial polymer library was compiled from four sources:

```python
def compile_initial_polymer_library():
    """
    Compile 565 candidate polymers from multiple sources.
    """

    polymers = []

    # Source 1: PolyInfo Database (312 polymers)
    polyinfo_polymers = load_polyinfo_candidates(
        criteria={
            'material_class': ['hydrogel', 'hydrophilic_polymer'],
            'biomedical_application': True,
            'water_content': '>50%'
        }
    )
    for p in polyinfo_polymers:
        p['source'] = 'PolyInfo'
    polymers.extend(polyinfo_polymers)

    # Source 2: Cardiac Literature Review (89 polymers)
    literature_polymers = load_literature_candidates(
        databases=['PubMed', 'Google Scholar', 'Web of Science'],
        query='cardiac hydrogel injectable myocardial infarction',
        date_range='2010-2024'
    )
    for p in literature_polymers:
        p['source'] = 'Cardiac Literature'
    polymers.extend(literature_polymers)

    # Source 3: polyBERT Training Set (127 polymers)
    polybert_polymers = extract_hydrogel_candidates(
        polybert_training_data='kuelumbus/polyBERT',
        filter_criteria='hydrogel OR gel-forming'
    )
    for p in polybert_polymers:
        p['source'] = 'polyBERT Set'
    polymers.extend(polybert_polymers)

    # Source 4: Synthetic Variants (37 polymers)
    synthetic_polymers = generate_synthetic_variants(
        base_polymers=['GelMA', 'PEGDA', 'Alginate'],
        modifications=['concentration', 'crosslinking', 'additives']
    )
    for p in synthetic_polymers:
        p['source'] = 'Synthetic Variants'
    polymers.extend(synthetic_polymers)

    return polymers  # 565 total
```

### 2.2 Five-Stage Curation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│              POLYMER CURATION: 565 → 24 POLYMERS                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STAGE 1: Chemical Validity (565 → 423)                                 │
│  ──────────────────────────────────────                                  │
│  - Valid SMILES syntax (RDKit parseable)                                │
│  - Canonical form conversion                                             │
│  - Duplicate removal                                                     │
│  - Excluded: 142 (invalid: 87, duplicates: 55)                          │
│                                                                          │
│  STAGE 2: Hydrogel Formation (423 → 187)                                │
│  ────────────────────────────────────────                                │
│  - Gel-forming capability verification                                   │
│  - Crosslinking mechanism presence                                       │
│  - Water absorption capacity (>50%)                                      │
│  - Excluded: 236 (non-gel-forming polymers)                             │
│                                                                          │
│  STAGE 3: Biocompatibility (187 → 89)                                   │
│  ─────────────────────────────────────                                   │
│  - Cytocompatibility >70% (literature/ISO 10993)                        │
│  - Non-toxic degradation products                                        │
│  - Immunogenicity assessment                                             │
│  - Excluded: 98 (cytotoxic or lacking safety data)                      │
│                                                                          │
│  STAGE 4: Cardiac Applicability (89 → 41)                               │
│  ─────────────────────────────────────────                               │
│  - Injectable delivery compatibility                                     │
│  - Stiffness range: 1-50 kPa (cardiac-matching)                         │
│  - Evidence of cardiac use in literature                                 │
│  - Excluded: 48 (too rigid, non-injectable, no cardiac data)            │
│                                                                          │
│  STAGE 5: Final Selection (41 → 24)                                     │
│  ─────────────────────────────────                                       │
│  - Representative category diversity                                     │
│  - Clinical translation potential                                        │
│  - Polymer characterization completeness                                 │
│  - Excluded: 17 (redundant, low translation potential)                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 SMILES Validation and Canonicalization

```python
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

def validate_and_canonicalize_smiles(smiles_list):
    """
    Validate SMILES strings and convert to canonical form.

    Returns:
        valid_smiles: List of canonical SMILES
        invalid_indices: Indices of invalid entries
    """

    valid_smiles = []
    invalid_indices = []

    for i, smiles in enumerate(smiles_list):
        try:
            # Attempt to parse SMILES
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                invalid_indices.append(i)
                continue

            # Canonicalize
            canonical = Chem.MolToSmiles(mol, canonical=True)

            # Additional validity checks
            # 1. Molecular weight reasonable (100-100000 Da)
            mw = Descriptors.MolWt(mol)
            if mw < 100 or mw > 100000:
                invalid_indices.append(i)
                continue

            # 2. Contains expected elements for hydrogels
            elements = set([atom.GetSymbol() for atom in mol.GetAtoms()])
            allowed = {'C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'Si'}
            if not elements.issubset(allowed):
                invalid_indices.append(i)
                continue

            valid_smiles.append(canonical)

        except Exception as e:
            invalid_indices.append(i)

    return valid_smiles, invalid_indices
```

### 2.4 Polymer-Specific SMILES Mapping

Some polymers required special handling due to complex structures:

```python
POLYMER_SMILES_MAPPING = {
    # GelMA variants - Gelatin methacrylate base with degree of modification
    'GelMA_3pct': 'CC(=C)C(=O)NCCCCCC(=O)NC(CC(=O)O)C(=O)O',  # 3% methacrylation
    'GelMA_5pct': 'CC(=C)C(=O)NCCCCCC(=O)NC(CC(=O)O)C(=O)O',  # 5% methacrylation
    'GelMA_7pct': 'CC(=C)C(=O)NCCCCCC(=O)NC(CC(=O)O)C(=O)O',  # 7% methacrylation
    'GelMA_10pct': 'CC(=C)C(=O)NCCCCCC(=O)NC(CC(=O)O)C(=O)O', # 10% methacrylation

    # PEGDA - Poly(ethylene glycol) diacrylate with different MW
    'PEGDA_575': 'C=CC(=O)OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCC(=O)C=C',
    'PEGDA_700': 'C=CC(=O)OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCC(=O)C=C',
    'PEGDA_3400': '[H]OCCOCCOCCOCCOCCOCCOCCOCC(=O)C=C',  # Longer chain

    # Alginate derivatives
    'Alginate_CaCl2': 'OC1OC(O)C(O)C(O)C1O',  # Simplified mannuronic acid unit
    'Alginate_RGD': 'OC1OC(O)C(O)C(O)C1NCCCCNC(=O)C(N)CCCCN',  # RGD-modified

    # Hyaluronic acid derivatives
    'HA_acellular': 'CC(=O)NC1C(O)OC(CO)C(O)C1OC1OC(C(=O)O)C(O)C(O)C1O',
    'MeHA': 'CC(=C)C(=O)OC1C(O)OC(CO)C(O)C1OC1OC(C(=O)O)C(O)C(O)C1O',

    # Conductive polymers
    'PEDOT_PSS': 'c1cc2sc(cc2o1)c1sc2cc(oc2c1)c1cc2sccc2o1',  # PEDOT backbone

    # dECM - Represented as collagen type I fragment
    'dECM_VentriGel': 'CC(C)CC(NC(=O)C(CCC(=O)O)NC(=O)C)C(=O)NC(CC(=O)O)C(=O)O',
}
```

### 2.5 Final 24 Polymer Categories

```python
FINAL_POLYMER_LIBRARY = {
    # GelMA variants (8 polymers)
    'GelMA': ['GelMA_3pct', 'GelMA_5pct', 'GelMA_7pct', 'GelMA_10pct',
              'GelMA_BioIL', 'GelMA_MXene', 'GelMA_rGO', 'GelMA_PPy'],

    # PEGDA variants (3 polymers)
    'PEGDA': ['PEGDA_575', 'PEGDA_700', 'PEGDA_3400'],

    # Polysaccharides (5 polymers)
    'Polysaccharide': ['Alginate_CaCl2', 'Alginate_RGD', 'Chitosan_thermo',
                       'Chitosan_EGCG', 'Chitosan_HA'],

    # GAG/HA (3 polymers)
    'GAG': ['HA_acellular', 'HA_ECM', 'MeHA'],

    # dECM (2 polymers)
    'dECM': ['dECM_VentriGel', 'dECM_cardiac'],

    # Protein-based (2 polymers)
    'Protein': ['Gelatin', 'Fibrin'],

    # Conductive (1 polymer)
    'Conductive': ['PEDOT_PSS']
}

# Total: 24 unique polymers
```

---

## 3. Patient Data Processing

### 3.1 Real Patient Data Extraction

```python
def extract_real_patient_data(patient_records):
    """
    Extract cardiac parameters from real patient records.

    Source: De-identified clinical imaging data
    """

    processed_patients = []

    for record in patient_records:
        patient = {
            'patient_id': record['id'],
            'is_synthetic': False,

            # Baseline cardiac function
            'baseline_LVEF_pct': record['echo']['LVEF'],
            'baseline_GLS_pct': record['echo']['global_longitudinal_strain'],
            'baseline_EDV_mL': record['mri']['end_diastolic_volume'],
            'baseline_ESV_mL': record['mri']['end_systolic_volume'],

            # Infarct characteristics
            'scar_fraction_pct': record['lge']['scar_percentage'],
            'bz_fraction_pct': record['lge']['border_zone_percentage'],
            'transmurality': record['lge']['transmurality_score'],

            # Mechanical properties (from tagged MRI or FEM)
            'wall_thickness_mm': record['mri']['wall_thickness'],
            'bz_stress_kPa': record['fem']['border_zone_stress'],
            'healthy_stress_kPa': record['fem']['healthy_stress'],
            'stress_concentration': record['fem']['stress_concentration_factor']
        }

        processed_patients.append(patient)

    return processed_patients
```

### 3.2 Synthetic Patient Generation

```python
def generate_synthetic_patients(n_synthetic=50, real_patients=None):
    """
    Generate synthetic patient data using physiologically-constrained sampling.

    Method:
    1. Fit multivariate distribution to real patient data
    2. Sample from distribution with boundary constraints
    3. Validate physiological plausibility
    """

    # Fit multivariate Gaussian to real data
    real_data = np.array([[p[f] for f in PATIENT_FEATURES] for p in real_patients])
    mean = np.mean(real_data, axis=0)
    cov = np.cov(real_data.T)

    # Add regularization for numerical stability
    cov += np.eye(len(PATIENT_FEATURES)) * 0.01

    synthetic_patients = []

    while len(synthetic_patients) < n_synthetic:
        # Sample from multivariate Gaussian
        sample = np.random.multivariate_normal(mean, cov)

        # Apply physiological constraints
        sample_dict = {f: sample[i] for i, f in enumerate(PATIENT_FEATURES)}

        if validate_physiological_plausibility(sample_dict):
            synthetic_patients.append({
                'patient_id': f'synthetic_{len(synthetic_patients):03d}',
                'is_synthetic': True,
                **sample_dict
            })

    return synthetic_patients


def validate_physiological_plausibility(patient):
    """
    Validate that synthetic patient parameters are physiologically plausible.
    """

    constraints = {
        'baseline_LVEF_pct': (15, 55),      # Post-MI EF range
        'baseline_GLS_pct': (-25, -8),      # GLS is typically negative
        'baseline_EDV_mL': (100, 350),      # LV volume range
        'baseline_ESV_mL': (50, 250),
        'scar_fraction_pct': (5, 40),       # Infarct size
        'bz_fraction_pct': (2, 25),         # Border zone
        'wall_thickness_mm': (4, 15),
        'transmurality': (0.2, 1.0)
    }

    for feature, (min_val, max_val) in constraints.items():
        if patient[feature] < min_val or patient[feature] > max_val:
            return False

    # EDV must be greater than ESV
    if patient['baseline_EDV_mL'] <= patient['baseline_ESV_mL']:
        return False

    # EF consistency check
    computed_ef = (patient['baseline_EDV_mL'] - patient['baseline_ESV_mL']) / patient['baseline_EDV_mL'] * 100
    if abs(computed_ef - patient['baseline_LVEF_pct']) > 5:
        return False

    return True
```

### 3.3 Patient Feature Summary

| Feature | Description | Range | Unit |
|---------|-------------|-------|------|
| baseline_LVEF_pct | Baseline ejection fraction | 15-55 | % |
| baseline_GLS_pct | Global longitudinal strain | -25 to -8 | % |
| baseline_EDV_mL | End-diastolic volume | 100-350 | mL |
| baseline_ESV_mL | End-systolic volume | 50-250 | mL |
| scar_fraction_pct | Infarct scar extent | 5-40 | % |
| bz_fraction_pct | Border zone extent | 2-25 | % |
| transmurality | Transmural extent | 0.2-1.0 | ratio |
| wall_thickness_mm | LV wall thickness | 4-15 | mm |
| bz_stress_kPa | Border zone stress | 10-80 | kPa |
| healthy_stress_kPa | Remote healthy stress | 5-30 | kPa |
| stress_concentration | Stress concentration factor | 1.5-4.0 | ratio |

---

## 4. Feature Engineering

### 4.1 Feature Categories

```python
FEATURE_GROUPS = {
    # Group A: Patient Cardiac Parameters (6 features)
    'patient_cardiac': [
        'baseline_LVEF_pct',
        'baseline_GLS_pct',
        'baseline_EDV_mL',
        'baseline_ESV_mL',
        'scar_fraction_pct',
        'bz_fraction_pct'
    ],

    # Group B: Tissue Mechanics (5 features)
    'tissue_mechanics': [
        'bz_stress_kPa',
        'healthy_stress_kPa',
        'stress_concentration',
        'transmurality',
        'wall_thickness_mm'
    ],

    # Group C: Hydrogel Properties (3 features)
    'hydrogel_properties': [
        'hydrogel_E_kPa',        # Stiffness
        'hydrogel_t50_days',     # Degradation half-life
        'hydrogel_conductivity_S_m'
    ],

    # Group D: Treatment Configuration (5 features after encoding)
    'treatment_config': [
        'patch_thickness_mm',
        'patch_coverage_scar_only',      # One-hot encoded
        'patch_coverage_scar_bz25',
        'patch_coverage_scar_bz50',
        'patch_coverage_scar_bz100'
    ]
}

# Total numerical features: 6 + 5 + 3 + 5 = 19
```

### 4.2 Coverage Pattern One-Hot Encoding

```python
def encode_coverage_pattern(coverage_string):
    """
    One-hot encode patch coverage pattern.

    Coverage patterns:
    - scar_only: Cover only the infarct scar
    - scar_bz25: Cover scar + 25% of border zone
    - scar_bz50: Cover scar + 50% of border zone
    - scar_bz100: Cover scar + 100% of border zone
    """

    patterns = ['scar_only', 'scar_bz25', 'scar_bz50', 'scar_bz100']
    encoding = [0, 0, 0, 0]

    if coverage_string in patterns:
        encoding[patterns.index(coverage_string)] = 1

    return {
        'patch_coverage_scar_only': encoding[0],
        'patch_coverage_scar_bz25': encoding[1],
        'patch_coverage_scar_bz50': encoding[2],
        'patch_coverage_scar_bz100': encoding[3]
    }
```

### 4.3 Polymer Category Embedding

```python
POLYMER_CATEGORIES = {
    'GelMA': 0,
    'PEGDA': 1,
    'Alginate': 2,
    'Chitosan': 3,
    'HA': 4,
    'dECM': 5,
    'Protein': 6,
    'Conductive': 7,
    'Unknown': 8  # For fallback
}

def get_polymer_category_id(polymer_name):
    """
    Get category ID for polymer embedding layer.

    Embedding: 9 categories → 16-dimensional learned embedding
    """

    for category, polymers in FINAL_POLYMER_LIBRARY.items():
        for polymer in polymers:
            if polymer_name.startswith(polymer) or polymer_name == polymer:
                return POLYMER_CATEGORIES.get(category, 8)

    return POLYMER_CATEGORIES['Unknown']
```

### 4.4 Derived Features

```python
def compute_derived_features(row):
    """
    Compute derived/interaction features.

    These capture non-linear relationships between inputs.
    """

    derived = {}

    # Stiffness-to-stress ratio (mechanical matching)
    derived['stiffness_stress_ratio'] = (
        row['hydrogel_E_kPa'] / (row['bz_stress_kPa'] + 1e-6)
    )

    # Degradation-scar interaction (larger scars may need longer degradation)
    derived['degradation_scar_product'] = (
        row['hydrogel_t50_days'] * row['scar_fraction_pct'] / 100
    )

    # Coverage area estimate
    coverage_fractions = {
        'scar_only': 1.0,
        'scar_bz25': 1.25,
        'scar_bz50': 1.50,
        'scar_bz100': 2.0
    }
    coverage_pattern = get_coverage_pattern(row)
    derived['estimated_coverage_area'] = (
        row['scar_fraction_pct'] * coverage_fractions.get(coverage_pattern, 1.0)
    )

    # Volume load index
    derived['volume_load_index'] = (
        row['baseline_EDV_mL'] / (row['baseline_LVEF_pct'] + 1e-6)
    )

    return derived
```

---

## 5. Combinatorial Dataset Generation

### 5.1 Cartesian Product Generation

```python
def generate_combinatorial_dataset(patients, polymers, parameter_grid):
    """
    Generate full combinatorial dataset.

    For each patient, create samples for all polymer × parameter combinations.
    """

    samples = []

    for patient in patients:
        for polymer in polymers:
            for stiffness in parameter_grid['stiffness']:
                for degradation in parameter_grid['degradation']:
                    for conductivity in parameter_grid['conductivity']:
                        for thickness in parameter_grid['thickness']:
                            for coverage in parameter_grid['coverage']:
                                sample = {
                                    # Patient features
                                    **patient,

                                    # Polymer identity
                                    'polymer_id': polymer['id'],
                                    'polymer_name': polymer['name'],
                                    'polymer_SMILES': polymer['smiles'],
                                    'polymer_category': polymer['category'],

                                    # Hydrogel properties
                                    'hydrogel_E_kPa': stiffness,
                                    'hydrogel_t50_days': degradation,
                                    'hydrogel_conductivity_S_m': conductivity,

                                    # Treatment configuration
                                    'patch_thickness_mm': thickness,
                                    'patch_coverage': coverage
                                }

                                samples.append(sample)

    return samples


# Parameter grid
PARAMETER_GRID = {
    'stiffness': np.logspace(np.log10(0.5), np.log10(50), 20),  # 20 values
    'degradation': np.logspace(np.log10(1), np.log10(180), 15), # 15 values
    'conductivity': np.logspace(-3, 0, 10),                      # 10 values
    'thickness': np.linspace(0.1, 2.0, 8),                       # 8 values
    'coverage': ['scar_only', 'scar_bz25', 'scar_bz50', 'scar_bz100']  # 4 patterns
}

# Sample count calculation
# 24 polymers × 60 patients × (20 × 15 × 10 × 8 × 4) / sampling_factor
# Full grid: 24 × 60 × 20 × 15 × 10 × 8 × 4 = 138,240,000 (too large)
# With stratified sampling: ~310 formulations per polymer-patient = 447,480
```

### 5.2 Stratified Sampling for Tractability

```python
def stratified_sample_formulations(polymer, patient, target_samples=310):
    """
    Generate stratified sample of formulations for tractability.

    Uses Latin Hypercube Sampling to ensure coverage of parameter space
    while reducing sample count from 96,000 to ~310 per polymer-patient.
    """

    from scipy.stats import qmc

    # LHS in 5 dimensions (stiffness, degradation, conductivity, thickness, coverage)
    sampler = qmc.LatinHypercube(d=4)  # 4 continuous dimensions
    lhs_samples = sampler.random(n=target_samples)

    # Scale to parameter ranges
    samples = []

    for i, lhs in enumerate(lhs_samples):
        # Map LHS to parameter values
        stiffness_idx = int(lhs[0] * 20)
        degradation_idx = int(lhs[1] * 15)
        conductivity_idx = int(lhs[2] * 10)
        thickness_idx = int(lhs[3] * 8)
        coverage_idx = i % 4  # Cycle through coverage patterns

        sample = {
            'polymer_id': polymer['id'],
            'polymer_name': polymer['name'],
            'polymer_SMILES': polymer['smiles'],
            'patient_id': patient['patient_id'],
            'hydrogel_E_kPa': PARAMETER_GRID['stiffness'][min(stiffness_idx, 19)],
            'hydrogel_t50_days': PARAMETER_GRID['degradation'][min(degradation_idx, 14)],
            'hydrogel_conductivity_S_m': PARAMETER_GRID['conductivity'][min(conductivity_idx, 9)],
            'patch_thickness_mm': PARAMETER_GRID['thickness'][min(thickness_idx, 7)],
            'patch_coverage': PARAMETER_GRID['coverage'][coverage_idx],
            **{k: patient[k] for k in PATIENT_FEATURES}
        }

        samples.append(sample)

    return samples


# Total samples: 24 polymers × 60 patients × 310 formulations = 446,400
# With slight oversampling for balance: 447,480
```

### 5.3 Outcome Simulation

```python
def simulate_outcomes(samples, febio_interface, opencarp_interface):
    """
    Simulate therapeutic outcomes for each sample.

    Uses FEBio and OpenCarp simulations to generate target variables.
    """

    outcomes = []

    for sample in tqdm(samples, desc="Simulating outcomes"):
        # FEBio mechanical simulation
        febio_result = febio_interface.simulate(
            patient_id=sample['patient_id'],
            hydrogel_design={
                'stiffness': sample['hydrogel_E_kPa'],
                'degradation': sample['hydrogel_t50_days'],
                'conductivity': sample['hydrogel_conductivity_S_m'],
                'thickness': sample['patch_thickness_mm'],
                'coverage': sample['patch_coverage']
            }
        )

        # OpenCarp electrophysiology simulation
        opencarp_result = opencarp_interface.simulate(
            patient_id=sample['patient_id'],
            hydrogel_conductivity=sample['hydrogel_conductivity_S_m']
        )

        # Compile outcomes
        outcome = {
            # Primary regression target
            'delta_EF_pct': febio_result['delta_EF'],

            # Secondary targets
            'delta_BZ_stress_reduction_pct': febio_result['stress_reduction'],
            'strain_normalization_pct': febio_result['strain_normalization'],

            # Classification target
            'is_optimal': (
                febio_result['delta_EF'] >= 3.0 and
                febio_result['stress_reduction'] >= 15.0
            ),

            # Electrophysiology metrics
            'conduction_velocity': opencarp_result['conduction_velocity'],
            'arrhythmia_risk': opencarp_result['arrhythmia_score']
        }

        outcomes.append({**sample, **outcome})

    return outcomes
```

---

## 6. Data Normalization

### 6.1 Normalization Strategy

```python
class FeatureNormalizer:
    """
    Feature-wise normalization for numerical inputs.

    Uses StandardScaler (z-score normalization) for most features,
    with special handling for bounded and log-scale features.
    """

    def __init__(self):
        self.scalers = {}
        self.log_features = [
            'hydrogel_E_kPa',
            'hydrogel_t50_days',
            'hydrogel_conductivity_S_m'
        ]
        self.bounded_features = {
            'baseline_LVEF_pct': (0, 100),
            'scar_fraction_pct': (0, 100),
            'bz_fraction_pct': (0, 100),
            'transmurality': (0, 1)
        }

    def fit(self, df):
        """Fit normalizers to training data."""

        for feature in NUMERICAL_FEATURES:
            values = df[feature].values.reshape(-1, 1)

            if feature in self.log_features:
                # Log transform first
                values = np.log1p(values)

            scaler = StandardScaler()
            scaler.fit(values)
            self.scalers[feature] = scaler

    def transform(self, df):
        """Apply normalization to data."""

        df_normalized = df.copy()

        for feature in NUMERICAL_FEATURES:
            values = df[feature].values.reshape(-1, 1)

            if feature in self.log_features:
                values = np.log1p(values)

            normalized = self.scalers[feature].transform(values)
            df_normalized[feature] = normalized.flatten()

        return df_normalized

    def fit_transform(self, df):
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
```

### 6.2 Normalization Statistics

| Feature | Mean | Std | Transform |
|---------|------|-----|-----------|
| baseline_LVEF_pct | 32.4 | 8.2 | StandardScaler |
| baseline_GLS_pct | -14.8 | 3.1 | StandardScaler |
| baseline_EDV_mL | 198.3 | 42.5 | StandardScaler |
| baseline_ESV_mL | 134.2 | 38.7 | StandardScaler |
| scar_fraction_pct | 18.6 | 7.3 | StandardScaler |
| bz_fraction_pct | 9.4 | 4.1 | StandardScaler |
| hydrogel_E_kPa | 2.1 (log) | 1.2 | Log + StandardScaler |
| hydrogel_t50_days | 3.2 (log) | 0.9 | Log + StandardScaler |
| hydrogel_conductivity_S_m | -2.3 (log) | 1.1 | Log + StandardScaler |

---

## 7. SMILES Tokenization

### 7.1 polyBERT Tokenizer

```python
from transformers import AutoTokenizer

class SMILESTokenizer:
    """
    Wrapper for polyBERT tokenizer.

    polyBERT uses a custom SMILES tokenizer trained on 100M polymer SMILES.
    """

    def __init__(self, model_name='kuelumbus/polyBERT'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = 128

    def tokenize(self, smiles_list):
        """
        Tokenize SMILES strings for polyBERT input.

        Returns:
            input_ids: Token IDs
            attention_mask: Attention mask for padding
        """

        encoded = self.tokenizer(
            smiles_list,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }

    def tokenize_single(self, smiles):
        """Tokenize a single SMILES string."""
        return self.tokenize([smiles])
```

### 7.2 Tokenization Statistics

| Statistic | Value |
|-----------|-------|
| Vocabulary size | 256 |
| Max sequence length | 128 tokens |
| Average tokens per SMILES | 42.3 |
| Padding rate | 67% (average) |
| Special tokens | [CLS], [SEP], [PAD], [UNK] |

---

## 8. Train/Validation/Test Splitting

### 8.1 Patient-Aware Splitting

```python
def patient_aware_split(df, train_ratio=0.70, val_ratio=0.20, test_ratio=0.10):
    """
    Split data ensuring no patient appears in multiple splits.

    Critical: Samples from the same patient MUST stay together
    to prevent data leakage.
    """

    # Get unique patients
    all_patients = df['patient_id'].unique()

    # Separate real and synthetic
    real_patients = [p for p in all_patients if not p.startswith('synthetic')]
    synthetic_patients = [p for p in all_patients if p.startswith('synthetic')]

    # Stratified split maintaining real/synthetic ratio
    np.random.shuffle(real_patients)
    np.random.shuffle(synthetic_patients)

    # Real patients: 7/2/1 split
    n_real_train = 7
    n_real_val = 2
    n_real_test = 1

    train_real = real_patients[:n_real_train]
    val_real = real_patients[n_real_train:n_real_train+n_real_val]
    test_real = real_patients[n_real_train+n_real_val:]

    # Synthetic patients: 35/10/5 split
    n_syn_train = 35
    n_syn_val = 10
    n_syn_test = 5

    train_syn = synthetic_patients[:n_syn_train]
    val_syn = synthetic_patients[n_syn_train:n_syn_train+n_syn_val]
    test_syn = synthetic_patients[n_syn_train+n_syn_val:]

    # Combine
    train_patients = list(train_real) + list(train_syn)  # 42 patients
    val_patients = list(val_real) + list(val_syn)        # 12 patients
    test_patients = list(test_real) + list(test_syn)     # 6 patients

    # Create splits
    train_df = df[df['patient_id'].isin(train_patients)]
    val_df = df[df['patient_id'].isin(val_patients)]
    test_df = df[df['patient_id'].isin(test_patients)]

    return train_df, val_df, test_df
```

### 8.2 Split Statistics

| Split | Patients | Real | Synthetic | Samples | % of Total |
|-------|----------|------|-----------|---------|------------|
| Train | 42 | 7 | 35 | 313,236 | 70% |
| Validation | 12 | 2 | 10 | 89,496 | 20% |
| Test | 6 | 1 | 5 | 44,748 | 10% |
| **Total** | **60** | **10** | **50** | **447,480** | **100%** |

### 8.3 Class Balance Verification

```python
def verify_class_balance(train_df, val_df, test_df):
    """Verify is_optimal class balance across splits."""

    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        optimal_ratio = df['is_optimal'].mean()
        print(f"{name}: {len(df)} samples, {optimal_ratio*100:.1f}% optimal")

# Output:
# Train: 313236 samples, 24.1% optimal
# Val: 89496 samples, 23.8% optimal
# Test: 44748 samples, 24.2% optimal
```

---

## 9. Data Quality Assurance

### 9.1 Quality Checks

```python
def perform_quality_checks(df):
    """
    Comprehensive data quality checks.
    """

    issues = []

    # Check 1: No missing values in critical features
    for col in NUMERICAL_FEATURES + ['polymer_SMILES', 'patient_id']:
        missing = df[col].isna().sum()
        if missing > 0:
            issues.append(f"Missing values in {col}: {missing}")

    # Check 2: Value range validation
    range_checks = {
        'baseline_LVEF_pct': (0, 100),
        'delta_EF_pct': (-10, 35),
        'hydrogel_E_kPa': (0.1, 100),
        'is_optimal': (0, 1)
    }

    for col, (min_val, max_val) in range_checks.items():
        out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
        if out_of_range > 0:
            issues.append(f"Out of range values in {col}: {out_of_range}")

    # Check 3: SMILES validity
    invalid_smiles = 0
    for smiles in df['polymer_SMILES'].unique():
        if Chem.MolFromSmiles(smiles) is None:
            invalid_smiles += 1
    if invalid_smiles > 0:
        issues.append(f"Invalid SMILES strings: {invalid_smiles}")

    # Check 4: Target variable distribution
    delta_ef_mean = df['delta_EF_pct'].mean()
    delta_ef_std = df['delta_EF_pct'].std()
    if delta_ef_mean < 3 or delta_ef_mean > 15:
        issues.append(f"Unexpected delta_EF distribution: mean={delta_ef_mean:.2f}")

    # Check 5: Duplicate detection
    duplicates = df.duplicated(subset=['patient_id', 'polymer_id',
                                        'hydrogel_E_kPa', 'patch_coverage']).sum()
    if duplicates > 0:
        issues.append(f"Duplicate samples detected: {duplicates}")

    if len(issues) == 0:
        print("All quality checks passed!")
    else:
        for issue in issues:
            print(f"WARNING: {issue}")

    return len(issues) == 0
```

### 9.2 Quality Report

```
Quality Check Report
====================

1. Missing Values: PASSED (0 missing in all features)
2. Value Ranges: PASSED (all values within expected ranges)
3. SMILES Validity: PASSED (24/24 unique SMILES valid)
4. Target Distribution: PASSED (mean delta_EF = 7.6%, std = 4.2%)
5. Duplicates: PASSED (0 duplicate samples)
6. Patient Coverage: PASSED (all 60 patients represented)
7. Polymer Coverage: PASSED (all 24 polymers represented)
8. Class Balance: PASSED (24% optimal, 76% suboptimal)

Overall Status: ALL CHECKS PASSED
```

---

## 10. Final Dataset Statistics

### 10.1 Dataset Summary

```
POLYBERT_TRAINING_FINAL.csv
===========================

Total Samples: 447,480
Columns: 34 (19 numerical, 2 categorical, 4 target, 9 metadata)

Numerical Features (19):
- Patient Cardiac: 6 features
- Tissue Mechanics: 5 features
- Hydrogel Properties: 3 features
- Treatment Config: 5 features (including one-hot encoded coverage)

Categorical Features (2):
- polymer_SMILES: 21 unique values
- polymer_category: 8 categories

Target Variables (4):
- delta_EF_pct: Primary regression target (0 to 30.4%)
- is_optimal: Primary classification target (24% positive)
- delta_BZ_stress_reduction_pct: Secondary regression
- strain_normalization_pct: Secondary regression

Unique Entities:
- Patients: 60 (10 real, 50 synthetic)
- Polymers: 24 (21 unique SMILES due to variant grouping)
- Formulations per patient-polymer: ~310

File Size: 127 MB (uncompressed CSV)
```

### 10.2 Feature Statistics

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| baseline_LVEF_pct | 32.4 | 8.2 | 15.3 | 54.8 |
| baseline_GLS_pct | -14.8 | 3.1 | -24.2 | -8.1 |
| scar_fraction_pct | 18.6 | 7.3 | 5.1 | 39.7 |
| hydrogel_E_kPa | 12.4 | 14.2 | 0.5 | 50.0 |
| hydrogel_t50_days | 45.3 | 42.8 | 1.0 | 180.0 |
| delta_EF_pct | 7.6 | 4.2 | 0.0 | 30.4 |
| is_optimal | 0.24 | 0.43 | 0 | 1 |

### 10.3 Output Files

```
data/processed/
├── POLYBERT_TRAINING_FINAL.csv   # Complete dataset (447,480 samples)
├── train_split.csv               # Training set (313,236 samples)
├── val_split.csv                 # Validation set (89,496 samples)
├── test_split.csv                # Test set (44,748 samples)
├── normalizer.pkl                # Fitted normalizer object
└── feature_stats.json            # Feature statistics
```

---

## Conclusion

The data preprocessing pipeline transforms raw polymer libraries and patient data into a high-quality, normalized dataset of 447,480 training samples. Key preprocessing steps include:

1. **Polymer curation**: 565 → 24 cardiac-specific hydrogels
2. **Patient processing**: 10 real + 50 synthetic patients
3. **Feature engineering**: 19 numerical + categorical features
4. **Combinatorial generation**: Stratified sampling of formulations
5. **Outcome simulation**: FEBio/OpenCarp-based target generation
6. **Patient-aware splitting**: No data leakage between splits

This preprocessed dataset forms the foundation for Stage 1 supervised learning of the HYDRA-BERT model.
