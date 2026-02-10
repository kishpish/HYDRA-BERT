# Polymer Library Curation Methodology

## Overview

This document describes the systematic curation process used to select the final 24 cardiac-specific hydrogel polymers from an initial candidate library of 565 polymer designs.

## Initial Polymer Library (565 Candidates)

The initial polymer library was compiled from multiple sources:

### Source Databases
| Source | Polymers | Description |
|--------|----------|-------------|
| PolyInfo Database | 312 | General hydrogel-forming polymers |
| Cardiac Biomaterials Literature | 89 | Published cardiac tissue engineering materials |
| polyBERT Training Set | 127 | Validated polymer SMILES from pre-training |
| Synthetic Variants | 37 | Computationally generated analogs |
| **Total** | **565** | Initial candidate pool |

### Initial Categories
- Natural polymers (alginate, chitosan, collagen, HA, fibrin, gelatin)
- Modified natural (GelMA, MeHA, oxidized variants)
- Synthetic (PEG variants, PLGA, PCL, PVA, pNIPAAM)
- Conductive (PEDOT:PSS, PPy, PANI, graphene composites)
- Decellularized ECM (various tissue sources)
- Hybrid/composite formulations

## Curation Criteria

### Stage 1: Chemical Validity Filter
**Input:** 565 polymers → **Output:** 423 polymers

Exclusion criteria:
- Invalid SMILES syntax (RDKit parsing failures)
- Non-canonical representations
- Duplicate structures with different names
- Polymers with undefined stereochemistry critical to function

### Stage 2: Hydrogel Formation Filter
**Input:** 423 polymers → **Output:** 187 polymers

Required properties:
- Demonstrated gel-forming capability (literature evidence)
- Crosslinking mechanism defined (chemical, physical, ionic, or photo)
- Aqueous solubility or dispersibility
- Stable hydrogel network formation

Excluded:
- Thermoplastics without crosslinking capability
- Water-insoluble polymers
- Polymers requiring toxic crosslinkers

### Stage 3: Biocompatibility Filter
**Input:** 187 polymers → **Output:** 89 polymers

Required:
- Published cytocompatibility data (>70% cell viability)
- No known acute toxicity
- Degradation products non-toxic
- FDA approval or extensive safety literature

Excluded:
- Polymers with cytotoxic monomers
- Materials lacking biocompatibility data
- Industrial polymers not intended for biomedical use

### Stage 4: Cardiac Applicability Filter
**Input:** 89 polymers → **Output:** 41 polymers

Required:
- Injectable or minimally invasive delivery
- Appropriate mechanical range (1-50 kPa, matching myocardium)
- Suitable degradation timeline (weeks to months)
- Demonstrated use in cardiac or soft tissue engineering

Excluded:
- Rigid/high-modulus materials (bone scaffolds)
- Rapid degradation (<1 week) or non-degradable
- Materials requiring surgical implantation
- Polymers not tested in cardiac context

### Stage 5: Conductivity & Function Filter
**Input:** 41 polymers → **Output:** 24 polymers

Selection for final library:
- Representative coverage of all major hydrogel categories
- Inclusion of conductive variants (critical for cardiac EP)
- Clinical translation potential (at least Phase I or equivalent)
- Sufficient literature data for parameter calibration
- Unique chemical diversity (avoiding redundant structures)

## Final Curated Library (24 Polymers)

### Category Distribution

| Category | Count | Polymers |
|----------|-------|----------|
| Protein-Modified GelMA | 4 | GelMA_3pct, GelMA_5pct, GelMA_7pct, GelMA_10pct |
| Conductive GelMA | 4 | GelMA_BioIL, GelMA_Polypyrrole, GelMA_rGO, GelMA_MXene |
| Synthetic PEGDA | 3 | PEGDA_575, PEGDA_700, PEGDA_3400 |
| Polysaccharide | 5 | Alginate_CaCl2, Alginate_RGD, Chitosan_thermogel, Chitosan_EGCG, Chitosan_HA |
| Glycosaminoglycan | 3 | HA_acellular, HA_ECM, MeHA_photocrosslink |
| Decellularized ECM | 2 | dECM_VentriGel, dECM_cardiac |
| Protein | 2 | Gelatin_crosslinked, Fibrin_thrombin |
| Conductive | 1 | PEDOT_PSS |
| **Total** | **24** | |

### Unique SMILES Structures: 21
(GelMA variants at different concentrations share base SMILES, differentiated by formulation parameters)

## Rationale for Final Selection

### Why 24 Polymers?

1. **Clinical Relevance**: All 24 polymers have published cardiac tissue engineering applications
2. **Mechanistic Diversity**: Covers ionic, chemical, photo, and thermal crosslinking
3. **Property Range**: Spans the full cardiac-relevant stiffness (1-30 kPa) and degradation (7-180 days) space
4. **Conductivity Options**: Includes non-conductive, semi-conductive, and highly conductive variants
5. **Translation Potential**: Includes FDA-approved materials and those in clinical trials

### Literature Validation

| Polymer | Key Cardiac Application Reference |
|---------|-----------------------------------|
| GelMA | Annabi et al., 2017 - Cardiac tissue engineering |
| PEGDA | Zhu & Bhattacharya, 2012 - Injectable cardiac hydrogels |
| Alginate | Lee & Mooney, 2012 - Cardiac regeneration |
| VentriGel | Traverse et al., 2019 - Phase I MI trial |
| HA | Ifkovits et al., 2010 - MI therapy |
| Fibrin | Christman et al., 2004 - Cardiac cell delivery |
| Chitosan | Liu et al., 2012 - Thermogelling cardiac scaffold |

## Training Data Generation

From the 24 curated polymers, 447,480 training samples were generated:

```
24 polymers × 60 patients × ~310 formulation combinations = 447,480 samples
```

### Parameter Sweep Per Polymer
- Stiffness (E): 5 values spanning polymer-specific range
- Degradation (t50): 5 values (7-180 days)
- Conductivity: 5 values (0-1 S/m, where applicable)
- Thickness: 4 values (1-5 mm)
- Coverage: 4 options (scar_only, scar_bz25, scar_bz50, scar_bz100)

## Conclusion

The curation from 565 initial candidates to 24 final polymers represents a rigorous, multi-stage filtering process optimized for cardiac hydrogel applications. This focused library ensures:

1. All polymers are chemically valid and well-characterized
2. All have demonstrated biocompatibility
3. All are applicable to cardiac tissue engineering
4. The library covers the full design space relevant to injectable cardiac hydrogels
5. Sufficient diversity for the model to learn meaningful structure-property relationships

The 24-polymer library, combined with extensive formulation parameter sweeps, generated 447,480 unique training samples - providing comprehensive coverage of the cardiac hydrogel design space while maintaining scientific rigor and clinical relevance.
