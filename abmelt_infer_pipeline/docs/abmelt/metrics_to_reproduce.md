# AbMelt Metrics to Reproduce

This file contains the antibody sequences, experimental values, and feature values for validating the AbMelt pipeline predictions.

## Table of Contents
- [Expected Holdout Test Performance](#expected-holdout-test-performance-original-6-sample-set)
- [Features & Targets](#features--targets)
- [Antibody Sequences](#antibody-sequences)
- [Running Predictions](#running-predictions)
- [Expected Results](#expected-results)
- [Data Sources](#data-sources)
- [Simulation Time Requirements](#simulation-time-requirements)
- [Notes](#notes)

---

## Expected Holdout Test Performance (Original 6-sample set)

These performance metrics were obtained from the original AbMelt study using jackknife resampling (k=18) on the full 6-antibody holdout set:

| Model | R² (Coefficient of Determination) | r_p² (Pearson Correlation Coefficient Squared) |
|-------|-----------------------------------|-----------------------------------------------|
| Tm (Melting Temperature) | 0.60 ± 0.06 | 0.64 ± 0.04 |
| Tagg (Aggregation Temperature) | 0.57 ± 0.11 | 0.71 ± 0.09 |
| Tmon (Onset Melting Temperature) | 0.56 ± 0.01 | 0.61 ± 0.0003 |

**Note**: Original metrics computed on the complete 6-antibody holdout set. Reproduction using the 4 public antibodies below may show different performance values due to the smaller sample size.

## Features & Targets

| ID | Antibody | PDB File | Tm (°C) | Tagg (°C) | Tmon (°C) | gyr_cdrs_Rg_std_350 | bonds_contacts_std_350 | rmsf_cdrl1_std_350 | rmsf_cdrs_mu_400 | gyr_cdrs_Rg_std_400 | all-temp_lamda_b=25_eq=20 | all-temp-sasa_core_mean_k=20_eq=20 | all-temp-sasa_core_std_k=20_eq=20 | r-lamda_b=2.5_eq=20 |
|----|----------|----------|---------|-----------|-----------|---------------------|------------------------|--------------------|------------------|---------------------|---------------------------|-------------------------------------|-----------------------------------|---------------------|
| DAB006808 | daclizumab | `data/abmelt/public_pdbs/daclizumab.pdb` | 69.84 | 94.64 | 54.07 | 0.0200 | 14.7060 | 0.0528 | 0.1694 | 0.0292 | 0.9350 | 0.0709 | 0.2059 | 0.8219 |
| DAB006768 | sirukumab | `data/abmelt/public_pdbs/sirukumab.pdb` | 67.90 | 66.00 | 62.66 | 0.0174 | 14.2404 | 0.0624 | 0.2189 | 0.0274 | 1.4331 | 0.0942 | 0.0632 | 0.8549 |
| DAB005027 | epratuzumab | `data/abmelt/public_pdbs/epratuzumab.pdb` | 64.30 | 78.39 | 59.48 | 0.0274 | 14.1147 | 0.0666 | 0.1522 | 0.0315 | 0.8980 | 0.0547 | 0.1270 | 0.8595 |
| DAB006766 | sifalimumab | `data/abmelt/public_pdbs/sifalimumab.pdb` | 65.95 | 79.14 | 61.46 | 0.0180 | 13.9916 | 0.0563 | 0.2206 | 0.0427 | 1.4279 | -0.1362 | -0.3937 | 0.8033 |

---

## Antibody Sequences

### daclizumab (DAB006808)
- **VH**: `QVQLVQSGAEVKKPGSSVKVSCKASGYTFTSYRMHWVRQAPGQGLEWIGYINPSTGYTEYNQKFKDKATITADESTNTAYMELSSLRSEDTAVYYCARGGGVFDYWGQGTLVTVSS`
- **VL**: `DIQMTQSPSTLSASVGDRVTITCSASSSISYMHWYQQKPGKAPKLLIYTTSNLASGVPARFSGSGSGTEFTLTISSLQPDDFATYYCHQRSTYPLTFGQGTKVEVK`

### sirukumab (DAB006768)
- **VH**: `EVQLVESGGGLVQPGGSLRLSCAASGFTFSPFAMSWVRQAPGKGLEWVAKISPGGSWTYYSDTVTGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARQLWGYYALDIWGQGTTVTVSS`
- **VL**: `EIVLTQSPATLSLSPGERATLSCSASISVSYMYWYQQKPGQAPRLLIYDMSNLASGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCMQWSGYPYTFGGGTKVEIK`

### epratuzumab (DAB005027)
- **VH**: `QVQLVQSGAEVKKPGSSVKVSCKASGYTFTSYWLHWVRQAPGQGLEWIGYINPRNDYTEYNQNFKDKATITADESTNTAYMELSSLRSEDTAFYFCARRDITTFYWGQGTTVTVSS`
- **VL**: `DIQLTQSPSSLSASVGDRVTMSCKSSQSVLYSANHKNYLAWYQQKPGKAPKLLIYWASTRESGVPSRFSGSGSGTDFTFTISSLQPEDIATYYCHQYLSSWTFGGGTKLEIK`

### sifalimumab (DAB006766)
- **VH**: `QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYSISWVRQAPGQGLEWMGWISVYNGNTNYAQKFQGRVTMTTDTSTSTAYLELRSLRSDDTAVYYCARDPIAAGYWGQGTLVTVSS`
- **VL**: `EIVLTQSPGTLSLSPGERATLSCRASQSVSSTYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGSSPRTFGQGTKVEIK`

---

## Running Predictions

### Using PDB files:
```bash
python infer.py --pdb data/abmelt/public_pdbs/daclizumab.pdb --name daclizumab --config configs/paper_config.yaml
python infer.py --pdb data/abmelt/public_pdbs/sirukumab.pdb --name sirukumab --config configs/paper_config.yaml
python infer.py --pdb data/abmelt/public_pdbs/epratuzumab.pdb --name epratuzumab --config configs/paper_config.yaml
python infer.py --pdb data/abmelt/public_pdbs/sifalimumab.pdb --name sifalimumab --config configs/paper_config.yaml
```

### Using sequences:
```bash
# daclizumab
python infer.py \
  --h "QVQLVQSGAEVKKPGSSVKVSCKASGYTFTSYRMHWVRQAPGQGLEWIGYINPSTGYTEYNQKFKDKATITADESTNTAYMELSSLRSEDTAVYYCARGGGVFDYWGQGTLVTVSS" \
  --l "DIQMTQSPSTLSASVGDRVTITCSASSSISYMHWYQQKPGKAPKLLIYTTSNLASGVPARFSGSGSGTEFTLTISSLQPDDFATYYCHQRSTYPLTFGQGTKVEVK" \
  --name "daclizumab" \
  --config configs/paper_config.yaml

# sirukumab
python infer.py \
  --h "EVQLVESGGGLVQPGGSLRLSCAASGFTFSPFAMSWVRQAPGKGLEWVAKISPGGSWTYYSDTVTGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARQLWGYYALDIWGQGTTVTVSS" \
  --l "EIVLTQSPATLSLSPGERATLSCSASISVSYMYWYQQKPGQAPRLLIYDMSNLASGIPARFSGSGSGTDFTLTISSLEPEDFAVYYCMQWSGYPYTFGGGTKVEIK" \
  --name "sirukumab" \
  --config configs/paper_config.yaml

# epratuzumab
python infer.py \
  --h "QVQLVQSGAEVKKPGSSVKVSCKASGYTFTSYWLHWVRQAPGQGLEWIGYINPRNDYTEYNQNFKDKATITADESTNTAYMELSSLRSEDTAFYFCARRDITTFYWGQGTTVTVSS" \
  --l "DIQLTQSPSSLSASVGDRVTMSCKSSQSVLYSANHKNYLAWYQQKPGQAPKLLIYWASTRESGVPSRFSGSGSGTDFTFTISSLQPEDIATYYCHQYLSSWTFGGGTKLEIK" \
  --name "epratuzumab" \
  --config configs/paper_config.yaml

# sifalimumab
python infer.py \
  --h "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYSISWVRQAPGQGLEWMGWISVYNGNTNYAQKFQGRVTMTTDTSTSTAYLELRSLRSDDTAVYYCARDPIAAGYWGQGTLVTVSS" \
  --l "EIVLTQSPGTLSLSPGERATLSCRASQSVSSTYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGSSPRTFGQGTKVEIK" \
  --name "sifalimumab" \
  --config configs/paper_config.yaml
```

## Expected Results

The pipeline should predict Tm, Tagg, and Tmon values that match the experimental values shown in the tables above. These antibodies are from the holdout test set and were not used in model training.

## Data Sources

- Experimental values: `data/abmelt/*_holdout_denormalized.csv`
- Feature values: `AbMelt/data/{tm,tagg,tmon}/holdout.csv`
- PDB structures: `data/abmelt/public_pdbs/`
- Variable region sequences: From the holdout dataset

## Simulation Time Requirements

### Features by Minimum Simulation Time

**Short simulations (~5-10ns)**: 6 features can be computed
- All 3 Tm features: `gyr_cdrs_Rg_std_350`, `bonds_contacts_std_350`, `rmsf_cdrl1_std_350`
- 2 Tagg features: `rmsf_cdrs_mu_400`, `gyr_cdrs_Rg_std_400`
- Models affected: **Tm** (complete), **Tagg** (partial)

**Longer simulations (45-100ns)**: All 9 features require equilibration
- 1 Tagg feature: `all-temp_lamda_b=25_eq=20` (needs 45ns: 20ns eq + 25ns block)
- 3 Tmon features: All require 20ns+ equilibration
  - `all-temp-sasa_core_mean_k=20_eq=20` (20ns min)
  - `all-temp-sasa_core_std_k=20_eq=20` (20ns min)
  - `r-lamda_b=2.5_eq=20` (22.5ns: 20ns eq + 2.5ns block)
- Models affected: **Tagg** (complete), **Tmon** (complete)

**Recommended**: 100ns per temperature (paper_config) for production-quality predictions

### Configuration Impact

- `testing_config.yaml` (2ns, eq=0): Can compute **6/9 features** - Cannot make full Tagg or Tmon predictions
- `paper_config.yaml` (100ns, eq=20): Can compute **9/9 features** - Full predictions for all models

## Notes

- Feature values are rounded to 4 decimal places for display
- All feature values shown are the **normalized** values used as model inputs
- Target values (Tm, Tagg, Tmon) are shown in **denormalized** form (°C)
