# Features Used by AbMelt Models

This document lists the specific features used by each trained model for predicting antibody thermostability metrics.

## Summary

- **Total unique features across all models**: 9
- **Total features per model**: Tm (3), Tagg (3), Tmon (4)

## Tm (Melting Temperature) Model
**Model Type**: Random Forest  
**Number of Features**: 3

1. `gyr_cdrs_Rg_std_350` - Standard deviation of radius of gyration for CDRs at 350K
2. `bonds_contacts_std_350` - Standard deviation of bond contacts at 350K
3. `rmsf_cdrl1_std_350` - Standard deviation of RMSF (root mean square fluctuation) for CDR-L1 at 350K

## Tagg (Aggregation Temperature) Model
**Model Type**: K-Nearest Neighbors (KNN)  
**Number of Features**: 3

1. `rmsf_cdrs_mu_400` - Mean RMSF for CDRs at 400K
2. `gyr_cdrs_Rg_std_400` - Standard deviation of radius of gyration for CDRs at 400K
3. `all-temp_lamda_b=25_eq=20` - Multi-temperature lambda feature with block size 25ns and 20ns equilibration

## Tmon (Onset Melting Temperature) Model
**Model Type**: ElasticNet  
**Number of Features**: 4

1. `bonds_contacts_std_350` - Standard deviation of bond contacts at 350K *(shared with Tm)*
2. `all-temp-sasa_core_mean_k=20_eq=20` - Mean core SASA across temperatures (k=20, 20ns equilibration)
3. `all-temp-sasa_core_std_k=20_eq=20` - Standard deviation of core SASA across temperatures (k=20, 20ns equilibration)
4. `r-lamda_b=2.5_eq=20` - Lambda feature with block size 2.5ns and 20ns equilibration

## Feature Categories

### Structural Dynamics
- **RMSF (Root Mean Square Fluctuation)**: Measures atomic position fluctuations
  - `rmsf_cdrl1_std_350` (Tm)
  - `rmsf_cdrs_mu_400` (Tagg)

### Gyration/Compactness
- **Radius of Gyration (Rg)**: Measures protein compactness
  - `gyr_cdrs_Rg_std_350` (Tm)
  - `gyr_cdrs_Rg_std_400` (Tagg)

### Contact/Bonding
- **Bond Contacts**: Number of atomic contacts/bonds
  - `bonds_contacts_std_350` (Tm, Tmon)

### Surface Properties
- **SASA (Solvent Accessible Surface Area)**: Core residue exposure
  - `all-temp-sasa_core_mean_k=20_eq=20` (Tmon)
  - `all-temp-sasa_core_std_k=20_eq=20` (Tmon)

### Order Parameters
- **Lambda Features**: Temperature-dependent order parameter slopes
  - `all-temp_lamda_b=25_eq=20` (Tagg)
  - `r-lamda_b=2.5_eq=20` (Tmon)

## Feature Notation

- **Temperature suffix** (e.g., `_350`, `_400`): Simulation temperature in Kelvin
- **mu**: Mean value
- **std**: Standard deviation
- **b**: Block size for averaging (in nanoseconds)
- **eq**: Equilibration time (in nanoseconds)
- **k**: k-nearest neighbors parameter for SASA core/surface classification
- **all-temp**: Feature computed across multiple temperatures

## Shared Features

Only **1 feature** is shared between models:
- `bonds_contacts_std_350` - Used by both **Tm** and **Tmon** models

All other features are unique to their respective models.

## Feature Source Files

- Tm features: `models/tm/rf_efs.csv`
- Tagg features: `models/tagg/rf_efs.csv`
- Tmon features: `models/tmon/rf_efs.csv`
