# AbMelt Trained Models

This directory contains the trained machine learning models for predicting antibody thermostability properties.

## Model Files

### T<sub>agg</sub> (Aggregation Temperature)
- **Model:** `tagg/efs_best_knn.pkl`
- **Algorithm:** K-Nearest Neighbors (KNN)
- **Features:** `tagg/rf_efs.csv`
  - `rmsf_cdrs_mu_400`
  - `gyr_cdrs_Rg_std_400`
  - `all-temp_lamda_b=25_eq=20`

### T<sub>m</sub> (Melting Temperature)
- **Model:** `tm/efs_best_randomforest.pkl`
- **Algorithm:** Random Forest
- **Features:** `tm/rf_efs.csv`
  - `gyr_cdrs_Rg_std_350`
  - `bonds_contacts_std_350`
  - `rmsf_cdrl1_std_350`

### T<sub>m,onset</sub> (Onset Melting Temperature)
- **Model:** `tmon/efs_best_elasticnet.pkl`
- **Algorithm:** ElasticNet
- **Features:** `tmon/rf_efs.csv`
  - `bonds_contacts_std_350`
  - `all-temp-sasa_core_mean_k=20_eq=20`
  - `all-temp-sasa_core_std_k=20_eq=20`
  - `r-lamda_b=2.5_eq=20`

## Model Training Details

These models were trained using:
- **Feature Selection:** Exhaustive Feature Selection (EFS) with Random Forest
- **Optimization:** Bayesian optimization (skopt) for hyperparameter tuning
- **Cross-validation:** Repeated K-Fold cross-validation
- **Training Data:** Internal antibody thermostability dataset

## Usage

Models are automatically loaded by the `AbMeltPredictor` class in `src/model_inference.py`:

```python
from src.model_inference import AbMeltPredictor

predictor = AbMeltPredictor()
predictions = predictor.predict_all(descriptors_df)
```

## Source

These models and feature definitions were copied from the original AbMelt paper implementation:
- Original location: `../AbMelt/models/`
- Paper: Rollins, Z.A., Widatalla, T., Cheng, A.C., & Metwally, E. (2024). AbMelt: Learning antibody thermostability from molecular dynamics.

## File Sizes

- `tagg/efs_best_knn.pkl`: ~1-5 KB (KNN is lightweight)
- `tm/efs_best_randomforest.pkl`: ~100-500 KB (depends on tree depth/count)
- `tmon/efs_best_elasticnet.pkl`: ~1-5 KB (linear model is lightweight)

## Notes

- Models expect features in the exact order specified in the `rf_efs.csv` files
- Feature values should be normalized/scaled consistently with training data
- The `rf_efs.csv` files include both feature columns and target columns (tagg, tm, tmonset)
- These are the best models selected from exhaustive feature selection experiments

