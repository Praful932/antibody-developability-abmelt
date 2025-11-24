# Corrected Missing Features Analysis

## IMPORTANT: Using Correct Feature Files

The models use features from `rf_efs.csv` files (as shown in `prediction.py` line 64), NOT from `rf_efs_best_features.csv`.

## Required Features by Model (CORRECTED)

### 1. TM Model
**Training data:** `AbMelt/data/tm/rf_efs.csv`

**Required features (3):**
1. `gyr_cdrs_Rg_std_350` - Radius of gyration std for CDRs at 350K
2. `bonds_contacts_std_350` - Contact bonds std at 350K  
3. `rmsf_cdrl1_std_350` - RMSF std for CDR-L1 at 350K

**Status:** ✅ **ALL COMPUTED** - These features are generated correctly by the pipeline

---

### 2. TAGG Model
**Training data:** `AbMelt/data/tagg/rf_efs.csv`

**Required features (3):**
1. `rmsf_cdrs_mu_400` - RMSF mean for all CDRs at 400K
2. `gyr_cdrs_Rg_std_400` - Radius of gyration std for CDRs at 400K
3. `all-temp_lamda_b=25_eq=20` - Multi-temperature lambda with block=25ns

**Status:** ⚠️ **PARTIALLY MISSING**
- ✅ `rmsf_cdrs_mu_400` - Should be computed (RMSF features are generated)
- ✅ `gyr_cdrs_Rg_std_400` - Should be computed (gyration features are generated)
- ❌ `all-temp_lamda_b=25_eq=20` - Config has block_length=10, need block_length=25

---

### 3. TMON Model
**Training data:** `AbMelt/data/tmon/rf_efs.csv`

**Required features (4):**
1. `bonds_contacts_std_350` - Contact bonds std at 350K
2. `all-temp-sasa_core_mean_k=20_eq=20` - Cross-temperature core SASA slope, k=20
3. `all-temp-sasa_core_std_k=20_eq=20` - Cross-temperature core SASA std slope, k=20
4. `r-lamda_b=2.5_eq=20` - Lambda correlation with block=2.5ns

**Status:** ⚠️ **PARTIALLY MISSING**
- ✅ `bonds_contacts_std_350` - Should be computed
- ⚠️ `all-temp-sasa_core_mean_k=20_eq=20` - **Computed but need to verify naming**
- ⚠️ `all-temp-sasa_core_std_k=20_eq=20` - **Computed but need to verify naming**
- ❌ `r-lamda_b=2.5_eq=20` - Config has block_length=10, need block_length=2.5

---

## Critical Issues (UPDATED)

### Issue 1: Lambda Features Need Multiple Block Lengths

**Models need:**
- TAGG: `all-temp_lamda_b=25_eq=20` (block=25)
- TMON: `r-lamda_b=2.5_eq=20` (block=2.5)

**Current implementation:**
```python
# compute_descriptors.py line ~545-547
descriptor_dict[f'all-temp_lamda_b={block_length}_eq={eq_time}'] = lambda_mean
descriptor_dict[f'r-lamda_b={block_length}_eq={eq_time}'] = lambda_mean
```
With config: `block_length: 10` (wrong for both models!)

**Fix:** Compute at multiple block lengths [2.5, 10, 25] or at minimum [2.5, 25]

---

### Issue 2: SASA Feature Naming Verification Needed

**TMON needs:**
- `all-temp-sasa_core_mean_k=20_eq=20`
- `all-temp-sasa_core_std_k=20_eq=20`

**Current implementation:**
```python
# compute_descriptors.py line ~569
descriptor_dict[f'all-temp-sasa_{key}_k={core_surface_k}_eq={eq_time}'] = slope
```

This looks correct! But need to verify:
1. That `key` includes both `core_mean` and `core_std`
2. That cross-temperature slopes are being computed for both
3. That k=20 is the default (it is in config)

**Action:** Test to verify these features are actually generated

---

### Issue 3: RMSF and Gyration for CDRs at 400K

**TAGG needs:**
- `rmsf_cdrs_mu_400`
- `gyr_cdrs_Rg_std_400`

**Current implementation generates:**
```python
# For RMSF (line ~476-477)
descriptor_dict[f'rmsf_{region}_mu_{temp}'] = mu
descriptor_dict[f'rmsf_{region}_std_{temp}'] = std

# For gyration (line ~481-482)  
descriptor_dict[f'gyr_{region}_Rg_mu_{temp}'] = mu
descriptor_dict[f'gyr_{region}_Rg_std_{temp}'] = std
```

Where `region` includes 'cdrs' and `temp` includes '400', so:
- `rmsf_cdrs_mu_400` ✅ Should be generated
- `gyr_cdrs_Rg_std_400` ✅ Should be generated

**Action:** Test to verify these are actually generated

---

## Updated Recommendations

### HIGH PRIORITY FIX: Support Multiple Block Lengths for Lambda

The pipeline must compute lambda features at multiple block lengths since:
- TAGG needs block=25
- TMON needs block=2.5

**Option 1: Compute at all needed block lengths**
```yaml
# In paper_config.yaml
descriptors:
  block_lengths: [2.5, 25]  # Only what's needed for the two models
```

**Option 2: Compute at many block lengths for flexibility**
```yaml
descriptors:
  block_lengths: [2.5, 5, 7.5, 10, 12.5, 15, 20, 25]
```

**Code changes needed in compute_descriptors.py:**

1. Modify `_compute_order_parameters()` to accept multiple block lengths:
```python
def _compute_order_parameters(work_dir: Path, temps: List[str], eq_time: int, 
                              block_lengths: List[float], antibody_name: str) -> Dict:
    """Returns dict mapping block_length to master_s2_dict"""
    all_s2_dicts = {}
    for block_length in block_lengths:
        master_s2_dict = {int(temp): {} for temp in temps}
        for temp in temps:
            s2_blocks_dict = order_s2(mab=antibody_name, temp=temp, 
                                     block_length=block_length, start=eq_time)
            master_s2_dict[int(temp)] = avg_s2_blocks(s2_blocks_dict)
        all_s2_dicts[block_length] = master_s2_dict
    return all_s2_dicts
```

2. Update `_compute_lambda_features()` signature and implementation

3. Update feature generation to create features for each block length:
```python
# In _aggregate_descriptors_to_dataframe()
if all_lambda_features:
    for block_length, (lambda_dict, r_dict) in all_lambda_features.items():
        lambda_mean = np.mean(list(lambda_dict.values()))
        r_mean = np.mean(list(r_dict.values()))
        descriptor_dict[f'all-temp_lamda_b={block_length}_eq={eq_time}'] = lambda_mean
        descriptor_dict[f'r-lamda_b={block_length}_eq={eq_time}'] = r_mean
```

---

### MEDIUM PRIORITY: Verify Feature Generation

Create a test to verify these features are actually generated:

**TM Model:**
- [x] `gyr_cdrs_Rg_std_350`
- [x] `bonds_contacts_std_350`
- [x] `rmsf_cdrl1_std_350`

**TAGG Model:**
- [ ] `rmsf_cdrs_mu_400` - need to verify
- [ ] `gyr_cdrs_Rg_std_400` - need to verify
- [ ] `all-temp_lamda_b=25_eq=20` - need to add block=25 computation

**TMON Model:**
- [ ] `bonds_contacts_std_350` - need to verify
- [ ] `all-temp-sasa_core_mean_k=20_eq=20` - need to verify naming
- [ ] `all-temp-sasa_core_std_k=20_eq=20` - need to verify naming  
- [ ] `r-lamda_b=2.5_eq=20` - need to add block=2.5 computation

---

## Quick Fix Summary

### What MUST be fixed:
1. **Lambda block lengths** - Compute at block=[2.5, 25] minimum
   - Change config from single value to list
   - Update `_compute_order_parameters()` to loop over block lengths
   - Update `_compute_lambda_features()` to handle multiple block lengths
   - Update feature aggregation to generate features for each block length

### What should be verified:
2. **SASA cross-temp features** - Verify naming matches `all-temp-sasa_core_mean_k=20_eq=20`
3. **RMSF CDRs features** - Verify `rmsf_cdrs_mu_400` is generated
4. **Gyration CDRs features** - Verify `gyr_cdrs_Rg_std_400` is generated

---

## Testing Checklist

After implementing fixes:

1. Run descriptor computation on test antibody with temps=[300, 350, 400]
2. Check output DataFrame contains exactly these features:

**For TM:**
```python
required_features = [
    'gyr_cdrs_Rg_std_350',
    'bonds_contacts_std_350', 
    'rmsf_cdrl1_std_350'
]
```

**For TAGG:**
```python
required_features = [
    'rmsf_cdrs_mu_400',
    'gyr_cdrs_Rg_std_400',
    'all-temp_lamda_b=25_eq=20'
]
```

**For TMON:**
```python
required_features = [
    'bonds_contacts_std_350',
    'all-temp-sasa_core_mean_k=20_eq=20',
    'all-temp-sasa_core_std_k=20_eq=20',
    'r-lamda_b=2.5_eq=20'
]
```

3. Verify feature values are reasonable (non-zero, finite)
4. Load each model and attempt prediction
5. Compare predictions to expected ranges

---

## Summary: What Was Wrong in Original Analysis

❌ **WRONG:** I was looking at `rf_efs_best_features.csv` which had:
- TAGG: `r-lamda_b=7.5_eq=20` 
- TMON: `350_core_mean_k=60_eq=20`, `350_core_mean_k=50_eq=20`

✅ **CORRECT:** Should use `rf_efs.csv` which has:
- TAGG: `rmsf_cdrs_mu_400`, `gyr_cdrs_Rg_std_400`, `all-temp_lamda_b=25_eq=20`
- TMON: `bonds_contacts_std_350`, `all-temp-sasa_core_mean_k=20_eq=20`, `all-temp-sasa_core_std_k=20_eq=20`, `r-lamda_b=2.5_eq=20`

The `rf_efs.csv` files are the actual training data used for inference (as confirmed in `prediction.py` line 64).

