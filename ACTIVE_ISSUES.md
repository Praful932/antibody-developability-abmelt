# Active Issues: AbMelt Inference Pipeline

**Document Date:** 2025-11-25  
**Purpose:** Track discrepancies between original AbMelt paper code and new inference pipeline

---

## 🔴 CRITICAL ISSUES

### 1. ❌ Potential Feature Extraction (INCORRECT VALUES)

**Location:** `abmelt_infer_pipeline/src/compute_descriptors.py` lines 509-512

**Issue:** Extracting mean of all radius values instead of value at specific radius index

**Original Code** (`AbMelt/src/anaylze.py` lines 192-202):
```python
elif "potential" in metric:
    t = np.array(t)
    x = np.array(x)
    if "cdrs" in metric:
        radius = 5  # Use 5th index for CDRs group
    else:
        radius = 2  # Use 2nd index for individual CDRs (cdrl1, cdrl2, etc.)
    x_mu = x[radius]  # ← Takes value at SPECIFIC INDEX
    params = {'eq_time':0, 'eq_mu':x_mu, 'eq_std':0}
```

**Your Code** (`compute_descriptors.py`):
```python
elif 'potential' in metric_name:
    region = metric_name.replace('potential_', '').replace(f'_{temp}', '')
    # Use mean value (potential at specific radius)
    descriptor_dict[f'potential_{region}_mu_{temp}'] = mu  # ← WRONG: Takes MEAN over ALL radii
```

**Potential File Structure** (10 slices/radii):
```
Radius    Potential
0         -nan
0.644     -0.256
1.289     -0.512
1.934     -0.683  ← radius index 2 (for individual CDRs)
2.579     -0.769
3.224     -0.820  ← radius index 5 (for cdrs)
...
```

**Impact:** All potential features will have incorrect values

**Fix Required:**
```python
elif 'potential' in metric_name:
    region = metric_name.replace('potential_', '').replace(f'_{temp}', '')
    # Potential is measured at specific radius indices
    if 'cdrs' in region and region != 'cdrs':
        # Individual CDRs (cdrl1, cdrl2, cdrl3, cdrh1, cdrh2, cdrh3)
        radius_idx = 2
    elif region == 'cdrs':
        # All CDRs combined
        radius_idx = 5
    else:
        radius_idx = 2  # Default
    
    # Extract value at specific radius (no equilibration for potential)
    if equilibrated_data.ndim == 2:
        descriptor_dict[f'potential_{region}_mu_{temp}'] = equilibrated_data[radius_idx, 1]
    else:
        descriptor_dict[f'potential_{region}_mu_{temp}'] = equilibrated_data[radius_idx]
    descriptor_dict[f'potential_{region}_std_{temp}'] = 0  # Original sets std to 0
```

---

## ⚠️ HIGH PRIORITY ISSUES

### 2. ⚠️ Dipole Moment Parsing (WRONG COLUMN)

**Location:** `abmelt_infer_pipeline/src/compute_descriptors.py` lines 513-515, 741-746

**Issue:** Parser may be extracting X column instead of Z column (magnitude)

**Original Code** (`AbMelt/src/anaylze.py` lines 204-214):
```python
elif "dipole" in metric:
    z = np.array(z)  # ← Uses Z column (Mz, magnitude)
    t0 = 2000
    z_equlibrated = z[t0:]
    z_mu = np.mean(z_equlibrated)
    z_std = np.std(z_equlibrated)
```

**Dipole File Structure** (4 columns):
```
Time    Mx          My          Mz(magnitude)    |Mtot|
0       -18.78      343.84      -81.97           353.98
10      -23.93      332.89      -54.45           338.16
...
```

**Your Parser** (`_parse_xvg_file` lines 719-746):
```python
elif len(cols) == 5:  # 5 columns including time
    t.append(float(cols[0]))
    r.append(float(cols[1]))  # ← Mx goes to r
    x.append(float(cols[2]))  # ← My goes to x
    y.append(float(cols[3]))  # ← Mz goes to y
    z.append(float(cols[4]))  # ← |Mtot| goes to z

# Return for 4 data columns
if len(r) > 0:
    return np.column_stack([r, x, y, z])  # Returns [Mx, My, Mz, |Mtot|]
```

**Then in 1D extraction** (lines 467-474):
```python
if equilibrated_data.ndim == 1:
    mu = np.mean(equilibrated_data)  # ← This would use FIRST column (Mx)
```

**Problem:** When `equilibrated_data.ndim == 2` with 4 columns, your code doesn't handle dipole correctly. It should extract the 3rd column (Mz) but may be using wrong logic.

**Impact:** Dipole features will have wrong values (Mx instead of Mz)

**Fix Required:**
Check `_parse_xvg_file` return for 4-column data and ensure dipole extraction uses correct column:
```python
elif 'dipole' in metric_name:
    # Dipole files have 4 columns: Mx, My, Mz(magnitude), |Mtot|
    # Original code uses Z (3rd data column = index 2)
    if equilibrated_data.ndim == 2 and equilibrated_data.shape[1] >= 3:
        dipole_z = equilibrated_data[:, 2]  # Mz column
        descriptor_dict[f'dipole_mu_{temp}'] = np.mean(dipole_z)
        descriptor_dict[f'dipole_std_{temp}'] = np.std(dipole_z)
    else:
        descriptor_dict[f'dipole_mu_{temp}'] = mu
        descriptor_dict[f'dipole_std_{temp}'] = std
```

---

## 📋 MEDIUM PRIORITY ISSUES

### 3. ⚠️ Index Group Creation Path Issue

**Location:** `abmelt_infer_pipeline/src/md_simulation.py` line 186

**Issue:** Using config dict path syntax incorrectly

**Current Code:**
```python
annotation = canonical_index(pdb=config['paths']['temp_dir'] / "processed.pdb")
gromacs.make_ndx(f=config['paths']['temp_dir'] / "processed.gro", ...)
```

**Problem:** 
- `config['paths']['temp_dir']` returns a STRING, not a Path object
- Cannot use `/` operator with strings
- This will cause a TypeError

**Fix Required:**
```python
from pathlib import Path

temp_dir = Path(config['paths']['temp_dir'])
annotation = canonical_index(pdb=str(temp_dir / "processed.pdb"))
gromacs.make_ndx(f=str(temp_dir / "processed.gro"), 
                 o=str(temp_dir / "index.ndx"), 
                 input=annotation)
```

Or simpler:
```python
annotation = canonical_index(pdb="processed.pdb")  # Already in working directory
gromacs.make_ndx(f="processed.gro", o="index.ndx", input=annotation)
```

---

## ✅ RESOLVED ISSUES (For Reference)

### ~~1. GROMACS hbond Command~~ ✅ FIXED
- **Was:** Using `gromacs.hbond()` (only 1 column output)
- **Fixed to:** `gromacs.hbond_legacy()` (2 columns: hbonds + contacts)
- **Location:** Line 185

### ~~2. RMSF Equilibration~~ ✅ FIXED
- **Was:** Applying time-based equilibration slicing to per-residue data
- **Fixed to:** Skip equilibration slicing for RMSF files
- **Location:** Lines 463-474

---

## 📊 COMPARISON SUMMARY

### Equilibration Handling

| Metric | Original `t0` | Your `eq_start_idx` | Status |
|--------|--------------|---------------------|---------|
| bonds | 2000 frames | `eq_time*1000/10` | ✅ Flexible |
| gyration | 2000 frames | `eq_time*1000/10` | ✅ Flexible |
| RMSD | 2000 frames | `eq_time*1000/10` | ✅ Flexible |
| SASA | 2000 frames | `eq_time*1000/10` | ✅ Flexible |
| **RMSF** | None (uses all) | Skip for RMSF | ✅ FIXED |
| **Potential** | None (index only) | Should skip | ⚠️ Need to skip |
| **Dipole** | 2000 frames | `eq_time*1000/10` | ✅ Flexible |

**Note:** Original code hardcodes `t0=2000` frames = 20,000 ps = 20 ns (for 10 ps/frame).  
Your code is MORE FLEXIBLE - just set `equilibration_time: 20` in config for production.

---

## 🎯 ACTION ITEMS

### Priority 1 (Critical - Wrong Values)
- [ ] Fix potential feature extraction to use specific radius indices
- [ ] Verify and fix dipole column extraction

### Priority 2 (Bugs - Will Fail)
- [ ] Fix index group creation path concatenation

### Priority 3 (Configuration)
- [ ] For production runs, set `equilibration_time: 20` to match paper
- [ ] Consider adding validation to ensure `simulation_time > equilibration_time + block_length`

### Priority 4 (Testing)
- [ ] Test potential features match original values
- [ ] Test dipole features match original values
- [ ] Verify all 348 features are generated correctly

---

## 📝 NOTES

1. **Your implementation is more flexible** with configurable equilibration time vs hardcoded values
2. **Parser structure is good** but needs column handling fixes for edge cases
3. **Most GROMACS commands match** the original implementation
4. **Index groups** are created the same way (using `canonical_index` from original code)
5. **For testing with 2ns simulations**: Use `use_dummy_s2: true` for order parameter features

---

## 🔗 REFERENCES

- Original code: `/workspace/antibody-developability-abmelt/AbMelt/src/anaylze.py`
- Your code: `/workspace/antibody-developability-abmelt/abmelt_infer_pipeline/src/compute_descriptors.py`
- Paper config: 100ns simulation, 20ns equilibration, block_length=[2.5, 25]
- Test config: 2ns simulation, 1ns equilibration, block_length=[2.5, 25], use_dummy_s2=true

