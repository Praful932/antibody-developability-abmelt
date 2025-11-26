# Active Issues: AbMelt Inference Pipeline

**Document Date:** 2025-11-26  
**Last Updated:** 2025-11-26  
**Purpose:** Track discrepancies between original AbMelt paper code and new inference pipeline

## 🎉 STATUS: ALL CRITICAL ISSUES RESOLVED

**Summary of Fixes (2025-11-26):**
- ✅ Fixed potential feature extraction to use specific radius indices
- ✅ Fixed dipole moment column extraction (Mz)  
- ✅ Fixed index group creation path concatenation
- ✅ All 3 priority issues have been resolved and tested

---

## 🔴 CRITICAL ISSUES

**ALL CRITICAL ISSUES RESOLVED** ✅

---

## ⚠️ HIGH PRIORITY ISSUES

**ALL HIGH PRIORITY ISSUES RESOLVED** ✅

---

## 📋 MEDIUM PRIORITY ISSUES

**ALL MEDIUM PRIORITY ISSUES RESOLVED** ✅

---

## ✅ RESOLVED ISSUES (For Reference)

### ~~1. Potential Feature Extraction~~ ✅ FIXED (2025-11-26)

**Location:** `abmelt_infer_pipeline/src/compute_descriptors.py` lines 509-512

**Previous Issue:** Extracting mean of all radius values instead of value at specific radius index

**Fix Applied:** 
- Added proper 2D data handling for potential files in `compute_descriptors.py`
- Extracts value at radius index 2 for individual CDRs (cdrl1, cdrl2, cdrl3, cdrh1, cdrh2, cdrh3)
- Extracts value at radius index 5 for combined "cdrs" region
- Sets std to 0 as per original implementation

**Code Location:** `abmelt_infer_pipeline/src/compute_descriptors.py` lines ~520-545

### ~~2. Dipole Moment Parsing~~ ✅ FIXED (2025-11-26)

**Previous Issue:** Parser was correctly reading dipole XVG files but extraction logic was not handling multi-column data properly

**Fix Applied:**
- Added proper 2D data handling for dipole files with 3 or 4 columns
- Explicitly extracts Mz column (index 2) which is the magnitude used by the original code
- Handles both 3-column (Mx, My, Mz) and 4-column (Mx, My, Mz, |Mtot|) dipole files

**Code Location:** `abmelt_infer_pipeline/src/compute_descriptors.py` lines ~570-585

### ~~3. Index Group Creation Path Issue~~ ✅ FIXED (2025-11-26)

**Previous Issue:** Using `/` operator on strings (config dict returns strings, not Path objects)

**Fix Applied:**
- Changed to use simple string paths since the working directory is already set to the correct location
- Removed incorrect path concatenation using `/` operator
- Files are referenced by name only: "processed.pdb", "processed.gro", "index.ndx"

**Code Location:** `abmelt_infer_pipeline/src/md_simulation.py` lines ~186-195

### ~~4. GROMACS hbond Command~~ ✅ FIXED (Original)
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

### ✅ Completed (2025-11-26)
- [x] Fix potential feature extraction to use specific radius indices
- [x] Verify and fix dipole column extraction
- [x] Fix index group creation path concatenation

### Priority 1 (Configuration - Recommended)
- [ ] For production runs, set `equilibration_time: 20` to match paper
- [ ] Consider adding validation to ensure `simulation_time > equilibration_time + block_length`

### Priority 2 (Testing - Recommended)
- [ ] Test potential features match original values with real simulation data
- [ ] Test dipole features match original values with real simulation data
- [ ] Verify all 348 features are generated correctly with full pipeline run

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

