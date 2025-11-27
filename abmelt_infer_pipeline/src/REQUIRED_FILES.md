# Required Files Reference

This document lists all files that must be preserved in the temp directory for the AbMelt inference pipeline to function correctly.

## File Categories

### 1. Structure Files (Always Required)
- `{antibody_name}.pdb` - Input structure
- `processed.pdb` - Processed structure (needed for re-indexing if index.ndx is missing)
- `processed.gro` - Processed GRO file (needed for re-indexing)
- `topol.top` - GROMACS topology file
- `index.ndx` - Index file defining CDR regions (critical for descriptor computation)

### 2. MD Simulation Final Outputs (Required for Descriptor Computation)
Per temperature (e.g., 300, 350, 400):
- `md_final_{temp}.xtc` - Final processed trajectory (PBC removed, no jumps)
- `md_final_{temp}.gro` - Final reference structure
- `md_{temp}.tpr` - Topology file for the production run

**Note:** These are the ONLY trajectory files needed after MD simulation completes. All intermediate trajectories (`md_whole_*.xtc`, `md_nopbcjump_*.xtc`, `md_{temp}.xtc`) can be deleted.

### 3. Descriptor Computation Outputs (Required for Inference)
- `descriptors.csv` or `descriptors.pkl` - Aggregated descriptor DataFrame
- `*.xvg` - All GROMACS descriptor files (needed if re-aggregating descriptors)
- `res_sasa_{temp}.np` - Residue-level SASA data per temperature
- `sconf_{temp}.log` - Conformational entropy log per temperature

### 4. Order Parameter Files (Optional but Recommended)
- `order_s2_{temp}K_{block}_{start}.csv` - Order parameter CSVs (can be regenerated)
- `order_lambda_{block}_{start}.csv` - Lambda CSV (can be regenerated)

**Note:** These are useful for debugging but can be regenerated from trajectories if needed.

### 5. Model Inference Outputs
- `{antibody_name}_predictions.csv` - Prediction results

## Files That Can Be Safely Deleted

### Intermediate Trajectory Files
- `md_whole_*.xtc` - Intermediate PBC removal step
- `md_nopbcjump_*.xtc` - Intermediate PBC removal step  
- `md_{temp}.xtc` - Raw trajectory before processing
- `md_{temp}.gro` - Raw structure before processing

### Equilibration Files
- `nvt_*.gro`, `nvt_*.xtc`, `nvt_*.tpr`, `nvt_*.cpt`, `nvt_*.edr`, `nvt_*.log`
- `npt_*.gro`, `npt_*.xtc`, `npt_*.tpr`, `npt_*.cpt`, `npt_*.edr`, `npt_*.log`

### System Setup Intermediates
- `box.gro`, `solv.gro`, `solv_ions.gro`
- `em.gro`, `em.tpr`, `em.edr`, `em.log`
- `ions.tpr`

### Covariance Analysis Intermediates
- `md_final_covar_*.xtc`
- `covar_*.trr`, `covar_*.xvg`
- `avg_covar*.pdb`, `covar_matrix_*.dat`

### Other Intermediates
- `#*#` - GROMACS backup files (created when overwriting existing files, e.g., `#aver.xvg.1#`)
- `*.pka` - PropKa output (not needed after protonation state is set)
- `*.cpt` - Checkpoint files (can be regenerated)
- `*.edr` - Energy files (not used in pipeline)
- `md_*.log` - MD run logs (except `sconf_*.log`)
- `*.mdp` - MDP files created in work dir (templates are in mdp/ directory)

## Usage

### Using the Cleanup Utility

```bash
# Dry run to see what would be deleted
python src/cleanup_temp_files.py run_data/run_2/temp \
    --antibody-name "my_antibody" \
    --temperatures 300 350 400 \
    --dry-run

# Actually delete intermediate files
python src/cleanup_temp_files.py run_data/run_2/temp \
    --antibody-name "my_antibody" \
    --temperatures 300 350 400

# Also delete order parameter CSVs
python src/cleanup_temp_files.py run_data/run_2/temp \
    --antibody-name "my_antibody" \
    --temperatures 300 350 400 \
    --delete-order-params
```

### Integration with Pipeline

The cleanup can be integrated into the pipeline by adding a cleanup step after descriptor computation or inference completes. This can be controlled via config:

```yaml
performance:
  cleanup_temp: true  # Already exists in config
  cleanup_after: "inference"  # When to cleanup: "descriptors" or "inference"
```

## File Size Estimates

Typical file sizes (for reference):
- `md_final_{temp}.xtc`: ~100-500 MB (depends on simulation length)
- `md_final_{temp}.gro`: ~1-5 MB
- `md_{temp}.tpr`: ~10-50 MB
- `*.xvg` files: ~1-10 MB each (many files)
- `descriptors.csv`: ~1-10 KB
- `res_sasa_{temp}.np`: ~100 KB - 1 MB

Intermediate files can easily total 10-50 GB for a single antibody run, while required files are typically <2 GB.

