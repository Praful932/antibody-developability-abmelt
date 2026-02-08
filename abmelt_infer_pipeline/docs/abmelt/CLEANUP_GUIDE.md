# Temp Directory Cleanup Guide

## Problem

The `temp` directory accumulates many intermediate GROMACS files during MD simulation and descriptor computation. Most of these files are only needed during computation and can be safely deleted afterward, significantly reducing storage requirements.

## Solution

A cleanup utility has been created that:
1. Identifies required files needed for pipeline functionality
2. Removes intermediate files that can be regenerated or are no longer needed
3. Can be run manually or integrated into the pipeline

## Required Files

The pipeline needs these files to function:

### Critical Files (Cannot Delete)
- **Structure files**: `{antibody_name}.pdb`, `processed.pdb`, `processed.gro`, `topol.top`, `index.ndx`
- **Final trajectories**: `md_final_{temp}.xtc`, `md_final_{temp}.gro`, `md_{temp}.tpr` (per temperature)
- **Descriptors**: `descriptors.csv`/`descriptors.pkl`, all `*.xvg` files, `res_sasa_{temp}.np`, `sconf_{temp}.log`

### Optional Files (Can Delete but Useful)
- Order parameter CSVs: `order_s2_*.csv`, `order_lambda_*.csv` (can be regenerated)

### Files That Can Be Deleted
- **GROMACS backup files**: `#*#` pattern (e.g., `#aver.xvg.1#`, `#aver.xvg.10#`) - These are backups created when GROMACS overwrites files
- Intermediate trajectories: `md_whole_*.xtc`, `md_nopbcjump_*.xtc`, `md_{temp}.xtc`
- Equilibration files: `nvt_*.gro`, `nvt_*.xtc`, `npt_*.gro`, `npt_*.xtc`, etc.
- System setup: `box.gro`, `solv.gro`, `solv_ions.gro`, `em.gro`, etc.
- Checkpoints: `*.cpt` files
- Energy files: `*.edr` files
- Logs: `md_*.log` (except `sconf_*.log`)

## Usage

### Manual Cleanup

```bash
# Dry run to see what would be deleted (recommended first step)
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

### Automatic Cleanup (Integrated)

The cleanup is now integrated into the pipeline. Configure it in your YAML config:

```yaml
performance:
  cleanup_temp: true             # Enable automatic cleanup
  cleanup_after: "inference"      # When to cleanup: "descriptors" or "inference"
  delete_order_params: false     # Also delete order param CSVs (default: keep)
```

The cleanup runs automatically after inference completes (if `cleanup_temp: true`).

## File Size Impact

**Before cleanup**: ~10-50 GB per antibody (depending on simulation length)
**After cleanup**: ~1-2 GB per antibody

**Savings**: 80-95% reduction in storage requirements

## Safety Features

1. **Dry run mode**: Always test with `--dry-run` first
2. **Required file protection**: Never deletes files needed for pipeline
3. **Suspicious file detection**: Warns about files not in known patterns
4. **Non-fatal errors**: Cleanup failures don't stop the pipeline

## When to Clean Up

### Option 1: After Inference (Recommended)
- Cleanup runs automatically after predictions are made
- All required files preserved for re-running inference
- Cannot re-run descriptor computation without re-running MD

### Option 2: After Descriptors
- Set `cleanup_after: "descriptors"` in config
- Preserves ability to re-run inference
- Cannot re-compute descriptors without re-running MD

### Option 3: Manual Only
- Set `cleanup_temp: false` in config
- Run cleanup utility manually when needed
- Full control over when cleanup happens

## Troubleshooting

### "Missing required files" error after cleanup
- Check that cleanup didn't accidentally delete required files
- Verify antibody name and temperatures match your run
- Restore from backup if needed

### Cleanup didn't delete expected files
- Check file patterns match your simulation setup
- Some files may be in use (close any open file handles)
- Check file permissions

### Want to keep more files
- Modify `REQUIRED_FILES` in `cleanup_temp_files.py`
- Or set `cleanup_temp: false` and clean manually

## Files Reference

See `src/REQUIRED_FILES.md` for complete list of required vs. intermediate files.

