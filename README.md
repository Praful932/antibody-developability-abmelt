# antibody-developability-abmelt

A re-implementation of the [AbMelt](https://www.nature.com/articles/s41467-023-44113-3) pipeline for predicting antibody thermal stability, built to benchmark the method on a public dataset and reproduce holdout results end-to-end.

This was an ongoing collaboration with the [HuggingFace Science Community.](https://discord.gg/2PP6Vqpmze) **The project is currently parked due to compute resource constraints.**

---

## What is AbMelt?

AbMelt predicts three thermal stability metrics for an antibody given its structure:

| Target | Description |
|--------|-------------|
| **Tm** | Melting temperature |
| **Tmon** | Onset melting temperature |
| **Tagg** | Aggregation temperature |

The pipeline runs MD simulations (GROMACS) at 300K, 350K, and 400K, extracts structural descriptors (RMSF, SASA, radius of gyration, order parameters), and feeds them into trained sklearn models (Random Forest, KNN, ElasticNet).

---

## What's different from the original

The original AbMelt codebase was not easily reproducible. This re-implementation:

- Rebuilds the full inference pipeline from scratch in `abmelt_infer_pipeline/`
- Accepts input as amino acid sequences (structure generated via ImmunBuilder) or an existing PDB file
- Adds skip flags (`--skip-structure`, `--skip-md`, etc.) to resume from any intermediate step
- Integrates experiment tracking via HuggingFace datasets and HF Jobs for remote runs
- Validates against the 4 publicly available holdout antibodies (daclizumab, sirukumab, epratuzumab, sifalimumab)

---

## Status

Parked. A successful end-to-end run requires ~100ns MD simulations per temperature, which demands significant GPU compute.

---

## Code

All implementation lives in [`abmelt_infer_pipeline/`](./abmelt_infer_pipeline/).

```bash
# Run inference from a PDB file
python infer.py --pdb data/abmelt/public_pdbs/daclizumab.pdb \
                --name daclizumab \
                --config configs/paper_config.yaml

# Run inference from sequences
python infer.py --h "<heavy_chain_seq>" --l "<light_chain_seq>" \
                --name antibody_name \
                --config configs/paper_config.yaml
```
