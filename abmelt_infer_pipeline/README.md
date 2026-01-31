## Quick Commands

### Local Inference

1. **Quick test** - `python quick_test.py`

2. **Run inference locally**
    - Using PDB file:
      ```bash
      python infer.py --pdb "/path/to/antibody.pdb" --name "alemtuzumab" --config configs/testing_config.yaml
      ```
    - Using sequences:
      ```bash
      python infer.py --h "QVQLQESGPGLVRPSQTLSLTCTVSGFTFTDFYMNWVRQPPGRGLEWIGFIRDKAKGYTTEYNPSVKGRVTMLVDTSKNQFSLRLSSVTAADTAVYYCAREGHTAAPFDYWGQGSLVTVSS" --l "DIQMTQSPSSLSASVGDRVTITCKASQNIDKYLNWYQQKPGKAPKLLIYNTNNLQTGVPSRFSGSGSGTDFTFTISSLQPEDIATYYCLQHISRPRTFGQGTKVEIK" --name "alemtuzumab" --config configs/testing_config.yaml
      ```

missing features because of eq_time being set to 0 in testing_config
- all-temp_lamda_b=25_eq=20
- all-temp-sasa_core_mean_k=20_eq=20
- all-temp-sasa_core_std_k=20_eq=20
- 

### Experiment Tracking with Hugging Face Jobs

The pipeline includes experiment tracking that automatically saves results to Hugging Face datasets.

#### Prerequisites

1. **Set up Hugging Face token**:
   ```bash
   export HF_TOKEN="hf_..."
   ```

2. **Set Docker image name** (if using custom image):
   ```bash
   export ABMELT_DOCKER_IMAGE="your-dockerhub/abmelt:latest"
   ```

3. **Set dataset names** (optional, defaults provided):
   ```bash
   export HF_MAIN_DATASET="username/abmelt-experiments"
   export HF_DETAILED_DATASET_PREFIX="username/abmelt-experiments-"
   ```

#### Submitting Experiments

**Basic submission**:
```bash
python submit_experiment.py \
  --name "alemtuzumab" \
  --heavy "QVQLQESGPGLVRPSQTLSLTCTVSGFTFTDFYMNWVRQPPGRGLEWIGFIRDKAKGYTTEYNPSVKGRVTMLVDTSKNQFSLRLSSVTAADTAVYYCAREGHTAAPFDYWGQGSLVTVSS" \
  --light "DIQMTQSPSSLSASVGDRVTITCKASQNIDKYLNWYQQKPGKAPKLLIYNTNNLQTGVPSRFSGSGSGTDFTFTISSLQPEDIATYYCLQHISRPRTFGQGTKVEIK" \
  --config configs/testing_config.yaml
```

**With custom hardware and timeout**:
```bash
python submit_experiment.py \
  --name "test_antibody" \
  --heavy "QVQLQESGPGLVR..." \
  --light "DIQMTQSPSSLSA..." \
  --config configs/paper_config.yaml \
  --flavor a100-large \
  --timeout 4h \
  --description "Testing longer simulation time"
```

**Available hardware flavors**:
- CPU: `cpu-basic`, `cpu-upgrade`
- GPU: `t4-small`, `t4-medium`, `l4x1`, `l4x4`, `a10g-small`, `a10g-large`, `a10g-largex2`, `a10g-largex4`, `a100-large` (default)

#### Monitoring Jobs

**Check job status**:
```python
from huggingface_hub import inspect_job
job_info = inspect_job(job_id="your_job_id")
print(job_info.status.stage)
```

**View logs**:
```python
from huggingface_hub import fetch_job_logs
for log in fetch_job_logs(job_id="your_job_id"):
    print(log)
```

#### Accessing Results

**Load main predictions dataset**:
```python
from datasets import load_dataset

# Load all experiments
ds = load_dataset("username/abmelt-experiments", split="train")

# Filter by experiment ID
exp_results = ds.filter(lambda x: x["experiment_id"] == "exp_20260130_123456")

# Filter by antibody name
antibody_results = ds.filter(lambda x: x["antibody_name"] == "alemtuzumab")

# Compare experiments with different configs
high_salt = ds.filter(lambda x: x["salt_concentration"] == 300)
low_salt = ds.filter(lambda x: x["salt_concentration"] == 0)
```

**Load detailed results** (descriptors, logs, config):
```python
from huggingface_hub import hf_hub_download

# Download descriptors CSV
descriptors_path = hf_hub_download(
    repo_id="username/abmelt-experiments-exp_20260130_123456",
    filename="descriptors.csv"
)

# Download config used for this run
config_path = hf_hub_download(
    repo_id="username/abmelt-experiments-exp_20260130_123456",
    filename="config.yaml"
)

# Download logs
log_path = hf_hub_download(
    repo_id="username/abmelt-experiments-exp_20260130_123456",
    filename="inference.log"
)
```

#### Experiment Tracking Features

- **Automatic parameter tracking**: All config parameters are logged (temperatures, simulation time, force field, etc.)
- **Hybrid dataset structure**: 
  - Main dataset: One row per experiment with predictions + metadata (fast queries)
  - Detailed datasets: Full descriptors, logs, configs per experiment (deep inspection)
- **Reproducibility**: Config hash enables quick comparison of identical configs
- **Error handling**: Failed experiments are logged with error messages
- **Git tracking**: Experiment metadata includes Git commit and branch

#### Dataset Schema

**Main Predictions Dataset** (`username/abmelt-experiments`):
- `experiment_id`: Unique identifier (exp_YYYYMMDD_HHMMSS)
- `antibody_name`: User-provided name
- `timestamp`: Job start time
- `heavy_chain`, `light_chain`: Input sequences
- `tagg`, `tm`, `tmon`: Predictions
- `job_id`: HF Jobs ID
- `status`: success/failed/timeout
- `duration_seconds`: Runtime
- `config_hash`: MD5 hash of config
- All config parameters (temperatures, simulation_time, force_field, etc.)

**Detailed Results Dataset** (`username/abmelt-experiments-{experiment_id}`):
- `descriptors.csv`: Full descriptor DataFrame (100+ features)
- `descriptors.pkl`: Pickled DataFrame
- `config.yaml`: Exact config used
- `inference.log`: Complete job logs
- `run_metadata.json`: Git info, timestamps, etc.

## Notes

**Missing features in testing_config** (due to eq_time=0):
- `all-temp_lamda_b=25_eq=20`
- `all-temp-sasa_core_mean_k=20_eq=20`
- `all-temp-sasa_core_std_k=20_eq=20`

These features require `equilibration_time >= 20` and are available in `paper_config.yaml`.