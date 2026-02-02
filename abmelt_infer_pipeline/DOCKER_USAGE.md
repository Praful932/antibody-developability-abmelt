# Docker Usage Guide

This document explains how to use the AbMelt Docker image for both RunPod and Hugging Face Jobs.

## Building the Image

```bash
cd abmelt_infer_pipeline

# Build with default CUDA compute capability (89 for RTX 4090, A100, H100)
docker build -t praful932/abmelt-pipeline:latest .

# Build for specific GPU architecture
docker build --build-arg CUDA_TARGET_COMPUTE=75 -t praful932/abmelt-pipeline:latest .
```

**CUDA Compute Capabilities:**
- 75: RTX 2080, T4
- 80: A100
- 86: RTX 3090, A6000
- 89: RTX 4090, L4, L40

## Push to Docker Hub

```bash
docker login
docker push praful932/abmelt-pipeline:latest

# Also tag with version
docker tag praful932/abmelt-pipeline:latest praful932/abmelt-pipeline:v1.0
docker push praful932/abmelt-pipeline:v1.0
```

## Usage: RunPod

### 1. Create RunPod Template

In RunPod console:
- **Container Image**: `praful932/abmelt-pipeline:latest`
- **Container Start Command**: `/runpod_start.sh`
- **Environment Variables**:
  ```
  REPO_URL=https://github.com/Praful932/antibody-developability-abmelt
  GIT_BRANCH=main
  WORK_DIR=/workspace
  ```

### 2. Start Pod

When your pod starts:
- Latest code is automatically cloned to `/workspace/repo/`
- Jupyter and SSH are available via RunPod UI
- Conda environment `abmelt-3-11-env` is ready

### 3. Run Inference

Access Jupyter or SSH and run:
```bash
cd /workspace/repo/abmelt_infer_pipeline
conda activate abmelt-3-11-env
python infer.py --pdb antibody.pdb --config configs/paper_config.yaml
```

### 4. Using Specific Branch or Commit

Set environment variables in RunPod template:
```bash
# Use development branch
GIT_BRANCH=dev

# Use specific commit (for reproducibility)
GIT_COMMIT=abc123def456
```

## Usage: Hugging Face Jobs

### 1. Prerequisites

```bash
pip install huggingface_hub
huggingface-cli login
```

### 2. Run a Job

```python
from huggingface_hub import run_job

job = run_job(
    image="praful932/abmelt-pipeline:latest",
    command=[
        "/hf_job_run.sh",
        "python", "infer.py",
        "--pdb", "antibody.pdb",
        "--config", "configs/paper_config.yaml"
    ],
    flavor="a10g-large",
    timeout="6h",
    env={
        "REPO_URL": "https://github.com/Praful932/antibody-developability-abmelt",
        "GIT_BRANCH": "main",
        "WORK_DIR": "/workspace"
    }
)

print(f"Job URL: {job.url}")
print(f"Job ID: {job.id}")
```

### 3. Using the Wrapper Script

Use the provided `run_hf_job.py` script for easier job submission:

```bash
# Quick test
python run_hf_job.py --pdb antibody.pdb --config testing_config.yaml --flavor cpu-basic

# Full pipeline on GPU
python run_hf_job.py --pdb antibody.pdb --config paper_config.yaml --flavor a10g-large

# Use development branch
python run_hf_job.py --pdb antibody.pdb --branch dev

# Pin to specific commit
python run_hf_job.py --pdb antibody.pdb --commit abc123def

# From sequences
python run_hf_job.py --sequences "EVQL..." "DIQMTQ..." --flavor a10g-large

# Monitor job progress
python run_hf_job.py --pdb antibody.pdb --monitor
```

### 4. Monitor Job

```python
from huggingface_hub import inspect_job, fetch_job_logs

# Check status
status = inspect_job(job.id)
print(f"Status: {status.status.stage}")

# View logs
for log in fetch_job_logs(job.id):
    print(log)
```

## Environment Variables

All three scripts (`/clone_repo.sh`, `/runpod_start.sh`, `/hf_job_run.sh`) support these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `REPO_URL` | `https://github.com/Praful932/antibody-developability-abmelt` | Git repository URL |
| `GIT_BRANCH` | `main` | Branch to checkout |
| `GIT_COMMIT` | (empty) | Specific commit SHA (overrides `GIT_BRANCH`) |
| `WORK_DIR` | `/workspace` | Working directory for code |

## Helper Scripts in Container

The image includes three helper scripts:

### `/clone_repo.sh`
Clones or updates the repository based on environment variables.

### `/runpod_start.sh`
RunPod-specific startup script:
1. Clones/updates repository
2. Starts RunPod services (Jupyter, SSH)

### `/hf_job_run.sh`
HF Jobs wrapper script:
1. Clones/updates repository
2. Activates conda environment
3. Executes the provided command

## Local Testing

Test the image locally:

```bash
# Test RunPod startup
docker run --rm -it \
  -e REPO_URL=https://github.com/Praful932/antibody-developability-abmelt \
  -e GIT_BRANCH=main \
  praful932/abmelt-pipeline:latest \
  /runpod_start.sh

# Test HF Jobs execution
docker run --rm -it \
  -e REPO_URL=https://github.com/Praful932/antibody-developability-abmelt \
  -e GIT_BRANCH=main \
  praful932/abmelt-pipeline:latest \
  /hf_job_run.sh python infer.py --help

# Test with GPU
docker run --rm -it --gpus all \
  -e REPO_URL=https://github.com/Praful932/antibody-developability-abmelt \
  praful932/abmelt-pipeline:latest \
  /hf_job_run.sh python -c "import torch; print(torch.cuda.is_available())"
```

## Troubleshooting

### Code not updating
If you're not seeing latest changes:
1. Check the git commit shown in logs
2. Verify `GIT_BRANCH` or `GIT_COMMIT` environment variable
3. For RunPod: restart the pod to trigger fresh clone

### Conda environment not activated
The helper scripts automatically activate the environment. If running commands manually:
```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate abmelt-3-11-env
```

### GROMACS GPU not working
Check CUDA compute capability matches your GPU. Rebuild with correct `--build-arg CUDA_TARGET_COMPUTE=XX`.

### Out of memory
Use a smaller simulation configuration or request more GPU memory:
- Testing: `configs/testing_config.yaml` (2 ns)
- Full: `configs/paper_config.yaml` (100 ns)

## Best Practices

1. **Version your base image**: Tag with versions (`v1.0`, `v1.1`) not just `latest`
2. **Pin commits for reproducibility**: Use `GIT_COMMIT` for production runs
3. **Use branches for development**: Set `GIT_BRANCH=dev` to test features
4. **Monitor costs on HF Jobs**: Set appropriate timeouts, use testing config for development
5. **Rebuild image only when dependencies change**: Code changes don't require rebuild!
