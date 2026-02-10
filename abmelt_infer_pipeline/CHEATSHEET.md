# AbMelt Pipeline Cheatsheet

Quick reference for common commands and workflows.

---

## 1. Loading Environment Variables from .env File

Create a `.env` file with your environment variables:

```bash
# .env file contents
HF_TOKEN=hf_your_token_here
ABMELT_DOCKER_IMAGE=your-dockerhub/abmelt:latest
REPO_URL=https://github.com/Praful932/antibody-developability-abmelt
HF_MAIN_DATASET=username/abmelt-experiments
HF_DETAILED_DATASET_PREFIX=username/abmelt-experiments-
```

Load the environment variables:

```bash
# Load all variables from .env into current shell
set -a  # automatically export all variables
source .env
set +a  # turn off automatic export
```

Verify the variables are loaded:
  
```bash
echo $HF_TOKEN
echo $ABMELT_DOCKER_IMAGE
```

## 2. Submitting an Experiment with Daclizumab

### Example: Submit daclizumab with testing config

```bash
# Navigate to the pipeline directory
cd abmelt_infer_pipeline

# Submit experiment to Hugging Face Jobs
python submit_experiment.py \
  --name "daclizumab" \
  --heavy "QVQLVQSGAEVKKPGSSVKVSCKASGYTFTSYRMHWVRQAPGQGLEWIGYINPSTGYTEYNQKFKDKATITADESTNTAYMELSSLRSEDTAVYYCARGGGVFDYWGQGTLVTVSS" \
  --light "DIQMTQSPSTLSASVGDRVTITCSASSSISYMHWYQQKPGKAPKLLIYTTSNLASGVPARFSGSGSGTDFTLTISSLQPDDFATYYCHQRSTYPLTFGQGTKVEVK" \
  --config configs/testing_config.yaml \
  --flavor a10g-large \
  --timeout 4h \
  --description "Testing daclizumab with testing config (2ns simulation)"
```

### Example: Override simulation_time without modifying config

```bash
# Use testing config but run for only 0.5ns (ultra-fast test)
python submit_experiment.py \
  --name "daclizumab" \
  --heavy "QVQLVQSGAEVKKPGSSVKVSCKASGYTFTSYRMHWVRQAPGQGLEWIGYINPSTGYTEYNQKFKDKATITADESTNTAYMELSSLRSEDTAVYYCARGGGVFDYWGQGTLVTVSS" \
  --light "DIQMTQSPSTLSASVGDRVTITCSASSSISYMHWYQQKPGKAPKLLIYTTSNLASGVPARFSGSGSGTDFTLTISSLQPDDFATYYCHQRSTYPLTFGQGTKVEVK" \
  --config configs/testing_config.yaml \
  --simulation-time 0.5 \
  --flavor a10g-large \
  --timeout 2h

# Use testing config but run for 10ns (more data, still fast)
python submit_experiment.py \
  --name "daclizumab" \
  --heavy "QVQLVQSGAEVKKPGSSVKVSCKASGYTFTSYRMHWVRQAPGQGLEWIGYINPSTGYTEYNQKFKDKATITADESTNTAYMELSSLRSEDTAVYYCARGGGVFDYWGQGTLVTVSS" \
  --light "DIQMTQSPSTLSASVGDRVTITCSASSSISYMHWYQQKPGKAPKLLIYTTSNLASGVPARFSGSGSGTDFTLTISSLQPDDFATYYCHQRSTYPLTFGQGTKVEVK" \
  --config configs/testing_config.yaml \
  --simulation-time 10 \
  --flavor a10g-large \
  --timeout 8h

# Use paper config but run for shorter time (50ns instead of 100ns)
python submit_experiment.py \
  --name "daclizumab" \
  --heavy "QVQLVQSGAEVKKPGSSVKVSCKASGYTFTSYRMHWVRQAPGQGLEWIGYINPSTGYTEYNQKFKDKATITADESTNTAYMELSSLRSEDTAVYYCARGGGVFDYWGQGTLVTVSS" \
  --light "DIQMTQSPSTLSASVGDRVTITCSASSSISYMHWYQQKPGKAPKLLIYTTSNLASGVPARFSGSGSGTDFTLTISSLQPDDFATYYCHQRSTYPLTFGQGTKVEVK" \
  --config configs/paper_config.yaml \
  --simulation-time 50 \
  --flavor a10g-large \
  --timeout 12h
```
