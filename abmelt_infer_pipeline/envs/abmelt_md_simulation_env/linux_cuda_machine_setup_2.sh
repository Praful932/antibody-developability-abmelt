# image - runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
# GPU - NVIDIA GeForce RTX 2000 Ada

# assumes conda and poetry are already installed
# https://gist.github.com/Praful932/246173142223a0495565dcb7b163ab5d

# set -e  # Exit on any error

export POETRY_REQUESTS_TIMEOUT=600

echo "Configuring Poetry to use conda environment..."
poetry config virtualenvs.create false
echo "Currently used python env: $(which python)"

echo "Installing dependencies via poetry..."
poetry install

echo "Testing installation..."

cd ../../

ls -la
echo "Working in: $(pwd)"
echo "Testing installation..."
python quick_test.py

echo "Done with setup!"
echo "Run inference eg: python infer.py --pdb ../AbMelt/public_tm/train_pdbs/alemtuzumab.pdb --name alemtuzumab.pdb"
