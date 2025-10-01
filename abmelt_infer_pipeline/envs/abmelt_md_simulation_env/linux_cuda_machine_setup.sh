# image - runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
# GPU - NVIDIA GeForce RTX 2000 Ada

# assumes conda and poetry are already installed
# https://gist.github.com/Praful932/246173142223a0495565dcb7b163ab5d

# set -e  # Exit on any error

CUDA_TARGET_COMPUTE=89

echo "Starting setup..."
echo "Installing apt packages..."
echo "Working in: $(pwd)"
cd ./abmelt_infer_pipeline/envs/abmelt_md_simulation_env
ls -la
echo "Working in: $(pwd)"
apt update
apt upgrade -y
apt install -y build-essential cmake libfftw3-dev libopenmpi-dev openmpi-bin

echo "Downloading GROMACS..."
mkdir temp/
cd temp/
wget ftp://ftp.gromacs.org/pub/gromacs/gromacs-2024.tar.gz
tar -xzf gromacs-2024.tar.gz
cd gromacs-2024

echo "Building GROMACS..."
mkdir build
cd build

cmake .. \
  -DGMX_BUILD_OWN_FFTW=ON \
  -DGMX_GPU=CUDA \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DGMX_CUDA_TARGET_COMPUTE=${CUDA_TARGET_COMPUTE} \
  -DCMAKE_INSTALL_PREFIX=/usr/local/gromacs \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DGMX_HWLOC=ON \
  -DGMX_OPENMP=ON

echo "Installing GROMACS..."
make -j$(nproc)
make install

export PATH=/usr/local/gromacs/bin:$PATH
echo "Installed GROMACS at /usr/local/gromacs/bin"
echo "GROMACS version: $(gmx --version)"

cd ../../../
echo "Working in: $(pwd)"
ls -la
rm -rf temp/
rm -rf gromacs-2024.tar.gz

echo "Setting up conda environment..."
eval "$(conda shell.bash hook)"
conda create -n abmelt-3-11-env python=3.11 -y
conda activate abmelt-3-11-env

echo "Installing dependencies via conda..."
conda install -c conda-forge openmm pdbfixer -y
conda install -c bioconda anarci -y
conda install -c conda-forge libgcc-ng libstdcxx-ng -y

echo "Changing directory"
cd ../
echo "Working in: $(pwd)"

echo "Configuring Poetry to use conda environment..."
poetry config virtualenvs.create false
poetry env use $(which python)

echo "Installing dependencies via poetry..."
poetry install

echo "Testing installation..."

ls -la
echo "Working in: $(pwd)"
echo "Testing installation..."
python quick_test.py

echo "Done with setup!"
echo "Run inference eg: python infer.py --pdb AbMelt/public_tm/train_pdbs/alemtuzumab.pdb --name alemtuzumab.pdb"