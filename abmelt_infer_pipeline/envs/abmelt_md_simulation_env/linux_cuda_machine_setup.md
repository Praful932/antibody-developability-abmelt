tested on machine RTX 2000 Ada runpod

- `apt install gromacs`
- check if in path `gmx --version`
- use poetry & conda
- `conda create -n abmelt-3-11-env python=3.11 -y`
- `conda activate abmelt-3-11-env`
- `conda install -c conda-forge openmm pdbfixer -y`
- `conda install -c bioconda anarci -y` 
- `conda install -c conda-forge libgcc-ng libstdcxx-ng`
- `poetry install`
- `python quick_test.py`
- `python infer.py --pdb AbMelt/public_tm/train_pdbs/alemtuzumab.pdb --name alemtuzumab.pdb`



apt upgrade
apt install build-essential cmake libfftw3-dev libopenmpi-dev openmpi-bin

wget ftp://ftp.gromacs.org/pub/gromacs/gromacs-2024.tar.gz
tar -xzf gromacs-2024.tar.gz
cd gromacs-2024

mkdir build
cd build

cmake .. \
  -DGMX_BUILD_OWN_FFTW=ON \
  -DGMX_GPU=CUDA \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DGMX_CUDA_TARGET_COMPUTE=89 \
  -DCMAKE_INSTALL_PREFIX=/usr/local/gromacs \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DGMX_HWLOC=ON \
  -DGMX_OPENMP=ON

make -j$(nproc)
make install

echo 'source /usr/local/gromacs/bin/GMXRC' >> ~/.bashrc
source ~/.bashrc


- `import gromacs; gromacs.config.setup()`