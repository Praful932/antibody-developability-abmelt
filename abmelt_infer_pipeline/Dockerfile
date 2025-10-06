FROM runpod/pytorch:1.0.1-cu1281-torch280-ubuntu2404

# ===========================================
# System dependencies (cached layer)
# ===========================================
RUN echo "=== Installing system dependencies ===" \
    && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    git \
    build-essential \
    cmake \
    libfftw3-dev \
    libopenmpi-dev \
    openmpi-bin \
    pipx \
    bash \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && echo "=== System dependencies installed successfully ==="

# ===========================================
# Install Miniconda (cached layer)
# ===========================================
ENV MINICONDA_HOME=/root/miniconda3
ENV PATH=${MINICONDA_HOME}/bin:${PATH}

RUN echo "=== Downloading and installing Miniconda ===" \
    && mkdir -p ${MINICONDA_HOME} \
    && wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${MINICONDA_HOME}/miniconda.sh \
    && bash ${MINICONDA_HOME}/miniconda.sh -b -u -p ${MINICONDA_HOME} \
    && rm -f ${MINICONDA_HOME}/miniconda.sh \
    && conda init bash \
    && conda clean -afy \
    && echo "=== Miniconda installed successfully ==="

# ===========================================
# Install Poetry (cached layer)
# ===========================================
ENV PIPX_BIN_DIR=/root/.local/bin
ENV PATH=${PIPX_BIN_DIR}:${PATH}

RUN echo "=== Installing Poetry via pipx ===" \
    && pipx ensurepath \
    && pipx install poetry==2.2.1 \
    && echo 'export PATH="$PATH:/root/.local/bin"' >> /root/.bashrc \
    && echo "=== Poetry installed successfully ==="

# Use bash login shell for subsequent RUN to make conda activation easier
SHELL ["/bin/bash", "-lc"]

# ===========================================
# Multi-stage GROMACS build (optimized)
# ===========================================
ARG CUDA_TARGET_COMPUTE=89
ENV CUDA_TARGET_COMPUTE=${CUDA_TARGET_COMPUTE}

RUN echo "=== Starting GROMACS build with CUDA compute capability ${CUDA_TARGET_COMPUTE} ===" \
    && mkdir -p /tmp/gromacs-build \
    && cd /tmp/gromacs-build \
    && echo "=== Downloading GROMACS source ===" \
    && wget -q ftp://ftp.gromacs.org/pub/gromacs/gromacs-2024.tar.gz \
    && tar -xzf gromacs-2024.tar.gz \
    && cd gromacs-2024 \
    && mkdir build \
    && cd build \
    && echo "=== Configuring GROMACS with CMake ===" \
    && cmake .. \
      -DGMX_BUILD_OWN_FFTW=ON \
      -DGMX_GPU=CUDA \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_TARGET_COMPUTE} \
      -DGMX_GPU_PME=ON \
      -DGMX_GPU_UPDATE=ON \
      -DCMAKE_INSTALL_PREFIX=/usr/local/gromacs \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_CXX_COMPILER=g++ \
      -DGMX_HWLOC=ON \
      -DGMX_OPENMP=ON \
      -DCMAKE_CUDA_FLAGS="-use_fast_math" \
    && echo "=== Compiling GROMACS ===" \
    && make -j$(nproc) \
    && echo "=== Installing GROMACS ===" \
    && make install \
    && cd / \
    && rm -rf /tmp/gromacs-build \
    && echo "=== GROMACS build completed successfully ==="

ENV PATH=/usr/local/gromacs/bin:${PATH}

# ===========================================
# Clone repository (cached layer)
# ===========================================
ARG REPO_URL=https://github.com/Praful932/antibody-developability-abmelt
ARG APP_HOME=/opt/antibody-developability-abmelt

RUN echo "=== Cloning antibody-developability-abmelt repository ===" \
    && git clone --depth=1 ${REPO_URL} ${APP_HOME} \
    && echo "=== Repository cloned successfully ==="

WORKDIR ${APP_HOME}/abmelt_infer_pipeline/envs/abmelt_md_simulation_env

# ===========================================
# Create conda env and install dependencies (optimized)
# ===========================================
RUN echo "=== Setting up conda environment ===" \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && eval "$(conda shell.bash hook)" \
    && conda create -n abmelt-3-11-env python=3.11 -y \
    && conda activate abmelt-3-11-env \
    && echo "=== Installing all conda dependencies in one go ===" \
    && conda install -c conda-forge -c bioconda \
    openmm \
    pdbfixer \
    anarci \
    libgcc-ng \
    libstdcxx-ng \
    -y \
    && conda clean -afy \
    && echo "=== Conda environment setup completed successfully ==="

# ===========================================
# Configure Poetry and install dependencies
# ===========================================
ENV POETRY_REQUESTS_TIMEOUT=300

RUN echo "=== Configuring Poetry and installing dependencies ===" \
    && eval "$(conda shell.bash hook)" \
    && conda activate abmelt-3-11-env \
    && poetry config virtualenvs.create false \
    && poetry config installer.max-workers 1 \
    && echo "=== Clearing Poetry cache ===" \
    && poetry cache clear --all . || true \
    && rm -rf $(poetry config cache-dir)/artifacts || true \
    && echo "=== Installing dependencies with Poetry ===" \
    && poetry install -vvv \
    && echo "=== Poetry installation completed successfully ==="

# ===========================================
# Final cleanup and success message
# ===========================================
RUN echo "=== Final cleanup ===" \
    && conda clean -afy \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && echo "=== Docker build completed successfully! ==="

# Set default command
CMD ["/bin/bash"]