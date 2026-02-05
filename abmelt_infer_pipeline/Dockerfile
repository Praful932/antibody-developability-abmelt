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
    && wget -q https://repo.anaconda.com/miniconda/Miniconda3-py312_25.1.1-0-Linux-x86_64.sh -O ${MINICONDA_HOME}/miniconda.sh \
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
# Build GROMACS with GPU support
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
ENV PYTHONUNBUFFERED=1

# ===========================================
# Copy dependency files only (for Docker cache optimization)
# ===========================================
ARG REPO_URL=https://github.com/Praful932/antibody-developability-abmelt

# Temporarily clone repo ONLY to get dependency files
RUN echo "=== Fetching dependency files ===" \
    && git clone --depth=1 ${REPO_URL} /tmp/repo-for-deps \
    && mkdir -p /opt/build \
    && cp /tmp/repo-for-deps/abmelt_infer_pipeline/envs/abmelt_md_simulation_env/pyproject.toml /opt/build/ \
    && cp /tmp/repo-for-deps/abmelt_infer_pipeline/envs/abmelt_md_simulation_env/poetry.lock /opt/build/ \
    && rm -rf /tmp/repo-for-deps \
    && echo "=== Dependency files copied ==="

WORKDIR /opt/build

# ===========================================
# Create conda env and install dependencies (optimized)
# ===========================================
RUN echo "=== Setting up conda environment ===" \
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
    && echo "=== Installing dependencies with Poetry ===" \
    && poetry install --no-root -vvv \
    && echo "=== Poetry installation completed successfully ==="

# ===========================================
# Pre-download ImmuneBuilder antibody weights
# ===========================================
RUN echo "=== Downloading ImmuneBuilder antibody weights ===" \
    && mkdir -p /opt/immunebuilder_weights \
    && cd /opt/immunebuilder_weights \
    && wget -q --show-progress \
        "https://zenodo.org/record/7258553/files/antibody_model_1?download=1" \
        -O antibody_model_1 \
    && wget -q --show-progress \
        "https://zenodo.org/record/7258553/files/antibody_model_2?download=1" \
        -O antibody_model_2 \
    && wget -q --show-progress \
        "https://zenodo.org/record/7258553/files/antibody_model_3?download=1" \
        -O antibody_model_3 \
    && wget -q --show-progress \
        "https://zenodo.org/record/7258553/files/antibody_model_4?download=1" \
        -O antibody_model_4 \
    && echo "=== Verifying downloaded weights ===" \
    && for model in antibody_model_*; do \
        if [ ! -s "$model" ]; then \
            echo "ERROR: $model is empty or missing"; \
            exit 1; \
        fi; \
        echo "✓ $model: $(stat -c%s "$model" 2>/dev/null || stat -f%z "$model") bytes"; \
    done \
    && chmod -R 755 /opt/immunebuilder_weights \
    && echo "=== Antibody weights downloaded successfully ==="

# ===========================================
# Create helper script for git operations (runtime clone)
# ===========================================
RUN cat > /clone_repo.sh << 'EOF'
#!/bin/bash
set -e

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate abmelt-3-11-env

# Git repository configuration
REPO_URL="${REPO_URL:-https://github.com/Praful932/antibody-developability-abmelt}"
GIT_BRANCH="${GIT_BRANCH:-main}"
GIT_COMMIT="${GIT_COMMIT:-}"
WORK_DIR="${WORK_DIR:-/workspace}"

mkdir -p ${WORK_DIR}
cd ${WORK_DIR}

echo "=== Cloning repository from ${REPO_URL} ==="
if [ -d "repo/.git" ]; then
    echo "Repository exists, pulling latest changes..."
    cd repo
    git fetch origin
    if [ -n "${GIT_COMMIT}" ]; then
        echo "Checking out commit: ${GIT_COMMIT}"
        git checkout ${GIT_COMMIT}
    else
        echo "Checking out branch: ${GIT_BRANCH}"
        git checkout ${GIT_BRANCH}
        git pull origin ${GIT_BRANCH}
    fi
else
    if [ -n "${GIT_COMMIT}" ]; then
        git clone ${REPO_URL} repo
        cd repo
        echo "Checking out commit: ${GIT_COMMIT}"
        git checkout ${GIT_COMMIT}
    else
        git clone --depth=1 --branch=${GIT_BRANCH} ${REPO_URL} repo
        cd repo
    fi
fi

echo "=== Repository ready at $(pwd) ==="
echo "=== Current commit: $(git rev-parse HEAD) ==="
EOF

RUN chmod +x /clone_repo.sh

# ===========================================
# Create RunPod-compatible start script
# ===========================================
RUN cat > /runpod_start.sh << 'EOF'
#!/bin/bash
set -e

echo "=== RunPod Startup ==="

# Clone/update repository
/clone_repo.sh

# Start RunPod services (Jupyter, SSH, etc.)
if [ -f /start.sh ]; then
    echo "=== Starting RunPod services ==="
    exec /start.sh
else
    echo "⚠️  /start.sh not found, starting bash"
    exec /bin/bash
fi
EOF

RUN chmod +x /runpod_start.sh

# ===========================================
# Create HF Jobs wrapper script
# ===========================================
RUN cat > /hf_job_run.sh << 'EOF'
#!/bin/bash
set -e

echo "=== HF Jobs Execution ==="

# Clone/update repository
/clone_repo.sh

# Navigate to pipeline directory
cd ${WORK_DIR:-/workspace}/repo/abmelt_infer_pipeline

# Activate environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate abmelt-3-11-env

# Ensure Python output is unbuffered
export PYTHONUNBUFFERED=1

# Execute the command
echo "=== Executing: $@ ==="
exec "$@"
EOF

RUN chmod +x /hf_job_run.sh

# ===========================================
# Final cleanup and success message
# ===========================================
RUN echo "=== Final cleanup ===" \
    && conda clean -afy \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /opt/build \
    && echo "=== Docker build completed successfully! ==="

# ===========================================
# Set working directory and default command
# NO ENTRYPOINT - flexible for both RunPod and HF Jobs
# ===========================================
WORKDIR /workspace
CMD ["/bin/bash"]