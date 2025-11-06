#!/bin/bash --login
#SBATCH --job-name=x5k_qc
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16           # tune as needed
#SBATCH --mem=250G                   # bump if needed
#SBATCH --partition=compute          # change to your cluster partition
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -euo pipefail

########################
# USER CONFIG          #
########################

# Container image that has python, scanpy, anndata, umap-learn, scrublet (optional), etc.
CONTAINER=""

# Script & data locations
WORKDIR="${SLURM_SUBMIT_DIR}"                # where this .sh and the python script live
SCRIPT="${WORKDIR}/xenium5k_qc_cluster.py"   # full path to the Python script
XENIUM_ROOT="/path/to/all_xenium_runs"       # directory containing run subfolders
OUT_H5AD="${WORKDIR}/xc_5k_all_runs.h5ad"
FIGDIR="${WORKDIR}/figures_all"

# Behavior / clustering
CLUSTER_ON="hvg"         # hvg | all
N_HVG=3000
N_PCS=50
N_NEIGHBORS=30
RESOLUTION=0.3
RANDOM_SEED=0
GENE_MIN_CELLS=10

# Performance toggles
UMAP_ON_SAMPLE=150000    # 0 to disable subsample+transform; else fit on this many cells
USE_SCRUBLET=0           # 1 to run scrublet, 0 to skip (recommended for first pass on huge sets)

# Optional include/exclude run folder names (space-separated)
INCLUDE=""               # e.g. "RunA RunB"
EXCLUDE=""               # e.g. "BadRun"

# Per-experiment exports for easy overlay/round-trips
WRITE_PER_EXPERIMENT=1   # 1 = write per-experiment .h5ad and CSV; 0 = skip

# Container runtime selector (singularity or apptainer)
RUNTIME="singularity"    # set to "apptainer" if your site uses that command

########################
# ENV & BINDING        #
########################

module load singularity || true


mkdir -p "${WORKDIR}/.mplconfig" slurm "${FIGDIR}"

# Use all allocated CPUs inside the container (for BLAS/numba/pynndescent)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${WORKDIR}/.mplconfig"

# Bind mounts: add anything else you need (scratch, project dirs, tmp)
BIND_DIRS="${WORKDIR},${XENIUM_ROOT}"

# Choose runtime cmd
if [[ "${RUNTIME}" == "apptainer" ]]; then
  RUN="apptainer"
else
  RUN="singularity"
fi

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK} | Mem: ${SLURM_MEM_PER_NODE:-NA}MB | Partition: ${SLURM_JOB_PARTITION}"

########################
# Version banner (in container)
########################
${RUN} exec --bind "${BIND_DIRS}" "${CONTAINER}" \
  /usr/bin/env python3 - <<'PY'
try:
    import scanpy as sc, anndata as ad, umap, numpy as np
    print("Scanpy:", sc.__version__, "| AnnData:", ad.__version__, "| umap-learn:", umap.__version__)
except Exception as e:
    print("Version check error:", e)
PY

########################
# Build CLI args
########################

ARGS=( \
  --xenium-root "${XENIUM_ROOT}" \
  --out-h5ad "${OUT_H5AD}" \
  --figdir "${FIGDIR}" \
  --cluster-on "${CLUSTER_ON}" \
  --n-hvg "${N_HVG}" \
  --n-pcs "${N_PCS}" \
  --n-neighbors "${N_NEIGHBORS}" \
  --resolution "${RESOLUTION}" \
  --random-seed "${RANDOM_SEED}" \
  --gene-min-cells "${GENE_MIN_CELLS}" \
  --neighbors-metric cosine \
  --neighbors-method umap \
  --umap-suffix "_x5k" \
)

# Optional toggles
if [[ "${UMAP_ON_SAMPLE}" -gt 0 ]]; then
  ARGS+=( --umap-on-sample "${UMAP_ON_SAMPLE}" )
fi

if [[ "${USE_SCRUBLET}" -eq 0 ]]; then
  ARGS+=( --no-doublet )
fi

if [[ "${WRITE_PER_EXPERIMENT}" -eq 1 ]]; then
  ARGS+=( --write-per-experiment )
fi

# Include / exclude lists
if [[ -n "${INCLUDE}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=( --include ${INCLUDE} )
fi
if [[ -n "${EXCLUDE}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=( --exclude ${EXCLUDE} )
fi

echo "Launching:"
printf '  %q ' "${SCRIPT}" "${ARGS[@]}"; echo; echo

########################
# Run (via srun for SLURM accounting/affinity)
########################
srun ${RUN} exec --bind "${BIND_DIRS}" "${CONTAINER}" /usr/bin/env python3 "${SCRIPT}" "${ARGS[@]}"

echo
echo "Job finished: $(date)"
