#!/bin/bash
# Produce BOTH multiseed figure sets on a COMPUTE NODE (never the login node).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/.venv-linux/bin/activate"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=""
export TMPDIR="${_CONDOR_SCRATCH_DIR:-/tmp}"
NCPUS="${OMP_NUM_THREADS:-4}"
export OMP_NUM_THREADS=$NCPUS OPENBLAS_NUM_THREADS=$NCPUS MKL_NUM_THREADS=$NCPUS
echo "host=$(hostname) HEAD=$(cd $REPO_ROOT && git log --oneline -1)"
cd "${REPO_ROOT}"
mkdir -p figures/pdfs figures/pngs
echo "=== mean-only set ==="
python produce_all_figures_multiseed.py
echo "=== band set ==="
MULTISEED_BAND=1 python produce_all_figures_multiseed.py
echo "EXIT=$?"
