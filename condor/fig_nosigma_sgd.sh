#!/bin/bash
# Produce the SGD with/without-sigma mode-loss figure on a COMPUTE NODE (never the login node).
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
python -m src.analysis.RELEVANT.plot_modelosses_with_out_sigma_sgd
echo "EXIT=$?"
