#!/bin/bash
# Generic HTCondor wrapper: run ONE execute_don.py training with the 20 arguments
# passed straight through, i.e.
#   condor/train_one.sh Nep vtag d w llw doplot batch_name lrstag init_lr decay \
#                       num_data whichT doSigma uendtag sisc exp doadam doStacked \
#                       adaptiveLR momentum
#
# execute_don.py does `from don_code import *`, so it must run with cwd=src/.
# All dataset/net paths inside don_code.py are absolute (derived from the repo root).
#
# Prints elapsed wall time as the last line so a sweep can be calibrated from the logs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -ne 20 ]]; then
    echo "ERROR: expected 20 arguments for execute_don.py, got $#: $*" >&2
    exit 2
fi

# --- Python environment (Linux venv built by condor/build_env.sub) ---------
if [[ -f "${REPO_ROOT}/.venv-linux/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv-linux/bin/activate"
else
    echo "ERROR: no .venv-linux — run condor/build_env.sub first." >&2
    exit 1
fi

# --- Force CPU + headless plotting ----------------------------------------
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=""

# /tmp is only 1 GB on this cluster; keep JAX's compilation cache off it.
export TMPDIR="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}"

# --- Cap thread pools to the CPUs we requested ----------------------------
NCPUS="${OMP_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${NCPUS}"
export OPENBLAS_NUM_THREADS="${NCPUS}"
export MKL_NUM_THREADS="${NCPUS}"
export NUMEXPR_NUM_THREADS="${NCPUS}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${NCPUS}"

echo "host=$(hostname) cpus=${NCPUS} tmpdir=${TMPDIR}"
echo "args: $*"

cd "${REPO_ROOT}/src"
START=$(date +%s)
python execute_don.py "$@"
END=$(date +%s)

echo "ELAPSED_SECONDS $((END - START)) FOR $*"
