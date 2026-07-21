#!/bin/bash
# Wrapper executed by HTCondor on the compute node for run_nosigma.py.
#
# run_nosigma.py does `os.system("python execute_don.py ...")`, so we must:
#   * be in the src/ directory (execute_don.py is referenced relatively), and
#   * have `python` on PATH point at an env with jax/flax/optax/matplotlib.
# All dataset/net paths inside don_code.py are absolute (derived from the repo
# root), so only the src/ cwd matters for finding execute_don.py itself.
set -euo pipefail

# --- Locate the repo root from this script's location ---------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Python environment ---------------------------------------------------
# The committed .venv is a macOS venv and will NOT work on the Linux cluster.
# Provide a Linux env one of two ways (edit to match your setup):
#   (a) point CONDA/venv activation here, or
#   (b) `module load` + a prebuilt env.
# By default we activate $REPO_ROOT/.venv if present, else fall back to PATH.
if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

# --- Force CPU + headless plotting ---------------------------------------
export JAX_PLATFORMS=cpu          # never touch a GPU even if one is visible
export MPLBACKEND=Agg             # execute_don.py imports pyplot + saves PNGs
export CUDA_VISIBLE_DEVICES=""

# --- Cap thread pools to the CPUs we requested (default 4) ----------------
NCPUS="${OMP_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${NCPUS}"
export OPENBLAS_NUM_THREADS="${NCPUS}"
export MKL_NUM_THREADS="${NCPUS}"
export NUMEXPR_NUM_THREADS="${NCPUS}"
# Keep XLA's CPU threadpool bounded to the same core count.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${NCPUS}"

echo "host=$(hostname) repo=${REPO_ROOT} python=$(command -v python) cpus=${NCPUS}"
python -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())"

cd "${REPO_ROOT}/src"
exec python run_nosigma.py
