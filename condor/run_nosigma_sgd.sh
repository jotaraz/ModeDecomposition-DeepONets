#!/bin/bash
# Wrapper executed by HTCondor on the compute node for run_nosigma_sgd.py.
#
# run_nosigma_sgd.py does `os.system("python execute_don.py ...")`, so we must:
#   * be in the src/ directory (execute_don.py is referenced relatively), and
#   * have `python` on PATH point at an env with jax/flax/optax/matplotlib.
# All dataset/net paths inside don_code.py are absolute (derived from the repo
# root), so only the src/ cwd matters for finding execute_don.py itself.
set -euo pipefail

# --- Locate the repo root from this script's location ---------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Python environment ---------------------------------------------------
# Use the Linux venv built on the cluster by condor/build_env.sub. (The repo's
# .venv is a macOS venv and will NOT run here; .venv-linux is the cluster one.)
if [[ -f "${REPO_ROOT}/.venv-linux/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv-linux/bin/activate"
elif [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
else
    echo "ERROR: no .venv-linux — run condor/build_env.sub first." >&2
    exit 1
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
exec python run_nosigma_sgd.py "$@"
