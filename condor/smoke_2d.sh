#!/bin/bash
# Smoke test: train a DeepONet for 100 epochs on the moved 2D heat sine dataset,
# to confirm the full pipeline runs end-to-end on 2D data (which_T=0 SVD trunk).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.venv-linux/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv-linux/bin/activate"
elif [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=""
NCPUS="${OMP_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${NCPUS}" OPENBLAS_NUM_THREADS="${NCPUS}"
export MKL_NUM_THREADS="${NCPUS}" NUMEXPR_NUM_THREADS="${NCPUS}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${NCPUS}"

echo "host=$(hostname) repo=${REPO_ROOT} python=$(command -v python) cpus=${NCPUS}"
python -c "import jax, flax, optax; print('jax', jax.__version__, 'devices', jax.devices())"

cd "${REPO_ROOT}/src"
mkdir -p nets

# execute_don.py argv order:
#  1 Nepochs 2 vtag 3 depth 4 width 5 llw 6 doplot 7 batch_name 8 lrstag
#  9 init_lr 10 decay 11 num_data 12 which_T 13 dotruesigma 14 uendtag
#  15 sigmascale 16 exponent 17 doadam 18 dostacked
exec python execute_don.py \
    100 0 10 100 50 0 heat2d_sine_nx64_K8_D1 32 1e-4 0.95 1000 0 0 t0.004 First 1.0 1 0
