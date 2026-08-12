#!/bin/bash
# Wrapper for one config of the 2D DeepONet sweep. Receives the 18 execute_don.py
# arguments as $@ (from the queue block in condor/run_2d_sweep.sub) and runs the
# training for that single config. CPU-only, same setup as the smoke test.
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

echo "host=$(hostname) cpus=${NCPUS} args=$*"
python -c "import jax; print('jax', jax.__version__, jax.devices())"


# --- completeness guard ------------------------------------------------------
# execute_don.py skips any net whose directory already exists, WITHOUT checking
# that it finished. An evicted job therefore restarts, finds the partial
# directory, and exits in seconds having done nothing (this silently cost a net
# in cluster 17448308). Require the final checkpoint; delete anything partial.
NEP="$1"; VTAG="$2"; DEPTH="$3"; WID="$4"; LLW="$5"; BATCH="$7"; LRSTAG="$8"
NUMD="${11}"; WHICHT="${12}"; DOSIG="${13}"; UEND="${14}"; SISC="${15}"; EXPO="${16}"
DSTACK="${18}"
[[ "${DSTACK}" == "0" ]] && STK="False" || STK="True"
NETDIR="${REPO_ROOT}/data/nets/whichT${WHICHT}_doStacked${STK}_doSigma${DOSIG}_sisc${SISC}"
NETDIR="${NETDIR}_aT0.0_aB0.0_exp${EXPO}_Nep${NEP}_d${DEPTH}_w${WID}_llw${LLW}"
NETDIR="${NETDIR}_bat${BATCH}_${UEND}_numd${NUMD}_lrAdam${LRSTAG}_v${VTAG}"
FINAL_CHP="${NETDIR}/$(( NEP - 99 ))cur_chp"
if [[ -f "${FINAL_CHP}" ]]; then
    echo "SKIP complete: $(basename "${NETDIR}")"
    exit 0
elif [[ -d "${NETDIR}" ]]; then
    echo "REPAIR removing incomplete $(basename "${NETDIR}")"
    rm -rf "${NETDIR:?}"
fi
# -----------------------------------------------------------------------------

cd "${REPO_ROOT}/src"
mkdir -p nets
exec python execute_don.py "$@"
