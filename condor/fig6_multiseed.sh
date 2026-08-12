#!/bin/bash
# Wrapper executed by HTCondor: draw the seed-averaged Figure 6 from the ten
# synthv09082026 seeds, both with and without the +/- 1 std band.
#
#   MULTISEED_BAND=0 -> figures/{pdfs,pngs}/Fig6_multiseed.{pdf,png}
#   MULTISEED_BAND=1 -> figures/{pdfs,pngs}/Fig6_multiseed_band.{pdf,png}
#
# The script imports don_code (which imports jax), so it needs a compute node;
# the login node's thread cap makes the XLA CPU client abort.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.venv-linux/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv-linux/bin/activate"
fi

export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export TMPDIR="${_CONDOR_SCRATCH_DIR:-/tmp}"
export MPLCONFIGDIR="${TMPDIR}/mplconfig"
mkdir -p "${MPLCONFIGDIR}"

NCPUS="${OMP_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${NCPUS}"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${NCPUS}"

echo "host=$(hostname) python=$(command -v python) pdflatex=$(command -v pdflatex || echo MISSING)"

cd "${REPO_ROOT}"
rc=0
for band in 0 1; do
    echo "========================================"
    echo "=== MULTISEED_BAND=${band}"
    echo "========================================"
    MULTISEED_BAND="${band}" python -m src.analysis.spectral_bias.plot_res3_sidebyside_mat_gridspec_multiseed 0.2 || rc=$?
done

echo "----------------------------------------"
echo "exit code : ${rc}"
ls -l "${REPO_ROOT}"/figures/pdfs/Fig6_multiseed*.pdf "${REPO_ROOT}"/figures/pngs/Fig6_multiseed*.png 2>&1
exit "${rc}"
