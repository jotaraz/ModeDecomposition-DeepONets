#!/bin/bash
# HTCondor wrapper for src/kdv_regen.py.  All arguments are passed straight
# through, e.g.
#   condor/kdv_regen.sh validate --nsamples 8
#   condor/kdv_regen.sh generate --refine 3 --outdir out/refine3
#
# Runs from the working directory it is submitted with (initialdir), so the
# script and its outputs live outside the repo's data/ tree.  The only thing
# taken from the repo is the Linux venv built by condor/build_env.sub.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fast/jtaraz/MISC/ModeDecomposition-DeepONets}"
WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${REPO_ROOT}/.venv-linux/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv-linux/bin/activate"
else
    echo "ERROR: no ${REPO_ROOT}/.venv-linux -- run condor/build_env.sub first." >&2
    exit 1
fi

# Pure NumPy job: no JAX, no CUDA, headless.
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=""
export TMPDIR="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}"

NCPUS="${OMP_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${NCPUS}"
export OPENBLAS_NUM_THREADS="${NCPUS}"
export MKL_NUM_THREADS="${NCPUS}"
export NUMEXPR_NUM_THREADS="${NCPUS}"

echo "host=$(hostname) cpus=${NCPUS} workdir=${WORKDIR}"
echo "args: $*"

cd "${WORKDIR}"
START=$(date +%s)
python kdv_regen.py "$@"
END=$(date +%s)
echo "ELAPSED_SECONDS $((END - START))"
