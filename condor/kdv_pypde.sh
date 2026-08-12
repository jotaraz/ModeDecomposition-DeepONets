#!/bin/bash
# HTCondor wrapper for src/kdv_pypde_check.py.
#
# py-pde is NOT part of the project environment -- it is only needed to re-run the
# original data generator -- so it goes into a throwaway venv inside this workdir,
# leaving the repo's .venv-linux untouched.  Compute nodes reach PyPI through the
# HTTP(S) proxy; the login node is never used for this.
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${WORKDIR}/.venv-pde"

export TMPDIR="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}"
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=""

if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "building ${VENV} ..."
    /usr/bin/python3.10 -m venv "${VENV}"
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    python -m pip install --upgrade pip wheel >/dev/null
    python -m pip install "py-pde" "numpy<2.3" scipy
else
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi

python -c "import pde, numpy, scipy; print('py-pde', pde.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)"

NCPUS="${OMP_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${NCPUS}" OPENBLAS_NUM_THREADS="${NCPUS}" MKL_NUM_THREADS="${NCPUS}"
# py-pde JITs its right-hand side with numba; keep its cache off the 1 GB /tmp.
export NUMBA_CACHE_DIR="${TMPDIR}/numba_cache"
mkdir -p "${NUMBA_CACHE_DIR}"

echo "host=$(hostname) cpus=${NCPUS}"
echo "args: $*"

cd "${WORKDIR}"
START=$(date +%s)
python kdv_pypde_check.py "$@"
echo "ELAPSED_SECONDS $(($(date +%s) - START))"
