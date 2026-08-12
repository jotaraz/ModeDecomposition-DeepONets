#!/bin/bash
# Wrapper executed by HTCondor on the compute node: generate the synthetic
# SVD-structured datasets (Phi Sigma V^T) by running
# src/analysis/spectral_bias/run_synthetic_data_gen.py unmodified. It writes
# <stem>_P.txt, <stem>_R.txt and one <stem>_<fs..ss..ns..>_U.txt per config into
# data/datasets. Pure NumPy, single-threaded; CPU-only.
#
# The generator rejection-samples the V directions and aborts with
# "no new direction found" (-> TypeError on the 5-way unpack) when no candidate
# clears innerprod_threshold. Unseeded that is a coin flip, so we try seeds in
# order and stop at the first that completes every config. Each attempt
# regenerates X and rewrites every output file, so the surviving files always
# come from one single successful draw.
set -uo pipefail

MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.venv-linux/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv-linux/bin/activate"
elif [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

export MPLBACKEND=Agg

echo "host=$(hostname) repo=${REPO_ROOT} python=$(command -v python)"
python -c "import numpy, matplotlib; print('numpy', numpy.__version__, 'matplotlib', matplotlib.__version__)"

good_seed=""
for (( seed=0; seed<MAX_ATTEMPTS; seed++ )); do
    echo "========================================"
    echo "=== attempt with numpy seed=${seed}"
    echo "========================================"
    if time python -u "${SCRIPT_DIR}/gen_synth_seeded.py" "${seed}" \
            "${REPO_ROOT}/src/analysis/spectral_bias/run_synthetic_data_gen.py"; then
        good_seed="${seed}"
        break
    fi
    echo "--- seed=${seed} did not find enough near-orthogonal directions, retrying"
done

echo "========================================"
if [[ -z "${good_seed}" ]]; then
    echo "FAILED: no seed in 0..$((MAX_ATTEMPTS-1)) produced a complete set"
    exit 1
fi

echo "SUCCESS with numpy seed=${good_seed}"
ls -l "${REPO_ROOT}/data/datasets/" | grep -i synth
