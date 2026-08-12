#!/bin/bash
# Fig 9 post-processing for ONE net, on a compute node (never the login node).
#   condor/components_one.sh <bid> <nep> <exponent> <w> <vtag>
#
# compute_components_fixedindices.py writes log_diagoffdiag_new.txt (40 x 5111),
# which show_components_2x2.py (fig 9b) reads. show_components_mult_multsizes.py
# (fig 9a) looks for log_diagoffdiag_big1e-08.txt instead -- on HuggingFace those
# two files are byte-identical, i.e. the published _big1e-08 is a COPY of _new,
# not a separate computation. So we copy it here.
#
# (old-stuff/compute_components.py is NOT the fig 9 path: it emits 11+2*llw+4*llw^2
#  = 10111 columns, which matches neither the published files nor the readers.)
set -euo pipefail
BID="$1"; NEP="$2"; EXPO="$3"; W="$4"; VTAG="$5"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/.venv-linux/bin/activate"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=""
export TMPDIR="${_CONDOR_SCRATCH_DIR:-/tmp}"
NCPUS="${OMP_NUM_THREADS:-4}"
export OMP_NUM_THREADS=$NCPUS OPENBLAS_NUM_THREADS=$NCPUS MKL_NUM_THREADS=$NCPUS
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${NCPUS}"

echo "host=$(hostname) args=$*"
cd "${REPO_ROOT}"
START=$(date +%s)
python -m src.analysis.RELEVANT.compute_components_fixedindices "$BID" "$NEP" "$EXPO" "$W" "$VTAG"
echo "ELAPSED_SECONDS $(( $(date +%s) - START )) FOR $*"

# --- copy _new -> _big1e-08 so fig 9a can read it (bid=3 naming) --------------
if [[ "${BID}" != "3" ]]; then
    echo "NOTE bid!=3, skipping the _big1e-08 copy (name convention is bid-specific)"
    exit 0
fi
D="${REPO_ROOT}/data/nets/whichT0_doStackedFalse_doSigma1_sisc1.0_aT0.0_aB0.0_exp${EXPO}_Nep${NEP}_d5_w${W}_llw50_batkdvnx401_dt0.0001_nc5_m5000_1999_numd1000_lrSGD32_v${VTAG}"
if [[ ! -f "${D}/log_diagoffdiag_new.txt" ]]; then
    echo "ERROR: ${D}/log_diagoffdiag_new.txt not produced" >&2
    exit 5
fi
cp -f "${D}/log_diagoffdiag_new.txt" "${D}/log_diagoffdiag_big1e-08.txt"
echo "COPIED _new -> _big1e-08 ($(awk 'NR==1{print NF;exit}' "${D}/log_diagoffdiag_new.txt") cols) for w${W} v${VTAG}"
