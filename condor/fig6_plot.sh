#!/bin/bash
# Wrapper executed by HTCondor on the compute node: redraw Figure 6 from the
# retrained synthv09082026 nets, i.e. the produce_all_figures.py "6" command
#
#     python src/analysis/spectral_bias/plot_res3_sidebyside_mat_gridspec.py 0.2
#
# The script uses the pgf backend with text.usetex, so the node needs pdflatex.
# It writes figures/pdfs/Fig6.pdf and figures/pngs/Fig6.png (the published
# versions were copied to figures/published_backup/ first).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.venv-linux/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv-linux/bin/activate"
fi

# usetex needs a writable home for the font/latex caches.
export TMPDIR="${_CONDOR_SCRATCH_DIR:-/tmp}"
export MPLCONFIGDIR="${TMPDIR}/mplconfig"
mkdir -p "${MPLCONFIGDIR}"

echo "host=$(hostname) python=$(command -v python) pdflatex=$(command -v pdflatex || echo MISSING)"

cd "${REPO_ROOT}"
python src/analysis/spectral_bias/plot_res3_sidebyside_mat_gridspec.py 0.2
rc=$?

echo "----------------------------------------"
echo "exit code : ${rc}"
ls -l "${REPO_ROOT}/figures/pdfs/Fig6.pdf" "${REPO_ROOT}/figures/pngs/Fig6.png" 2>&1
exit "${rc}"
