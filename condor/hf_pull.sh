#!/bin/bash
# Pull datasets from HuggingFace ON a compute node (keeps the multi-hundred-MB
# download off the login node). Requires .venv-linux (build_env.sub) first.
#   condor_submit_bid <BID> condor/hf_pull.sub
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "${REPO_ROOT}/.venv-linux/bin/activate" ]]; then
    echo "ERROR: .venv-linux missing — run condor/build_env.sub first." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${REPO_ROOT}/.venv-linux/bin/activate"

if ! curl -sI --max-time 15 https://huggingface.co >/dev/null 2>&1; then
    echo "ERROR: huggingface.co unreachable from this node." >&2
    exit 1
fi

# Config via environment (set in the .sub or on the condor_submit_bid line):
REPO_ID="${HF_REPO_ID:-jo-chen/deeponet-data}"
SUBSET="${HF_SUBSET:-datasets}"        # space-separated: "datasets sb_data" | "nets"

# Token for a private repo: put it in ~/.hf_token (chmod 600) — kept out of
# condor_q, unlike passing it through the job environment.
if [[ -f "${HOME}/.hf_token" ]]; then
    HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.hf_token")"
fi
TOKEN_ARG=()
[[ -n "${HF_TOKEN:-}" ]] && TOKEN_ARG=(--token "${HF_TOKEN}")

echo "host=$(hostname) pull ${REPO_ID} subset=[${SUBSET}] -> ${REPO_ROOT}/data"
# shellcheck disable=SC2086
python hf_pull.py "${REPO_ID}" "${TOKEN_ARG[@]}" --subset ${SUBSET}
echo "HF PULL DONE"
