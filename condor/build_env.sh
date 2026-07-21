#!/bin/bash
# Build the Linux virtualenv ON a compute node (keeps pip/network work off the
# login node). Home is on shared Lustre, so the resulting .venv-linux is visible
# to every later job. Run via HTCondor: condor_submit_bid <BID> condor/build_env.sub
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYBIN="${PYBIN:-/usr/bin/python3.10}"     # cluster default interpreter
VENV="${REPO_ROOT}/.venv-linux"

echo "host=$(hostname) repo=${REPO_ROOT} python=$(${PYBIN} --version 2>&1)"

# Compute node must be able to reach PyPI to install wheels.
if ! curl -sI --max-time 15 https://pypi.org >/dev/null 2>&1; then
    echo "ERROR: PyPI unreachable from this node — cannot pip install here." >&2
    echo "       Either this node has no internet, or use a node/proxy that does." >&2
    exit 1
fi

# Fresh venv (idempotent: safe to re-run).
rm -rf "${VENV}"
"${PYBIN}" -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt

# Smoke-test the imports the training code actually uses, and confirm x64.
python - <<'PY'
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import flax, optax, numpy, scipy, matplotlib, huggingface_hub
print("jax", jax.__version__, "devices", jax.devices())
print("flax", flax.__version__, "optax", optax.__version__,
      "numpy", numpy.__version__, "scipy", scipy.__version__)
assert jnp.array(1.0).dtype == jnp.float64, "x64 not enabled!"
print("SMOKE OK")
PY

# Freeze the resolved versions for reproducibility.
python -m pip freeze > requirements.lock.txt
echo "wrote ${REPO_ROOT}/requirements.lock.txt"
echo "ENV BUILD DONE -> ${VENV}"
