#!/bin/bash
# ---------------------------------------------------------------------------
# Shared base for all SupraBench CRC job scripts (pretraining, eval, etc.).
# Source this at the top of each job script after the SGE directives so
# AFS token renewal, environment, and venv activation are all handled.
# ---------------------------------------------------------------------------

# Cleanup function — kills child processes and clears shared memory
cleanup_on_exit() {
  local exit_code=$?
  trap - EXIT
  echo "[Trap] Cleanup triggered with exit code $exit_code at $(date)"
  rm -f /dev/shm/torch_* 2>/dev/null || true
  pkill -9 -P $$ 2>/dev/null || true
  echo "[Trap] Cleanup completed at $(date)"
  exit $exit_code
}
trap cleanup_on_exit EXIT

# Compute nodes need a fresh AFS token before /groups/ is accessible.
kinit -R 2>/dev/null || true
aklog 2>/dev/null || true

# Resolve repo root and activate environment
_REPO_ROOT="/path/to/SupraBench"
source "${_REPO_ROOT}/scripts/crc/bashrc.sh"
cd "${_REPO_ROOT}" || {
  echo "[$(basename "$0")] ERROR: cannot cd to ${_REPO_ROOT}"
  exit 1
}

# Resolve python binary
PYTHON="${_REPO_ROOT}/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="$(which python3 2>/dev/null || which python 2>/dev/null || echo python)"
ACCELERATE="${_REPO_ROOT}/.venv/bin/accelerate"
[ -x "$ACCELERATE" ] || ACCELERATE="$(which accelerate 2>/dev/null || echo accelerate)"

unset _REPO_ROOT
