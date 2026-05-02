#!/bin/bash
# ---------------------------------------------------------------------------
# Shared environment for SupraBench CRC job scripts.
# Source this at the top of every job script instead of ~/.bashrc directly.
# ---------------------------------------------------------------------------

source ~/.bashrc

cd "/groups/yye7/BILLY/SupraBench"
source "./.venv/bin/activate"

# ---- Caches & Tmp ---------------------------------------------------------
# Shared HF cache across all yye7 projects — avoids re-downloading the same
# base models. Lives on the group filesystem so it's stable through AFS
# token expiry mid-job.
export HF_HOME="/groups/yye7/BILLY/.cache/huggingface"

# Pre-export HF_TOKEN while AFS is fresh; huggingface_hub's
# _get_token_from_environment() takes precedence over file-based lookup,
# so even after the AFS token expires the in-process token stays valid.
if [ -z "${HF_TOKEN:-}" ] && [ -f "${HF_HOME}/token" ]; then
  _hf_tok="$(cat "${HF_HOME}/token" 2>/dev/null || true)"
  if [ -n "${_hf_tok}" ]; then
    export HF_TOKEN="${_hf_tok}"
  fi
  unset _hf_tok
fi

# Outputs land on the group filesystem so they survive scratch wipes and
# stay readable across nodes. Training scripts respect $OUTPUTS_DIR.
export OUTPUTS_DIR="/groups/yye7/BILLY/SupraBench/outputs"
mkdir -p "${OUTPUTS_DIR}"

# ---- Python ---------------------------------------------------------------
export PYTHONDONTWRITEBYTECODE=1

# ---- Weights & Biases -----------------------------------------------------
export WANDB_PROJECT=suprabench
export WANDB_ENTITY=mtybilly
