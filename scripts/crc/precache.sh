#!/bin/bash
#$ -q long
#$ -N suprabench_precache
#$ -pe smp 8
#$ -l h_rt=08:00:00
#$ -o logs/$JOB_NAME_$JOB_ID.log
#$ -e logs/$JOB_NAME_$JOB_ID.err
#$ -m abe
#$ -M tma2@nd.edu
# ---------------------------------------------------------------------------
# Pre-cache Qwen3.5-27B and the EU-PMC corpus into the shared HF cache
# before submitting GPU training. CPU-only, ~30-60 min wall depending on
# bandwidth.
# ---------------------------------------------------------------------------
set -eo pipefail   # not -u: ~/.bashrc -> /etc/bashrc references BASHRCSOURCED before defining it

mkdir -p logs

source /groups/yye7/BILLY/SupraBench/scripts/crc/base.sh

echo "[precache] HF_HOME=${HF_HOME}"
echo "[precache] $(date) downloading model"
hf download Qwen/Qwen3.5-27B \
  --repo-type model \
  --max-workers 4

echo "[precache] $(date) downloading dataset"
hf download mtybilly/EU-PMC \
  --repo-type dataset \
  --max-workers 4

echo "[precache] $(date) done"
du -sh "${HF_HOME}"/hub/models--Qwen--Qwen3.5-27B 2>/dev/null || true
du -sh "${HF_HOME}"/hub/datasets--mtybilly--EU-PMC 2>/dev/null || true
