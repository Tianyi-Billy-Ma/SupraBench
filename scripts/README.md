# `scripts/`

Cluster-submission scripts for running SupraBench on shared HPC resources.
One subdirectory per cluster — submission semantics and module
environments differ, so scripts do not cross-pollinate.

| Directory | Cluster | Scheduler |
| --- | --- | --- |
| `crc/` | Anonymous University CRC | to be confirmed |
| `delta/` | NCSA Delta at UIUC | Slurm |

Subdirectories are intentionally empty at initialization — add scripts as
runs are provisioned.

## Conventions

- Every submission script ends in a single `uv run python src/main.py ...`
  invocation so local and cluster runs stay byte-identical.
- Hard-coded per-cluster paths (scratch, module loads, conda envs) stay
  inside the cluster's subdirectory; shared logic lives in `src/`.
- Output files land under `outputs/` by default; redirect elsewhere only
  when scratch-space quotas demand it.
