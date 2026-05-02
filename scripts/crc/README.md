# `scripts/crc/`

SGE submission scripts for the Notre Dame CRC. Source the shared helpers
at the top of every job script — never hand-roll module loads or venv
activation.

| File | Role |
| --- | --- |
| `bashrc.sh` | Per-job environment: `cd` to repo root, activate `.venv`, export `HF_HOME`, `HF_TOKEN`, `OUTPUTS_DIR`, wandb. |
| `base.sh` | Wraps `bashrc.sh` with AFS token renewal (`kinit -R`, `aklog`), an EXIT trap that cleans up `/dev/shm/torch_*`, and resolves `$PYTHON` / `$ACCELERATE`. |
| `accelerate_fsdp.yaml` | Accelerate launch config for 4× A40 FSDP_FULL_SHARD bf16. Switch `fsdp_auto_wrap_policy` to `TRANSFORMER_BASED_WRAP` after the smoke test reveals the decoder layer class name. |
| `precache.sh` | CPU job (`-q long`, 8 cores) to pull `Qwen/Qwen3.5-27B` and `mtybilly/EU-PMC` into the shared HF cache. Run once. |
| `cpt_qwen35_eupmc_smoke.sh` | 1.5 h GPU smoke test: 20 steps over 256 rows at seq_len=1024. |
| `cpt_qwen35_eupmc.sh` | Full LoRA CPT, 24 h walltime. Re-submit to resume from the latest checkpoint. |

Submit pattern:

```bash
ssh crc
cd /groups/yye7/BILLY/SupraBench
qsub scripts/crc/precache.sh                # one-time
qsub scripts/crc/cpt_qwen35_eupmc_smoke.sh  # validate
qsub scripts/crc/cpt_qwen35_eupmc.sh        # full run
```

Logs land under `logs/<job_name>_<job_id>.{log,err}`. Outputs land under
`$OUTPUTS_DIR` (group-filesystem path defined in `bashrc.sh`).
