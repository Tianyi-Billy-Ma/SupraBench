# `scripts/crc/`

SGE submission scripts for the Notre Dame CRC. Source the shared helpers
at the top of every job script — never hand-roll module loads or venv
activation.

| File | Role |
| --- | --- |
| `bashrc.sh` | Per-job environment: `cd` to repo root, activate `.venv`, export `HF_HOME`, `HF_TOKEN`, `OUTPUTS_DIR`, wandb. |
| `base.sh` | Wraps `bashrc.sh` with AFS token renewal (`kinit -R`, `aklog`), an EXIT trap that cleans up `/dev/shm/torch_*`, and resolves `$PYTHON` / `$ACCELERATE`. |
| `ds_config_zero3.json` | DeepSpeed ZeRO-3 config (bf16, no CPU offload, stage3 with hardcoded train_batch_size=16 / grad_accum=4 / micro_batch=1 — `auto` sentinels can't be used because `deepspeed.zero.Init` validates batch sizes at `from_pretrained` time). |
| `accelerate_deepspeed.yaml` | Accelerate launch config for 4× A40 → DeepSpeed ZeRO-3 (current). |
| `accelerate_fsdp.yaml` | Alternative FSDP_FULL_SHARD launch config with TRANSFORMER_BASED_WRAP. Validated working in smoke v9; kept as the fallback path if DeepSpeed integration breaks. |
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
