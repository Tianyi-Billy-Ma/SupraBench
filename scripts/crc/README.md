# `scripts/crc/`

CRC launchers for SupraBench. All jobs go through AutoExp's warm pool
(`gpu@@anonymous_lab`, 4× A40); the SGE directives at the top of each `cpt_*.sh`
script are inert when AutoExp dispatches them but stay there for the rare
hand-`qsub` case.

## Top-level files

| File | Role |
| --- | --- |
| `bashrc.sh` | Per-job environment: `cd` to repo root, activate `.venv`, export `HF_HOME`, `HF_TOKEN`, `OUTPUTS_DIR`, wandb. |
| `base.sh` | Wraps `bashrc.sh` with AFS token renewal (`kinit -R`, `aklog`), an EXIT trap that cleans up `/dev/shm/torch_*`, and resolves `$PYTHON` / `$ACCELERATE`. |
| `accelerate_fsdp.yaml` | Accelerate launch config for 4× A40 FSDP_FULL_SHARD bf16. Overridden per-job for plain causal-LM bases via `--fsdp_transformer_layer_cls_to_wrap`. |
| `precache.sh` | CPU job to warm the shared HF cache. Run once per new base model. |
| `eval_task.sh` | Central eval runner. Reads `EVAL_TASK_CONFIG` + `EVAL_MODEL_CONFIG` from env and calls `src/main.py`. Not submitted directly; the `submit_eval.sh` helper generates per-submission launchers that exec into it. |
| `extract_adapter.sh` | Parameterized FSDP-shard → PEFT-adapter consolidator. Reads `CHECKPOINT`, `OUTPUT`, `BASE_MODEL`, `LORA_R`, `LORA_ALPHA`, `BASE_PATH` from env. Not submitted directly; use the `submit_extract.sh` recipe helper. |
| `submit_eval.sh` | `./submit_eval.sh <task_yaml> <model_yaml> [gpus]` — writes a per-submission launcher under `_generated/`, calls `autoexp submit`. |
| `submit_extract.sh` | `./submit_extract.sh <recipe>` — recipes: `qwen35_27b_v1`, `qwen35_27b_v2`, `qwen35_9b_v2`, `llama31_8b_v2`. Writes a per-submission launcher under `_generated/`, calls `autoexp submit`. |

### CPT launchers (per-model, kept as full scripts)

| File | Recipe |
| --- | --- |
| `cpt_qwen35_27b_supra_v2.sh` + `_smoke.sh` | v2 LoRA r=32/α=64 on Qwen3.5-27B VLM (vision tower frozen) with the EvoLM-style mix. |
| `cpt_qwen35_9b_supra_v2.sh` + `_smoke.sh` | v2 LoRA on Qwen3.5-9B plain causal LM. Sets `HF_HUB_OFFLINE=1` and `--fsdp_transformer_layer_cls_to_wrap Qwen3_5DecoderLayer`. |
| `cpt_llama31_8b_supra_v2.sh` + `_smoke.sh` | v2 LoRA on Llama-3.1-8B. Sets `--fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer`. |

CPT launchers are *not* parameterized — each one bundles model-specific
environment overrides (FSDP wrap class, HF offline mode, vision-tower freeze
keywords) that vary too much to share a template.

## Submitting jobs

```bash
# (1) One-time per new base model
qsub /path/to/SupraBench/scripts/crc/precache.sh

# (2) CPT — submit the per-model launcher directly
autoexp submit --gpus 4 --name cpt_qwen9b_v2 --cwd $PWD \
  scripts/crc/cpt_qwen35_9b_supra_v2_smoke.sh
autoexp submit --gpus 4 --name cpt_qwen9b_v2 --cwd $PWD \
  scripts/crc/cpt_qwen35_9b_supra_v2.sh

# (3) Adapter extract — recipe-named
scripts/crc/submit_extract.sh qwen35_9b_v2

# (4) Eval — one call per (task, model) pair
scripts/crc/submit_eval.sh bap_base.yaml qwen35_9b_supra_v2_lora.yaml
scripts/crc/submit_eval.sh tbs_base.yaml qwen35_9b_supra_v2_lora.yaml
scripts/crc/submit_eval.sh hgd_base.yaml qwen35_9b_supra_v2_lora.yaml
scripts/crc/submit_eval.sh sid_base.yaml qwen35_9b_supra_v2_lora.yaml
```

The generated launchers under `_generated/` are tracked-but-ignored
(`.gitignore`'d) and serve as a permanent on-disk record of every job
submitted via the helpers.

## `archive/`

Frozen per-(task, model) launchers from earlier sessions. Kept verbatim so
the exact submission used for a given metrics.json can always be retrieved.
**Do not edit** files in `archive/` — if you need a variant, run the
parameterized helpers above.

Mapping from old per-script naming to the helper invocation:

| Archived script | Equivalent helper invocation |
| --- | --- |
| `eval_task<N>_qwen35_27b_v1.sh` | `submit_eval.sh task<N>_base.yaml qwen35_27b_eupmc_lora.yaml` |
| `eval_task<N>_qwen35_27b_v2.sh` | `submit_eval.sh task<N>_base.yaml qwen35_27b_supra_v2_lora.yaml` |
| `eval_bap_qwen35_27b_v2_guided.sh` | `submit_eval.sh bap_base.yaml qwen35_27b_supra_v2_lora_guided.yaml` |
| `eval_task<N>_qwen35_9b_v2.sh` | `submit_eval.sh task<N>_base.yaml qwen35_9b_supra_v2_lora.yaml` |
| `eval_task<N>_llama31_8b_v2.sh` | `submit_eval.sh task<N>_base.yaml llama31_8b_supra_v2_lora.yaml` |
| `extract_adapter_qwen35_27b_v1.sh` | `submit_extract.sh qwen35_27b_v1` |
| `extract_adapter_qwen35_27b_v2.sh` | `submit_extract.sh qwen35_27b_v2` |
| `extract_adapter_qwen35_9b_v2.sh` | `submit_extract.sh qwen35_9b_v2` |
| `extract_adapter_llama31_8b_v2.sh` | `submit_extract.sh llama31_8b_v2` |
| `cpt_qwen35_27b_eupmc_v1{,_smoke}.sh` | (no helper; v1 recipe frozen and not retrained) |

Outputs land under `$OUTPUTS_DIR` (`outputs/<run_name>/`); per-job logs land
under `logs/<job_name>_<job_id>.{log,err}`.
