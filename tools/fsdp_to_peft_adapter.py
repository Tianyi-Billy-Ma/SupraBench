"""Extract a PEFT LoRA adapter from an FSDP-sharded HF Trainer checkpoint.

Background
----------
The CPT training run (src/train/cpt_lora.py) used accelerate's FSDP plugin
with ``fsdp_state_dict_type: SHARDED_STATE_DICT``. That choice is faster
during training but causes ``trainer.save_model(output_dir)`` to write
sharded FSDP files instead of the standard PEFT adapter format
(``adapter_config.json`` + ``adapter_model.safetensors``). Loading the
adapter for inference via ``PeftModel.from_pretrained(...)`` therefore
fails:

    ValueError: Can't find 'adapter_config.json' at '<checkpoint>'

This tool reads the DCP (PyTorch Distributed Checkpoint) shards under
``<checkpoint>/pytorch_model_fsdp_0/``, consolidates them into a flat
``dict[str, torch.Tensor]``, filters for LoRA parameter names
(``*.lora_A.*`` and ``*.lora_B.*``), and rewrites them as
``adapter_model.safetensors`` plus a hand-built ``adapter_config.json``.

Usage
-----
::

    python tools/fsdp_to_peft_adapter.py \\
        --checkpoint /groups/yye7/BILLY/SupraBench/outputs/cpt_qwen35_eupmc/checkpoint-1302 \\
        --output     /groups/yye7/BILLY/SupraBench/outputs/cpt_qwen35_eupmc/adapter \\
        --base-model Qwen/Qwen3.5-27B \\
        --lora-r 64 --lora-alpha 128 --lora-dropout 0.05 \\
        --target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \\
        --base-path model.language_model

Runs single-process (no distributed init); needs ~50–60 GB CPU RAM to
hold the full consolidated state dict in bf16 before filtering.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="HF Trainer FSDP checkpoint dir (contains pytorch_model_fsdp_0/).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write adapter_config.json + adapter_model.safetensors.")
    parser.add_argument("--base-model", required=True,
                        help="Base model HF id (recorded in adapter_config).")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=float, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--base-path", default="model.language_model",
                        help="Module path that hosts LoRA targets (matches training).")
    parser.add_argument("--bias", default="none", choices=["none", "all", "lora_only"])
    parser.add_argument("--task-type", default="CAUSAL_LM")
    args = parser.parse_args()

    fsdp_dir = args.checkpoint / "pytorch_model_fsdp_0"
    if not fsdp_dir.is_dir():
        raise SystemExit(f"missing FSDP shards: {fsdp_dir}")

    args.output.mkdir(parents=True, exist_ok=True)

    # Step 1 — consolidate the DCP shards into a single flat state_dict.
    # We use torch.distributed.checkpoint.format_utils.dcp_to_torch_save
    # to write a temporary monolithic file, then torch.load it back. This
    # avoids touching torch.distributed init in this single-process tool.
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    print(f"[fsdp_to_peft] consolidating shards from {fsdp_dir} ...", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        consolidated = Path(tmpdir) / "model.bin"
        dcp_to_torch_save(str(fsdp_dir), str(consolidated))
        print(f"[fsdp_to_peft] consolidated -> {consolidated} "
              f"({consolidated.stat().st_size / 1e9:.1f} GB)", flush=True)
        state_dict = torch.load(consolidated, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

    # Step 2 — keep only LoRA params. PEFT names them like
    # "<full.path.to.target>.lora_A.<adapter_name>.weight" and same for lora_B.
    print(f"[fsdp_to_peft] filtering LoRA tensors from "
          f"{len(state_dict):,} parameters ...", flush=True)
    lora_state = {
        k: v for k, v in state_dict.items()
        if ".lora_A." in k or ".lora_B." in k
    }
    if not lora_state:
        raise SystemExit(
            "No LoRA parameters found. Verify --base-path and --target-modules "
            "match how the training script registered LoRA."
        )
    print(f"[fsdp_to_peft] kept {len(lora_state):,} LoRA tensors "
          f"(sample: {next(iter(lora_state))})", flush=True)

    # PEFT's adapter_model.safetensors uses key prefix "base_model.model.<...>"
    # exactly as PEFT writes them — so just save what FSDP gave us, since the
    # PEFT-wrapped model used the same key conventions.

    # Step 3 — write adapter_model.safetensors via safetensors.torch.save_file.
    from safetensors.torch import save_file
    out_safetensors = args.output / "adapter_model.safetensors"
    save_file(lora_state, str(out_safetensors))
    print(f"[fsdp_to_peft] wrote {out_safetensors} "
          f"({out_safetensors.stat().st_size / 1e6:.1f} MB)", flush=True)

    # Step 4 — write adapter_config.json. Match the LoraConfig used during
    # training so PEFT's _get_peft_type / .from_pretrained accept it.
    target_modules = sorted({k.split(".lora_A.")[0].rsplit(".", 1)[-1]
                             for k in lora_state if ".lora_A." in k})
    if not target_modules:
        target_modules = list(args.target_modules)
    print(f"[fsdp_to_peft] inferred target_modules: {target_modules}", flush=True)

    adapter_cfg = {
        "peft_type": "LORA",
        "task_type": args.task_type,
        "base_model_name_or_path": args.base_model,
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": args.bias,
        "target_modules": target_modules,
        "fan_in_fan_out": False,
        "init_lora_weights": True,
        "modules_to_save": None,
        "inference_mode": True,
    }
    out_cfg = args.output / "adapter_config.json"
    with out_cfg.open("w") as fh:
        json.dump(adapter_cfg, fh, indent=2)
    print(f"[fsdp_to_peft] wrote {out_cfg}", flush=True)

    print(f"[fsdp_to_peft] DONE — adapter is at {args.output}")


if __name__ == "__main__":
    main()
