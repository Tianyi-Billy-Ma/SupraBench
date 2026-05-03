"""LoRA continued pretraining for Qwen3.5-27B (vision-language) on a
text-only biomedical corpus.

Design notes:

- The base model is loaded via ``AutoModelForImageTextToText`` so vision
  components stay structurally intact even while we only train on text.
- Every parameter whose fully-qualified name contains any of
  ``model.freeze_keywords`` is set to ``requires_grad=False`` *before* PEFT
  wrapping, so the vision tower / projector never receive gradients.
- LoRA targets are restricted to modules under
  ``model.language_model_path`` whose short name matches
  ``lora.target_module_names``. This keeps adapters off the vision side.
- Sequences are packed by concatenating tokenised rows and chunking to
  ``dataset.seq_len`` — the standard CPT recipe (e.g. Meditron, BioMistral).
- Launched via ``accelerate launch`` with FSDP_FULL_SHARD; the launch
  config lives at ``scripts/crc/accelerate_fsdp.yaml``.

Run:

    accelerate launch --config_file scripts/crc/accelerate_fsdp.yaml \\
        src/train/cpt_lora.py \\
        --config configs/train/cpt_qwen35_eupmc.yaml \\
        --override training.max_steps=20 dataset.train_rows=256
"""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Config loading & dotted-key override
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def _apply_override(cfg: dict, dotted_key: str, raw_value: str) -> None:
    parts = dotted_key.split(".")
    cursor = cfg
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = _coerce(raw_value)


def load_config(config_path: Path, overrides: Iterable[str]) -> dict:
    cfg = _load_yaml(config_path)
    for entry in overrides:
        if "=" not in entry:
            raise ValueError(f"--override expects key=value, got: {entry!r}")
        key, value = entry.split("=", 1)
        _apply_override(cfg, key.strip(), value.strip())
    return cfg


# ---------------------------------------------------------------------------
# Model loading & PEFT wiring
# ---------------------------------------------------------------------------


_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_base_model(model_cfg: dict):
    dtype = _DTYPE_MAP[model_cfg.get("dtype", "bfloat16")]
    model = AutoModelForImageTextToText.from_pretrained(
        model_cfg["model_id"],
        torch_dtype=dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    return model


def freeze_non_lm_modules(model: torch.nn.Module, freeze_keywords: list[str]) -> int:
    frozen = 0
    keywords = [kw.lower() for kw in freeze_keywords]
    for name, param in model.named_parameters():
        lower = name.lower()
        if any(kw in lower for kw in keywords):
            param.requires_grad = False
            frozen += param.numel()
    return frozen


def find_lora_target_modules(
    model: torch.nn.Module,
    language_model_path: str,
    short_names: list[str],
) -> list[str]:
    short = set(short_names)
    targets: list[str] = []
    prefix = language_model_path.rstrip(".") + "."
    for name, module in model.named_modules():
        if not name.startswith(prefix):
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        if name.rsplit(".", 1)[-1] in short:
            targets.append(name)
    if not targets:
        raise RuntimeError(
            f"No LoRA target modules found under {language_model_path!r} "
            f"with short names {sorted(short)}. Inspect model.named_modules() "
            f"and update model.language_model_path or lora.target_module_names."
        )
    return targets


def attach_lora(model: torch.nn.Module, lora_cfg: dict, target_modules: list[str]):
    peft_cfg = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg.get("bias", "none"),
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, peft_cfg)


# ---------------------------------------------------------------------------
# Dataset: title+abstract → packed token chunks
# ---------------------------------------------------------------------------


def build_packed_dataset(tokenizer, dataset_cfg: dict, train_rows: int | None):
    raw = load_dataset(dataset_cfg["hub_id"], split=dataset_cfg.get("split", "filtered"))
    if train_rows is not None:
        raw = raw.select(range(min(train_rows, len(raw))))

    template: str = dataset_cfg["text_template"]
    title_field = dataset_cfg["title_field"]
    abstract_field = dataset_cfg["abstract_field"]

    def _format(row):
        title = row.get(title_field) or ""
        abstract = row.get(abstract_field) or ""
        return {"text": template.format(title=title, abstract=abstract).strip() + tokenizer.eos_token}

    formatted = raw.map(
        _format,
        num_proc=dataset_cfg.get("num_proc", 1),
        remove_columns=raw.column_names,
        desc="format title+abstract",
    )

    def _tokenize(batch):
        return tokenizer(batch["text"], add_special_tokens=False)

    tokenized = formatted.map(
        _tokenize,
        batched=True,
        num_proc=dataset_cfg.get("num_proc", 1),
        remove_columns=["text"],
        desc="tokenize",
    )

    seq_len = int(dataset_cfg["seq_len"])

    def _pack(batch):
        concatenated = {key: list(chain(*batch[key])) for key in batch}
        total = (len(concatenated["input_ids"]) // seq_len) * seq_len
        if total == 0:
            return {"input_ids": [], "attention_mask": []}
        chunks = {
            key: [values[i : i + seq_len] for i in range(0, total, seq_len)]
            for key, values in concatenated.items()
        }
        chunks["labels"] = deepcopy(chunks["input_ids"])
        return chunks

    packed = tokenized.map(
        _pack,
        batched=True,
        num_proc=dataset_cfg.get("num_proc", 1),
        desc=f"pack into {seq_len}-token chunks",
    )
    return packed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA continued pretraining for Qwen3.5 on EU-PMC")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config, args.override)

    # Instantiate accelerate state BEFORE from_pretrained so the DeepSpeedPlugin
    # registers the global HfDeepSpeedConfig and zero3_init_flag activates the
    # `deepspeed.zero.Init()` context that partitions the 56 GB base across
    # ranks at construction time. Without this, the model lands full on each
    # A40 and OOMs.
    _accel_state = PartialState()  # noqa: F841 — must outlive load_base_model

    is_main = int(os.environ.get("RANK", "0")) == 0
    if is_main:
        print(f"[cpt_lora] config = {cfg['run_name']}")

    # --- tokenizer -------------------------------------------------------
    tokenizer_id = cfg["model"]["model_id"]
    try:
        processor = AutoProcessor.from_pretrained(
            tokenizer_id, trust_remote_code=cfg["model"].get("trust_remote_code", True)
        )
        tokenizer = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
            tokenizer_id, trust_remote_code=cfg["model"].get("trust_remote_code", True)
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, trust_remote_code=cfg["model"].get("trust_remote_code", True)
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- dataset ---------------------------------------------------------
    train_rows = cfg.get("dataset", {}).pop("train_rows", None)
    packed = build_packed_dataset(tokenizer, cfg["dataset"], train_rows)
    if is_main:
        print(f"[cpt_lora] packed sequences = {len(packed):,} of len {cfg['dataset']['seq_len']}")

    # --- model + LoRA ----------------------------------------------------
    model = load_base_model(cfg["model"])
    frozen_params = freeze_non_lm_modules(model, cfg["model"].get("freeze_keywords", []))
    if is_main:
        print(f"[cpt_lora] frozen non-LM params: {frozen_params:,}")

    targets = find_lora_target_modules(
        model,
        cfg["model"]["language_model_path"],
        cfg["lora"]["target_module_names"],
    )
    if is_main:
        print(f"[cpt_lora] LoRA targets: {len(targets)} modules (sample: {targets[:3]} ...)")

    model = attach_lora(model, cfg["lora"], targets)
    if cfg["training"].get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if is_main:
        model.print_trainable_parameters()

    # --- trainer ---------------------------------------------------------
    train_kwargs = dict(cfg["training"])
    output_dir = str(train_kwargs.pop("output_dir"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=output_dir,
        run_name=cfg.get("run_name"),
        **train_kwargs,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=packed,
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(output_dir)  # saves LoRA adapter only
    if is_main:
        print(f"[cpt_lora] adapter saved to {output_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
