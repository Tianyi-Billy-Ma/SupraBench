# SupraBench

**SupraBench** is the first benchmark for evaluating large language models on
**supramolecular host–guest chemistry** reasoning. It comprises four
fundamental tasks plus an auxiliary vision task, and ships a domain text corpus
for domain-adaptive pretraining (DAPT).

> Supramolecular chemistry studies non-covalent host–guest assemblies that
> underpin drug delivery, chemical sensing, and in-vivo toxin sequestration.
> Designing host–guest systems is slow (days of dry-lab verification per pair);
> SupraBench probes whether LLMs can reason about these systems directly.

- 📄 **Paper:** [`arXiv:2606.13477`](https://arxiv.org/abs/2606.13477)
- 🤗 **Datasets:** [`huggingface.co/SupraBench`](https://huggingface.co/SupraBench)
- 💻 **Code:** [`github.com/Tianyi-Billy-Ma/SupraBench`](https://github.com/Tianyi-Billy-Ma/SupraBench)

## Tasks & datasets

| Dataset | Task | Description |
|---|---|---|
| [`SupraBench/bap`](https://huggingface.co/datasets/SupraBench/bap) | Binding Affinity Prediction | regress log *K*ₐ for a host–guest pair |
| [`SupraBench/tbs`](https://huggingface.co/datasets/SupraBench/tbs) | Top-Binder Selection | pick the strongest binder among 4 candidate guests |
| [`SupraBench/sid`](https://huggingface.co/datasets/SupraBench/sid) | Solvent Identification | 6-way solvent classification from structure |
| [`SupraBench/hgd`](https://huggingface.co/datasets/SupraBench/hgd) | Host-Guest Description | open-ended QA on host/guest property profiles |
| [`SupraBench/vqa`](https://huggingface.co/datasets/SupraBench/vqa) | Molecular Identification | auxiliary vision task: identify a molecule from its image |
| [`SupraBench/EU-PMC`](https://huggingface.co/datasets/SupraBench/EU-PMC) | Text corpus | ~16M-token supramolecular corpus for DAPT |
| [`SupraBench/Binding-Affinity`](https://huggingface.co/datasets/SupraBench/Binding-Affinity) | Comprehensive anchor | per-record binding data + host/guest SMILES, 2D, 3D, environment |

Each task config lives in `configs/tasks/` (one YAML per task × prompting
strategy: `base`, `fewshot`, `cot`).

## Setup

We use [**uv**](https://docs.astral.sh/uv/) for all Python dependency and
interpreter management. Python is pinned in `.python-version`.

```bash
# Install uv (once per machine)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the project venv and install base dependencies
uv sync

# Optional extras, install what you actually need:
uv sync --extra api     # OpenAI / Anthropic / httpx  (hosted-API inference)
uv sync --extra hf      # torch / transformers / accelerate / peft  (local inference + LoRA)
uv sync --extra vllm    # vLLM
uv sync --extra dev     # pytest / ruff
```

Secrets (API keys, HF tokens) are read from environment variables — never put
them in YAML or code.

## Running a task against a model

```bash
uv run python src/main.py \
    --task-config  configs/tasks/bap_base.yaml \
    --model-config configs/models/openrouter_qwen35_27b.yaml \
    --output-dir   outputs/
```

Results land at `outputs/<task>_<model>/{predictions.jsonl,metrics.json}`
(the entire `outputs/` tree is gitignored).

## Repository layout

```
SupraBench/
├── configs/
│   ├── tasks/            # one YAML per task × prompting strategy
│   ├── models/           # one YAML per evaluated model / backend
│   └── train/            # continued-pretraining (DAPT/LoRA) recipes
├── src/
│   ├── datasets/         # task-specific dataset loaders
│   ├── eval/             # task-specific evaluators + metrics
│   ├── inference/        # inference backends (OpenAI/OpenRouter, HF+PEFT, vLLM)
│   ├── models/           # model-specific glue (chat templates, stop tokens)
│   ├── train/            # continued-pretraining (LoRA) pipeline
│   ├── extras/           # shared code-level constants
│   ├── templates/        # prompt rendering helpers
│   ├── scripts/          # data construction, plotting, one-off tools
│   └── main.py           # entry point
├── scripts/              # result aggregation + analysis helpers
├── outputs/              # run artifacts (gitignored)
└── pyproject.toml        # uv-managed dependencies
```

Most subdirectories carry their own `README.md` documenting the local contract.

## Prompt templates

Prompts for every model go through `src/templates/` so layout stays identical
across evaluations. See the docstrings in
[`src/templates/template.py`](src/templates/template.py) for `generate_options`
and `generate_prompt` usage.

## Extending SupraBench

Adding a task or a model is driven by config + registration, never by editing
`main.py`. Each key is resolved through a string-based registry populated by the
`@register_dataset`, `@register_evaluator`, and `@register_backend` decorators.

**Add a task:**
1. `configs/tasks/<task>.yaml` with `dataset:` and `evaluator:` keys.
2. `src/datasets/<task>.py` — subclass `SupraDataset`, decorate with `@register_dataset("<key>")`.
3. `src/eval/<task>.py` — subclass `Evaluator`, decorate with `@register_evaluator("<key>")`.
4. Import both from their package `__init__.py` so registration runs.
5. Smoke test: `uv run python src/main.py --task-config ... --limit 2`.

**Add a model:**
1. `configs/models/<model>.yaml` with `backend:` + `model_id:` + `generation:`.
2. New delivery mechanism → add a backend under `src/inference/<backend>.py` decorated with `@register_backend(...)`.
3. Model quirks (chat template, stop tokens, response scrubbing) → add a helper under `src/models/<model>.py`.

## Authors

Tianyi Ma¹\*, Yijun Ma¹\*, Zehong Wang¹†, Weixiang Sun¹, Ziming Li², Connor R. Schmidt¹, Chuxu Zhang², Matthew J. Webber¹, Yanfang Ye¹†

¹ University of Notre Dame  ² University of Connecticut
\*Equal contribution  †Corresponding authors (`{tma2, yye7}@nd.edu`)

## Sources & license

SupraBench (code and curated benchmark data) is released under
[CC BY 4.0](./LICENSE).

Upstream data: binding records are derived from
[SupraBank](https://suprabank.org/) (CC-BY-4.0); the text corpus is built from
open-access [Europe PMC](https://europepmc.org/) articles subject to each
article's individual license; molecular structures use
[PubChem](https://pubchem.ncbi.nlm.nih.gov/) and
[OPSIN](https://github.com/dan2097/opsin).

## Citation

If you use SupraBench, please cite the paper and the upstream data sources.

```bibtex
@article{ma2026suprabench,
  title   = {SupraBench: A Benchmark for Supramolecular Host--Guest Chemistry Reasoning in Large Language Models},
  author  = {Ma, Tianyi and Ma, Yijun and Wang, Zehong and Sun, Weixiang and Li, Ziming and Schmidt, Connor R. and Zhang, Chuxu and Webber, Matthew J. and Ye, Yanfang},
  year    = {2026},
  eprint        = {2606.13477},
  archivePrefix = {arXiv},
  journal = {arXiv preprint arXiv:2606.13477}
}
```
