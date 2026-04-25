# `src/inference/`

Thin wrappers around actual model-serving stacks. One backend per
delivery mechanism, not per model — multiple models share a backend by
pointing to the same registry key in their YAML.

## Contract

```python
from inference import register_backend, InferenceBackend

@register_backend("my_backend")
class MyBackend(InferenceBackend):
    def generate(self, prompt: str) -> str:
        ...
```

`build_inference_backend(config)` returns the class matching
`config["backend"]` in the model YAML. Remaining config fields (model id,
decoding kwargs, etc.) are available via `self.config`.

Model-specific quirks (chat templates, stop tokens, response scrubbing)
belong in [`src/models/`](../models/), not here.

## Reference: `example.py`

A full working backend using HuggingFace Transformers plus
`device_map="auto"` for **single-node multi-GPU** inference (naive model
parallel via Accelerate). Handles Qwen3-style chat templating and
thinking-mode cleanup through [`models.qwen3`](../models/qwen3.py).

```bash
# Install the required extras
uv sync --extra hf

# Run Qwen3 across all visible GPUs (set CUDA_VISIBLE_DEVICES to restrict)
uv run python src/main.py \
    --task-config  configs/tasks/task1.yaml \
    --model-config configs/models/qwen3.yaml
```

Relevant knobs in the model YAML:

| Key | Purpose |
| --- | --- |
| `model_id` | HF repo id (e.g. `Qwen/Qwen3-8B`). |
| `dtype` | `auto`, `bfloat16`, `float16`, …. |
| `device_map` | `auto` / `balanced` / `cuda:0` / explicit dict. `auto` shards across all visible GPUs. |
| `enable_thinking` | Toggle Qwen3 thinking mode on the tokenizer chat template. |
| `strip_thinking` | Drop `<think>...</think>` from the stored prediction (default true). |
| `system_prompt` | Override the Qwen3 default system prompt. |
| `generation.*` | Passed directly to `model.generate`. |

## Scaling beyond `device_map="auto"`

Naive model parallel keeps most GPUs idle during each other's forward
pass. For higher throughput on larger models, register a separate
backend that uses **tensor parallelism** — for example vLLM:

```python
@register_backend("vllm")
class VLLMBackend(InferenceBackend):
    def __init__(self, config):
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=config["model_id"],
            tensor_parallel_size=config.get("tensor_parallel_size", 1),
            dtype=config.get("dtype", "auto"),
        )
        ...
```

vLLM handles single-node tensor parallel internally — no `torchrun`
launch required.

## Planned backend keys

| `backend:` key | Targets | Status | Extras |
| --- | --- | --- | --- |
| `example` | Qwen3 via HF Transformers + `device_map="auto"` | ✅ implemented (`example.py`) | `hf` |
| `vllm` | Same, but tensor-parallel via vLLM | planned | `vllm` |
| `openai` | ChatGPT / GPT-4 / o-series | planned | `api` |
| `anthropic` | Claude family | planned | `api` |
