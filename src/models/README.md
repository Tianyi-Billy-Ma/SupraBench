# `src/models/`

Model-specific logic that does not belong in a generic inference backend.

The split vs. `src/inference/`:

- `src/inference/` — **how** to call a model (HF pipeline, vLLM, OpenAI API).
- `src/models/` — **what is special about a particular model** (chat
  template, stop tokens, response cleanup, tokenizer quirks).

A model that slots cleanly into a backend does not need a file here.
Add a module only when you have per-model logic that would otherwise
bleed into the backend implementation.

## Reference: `qwen3.py`

The Qwen3 family supports a toggleable "thinking" mode that emits a
`<think>...</think>` block before the final answer. Rather than teach
every backend about this, we concentrate Qwen3-specific helpers here:

| Helper | Purpose |
| --- | --- |
| `build_messages(user_prompt, system_prompt=...)` | Assemble the chat-messages list fed to `tokenizer.apply_chat_template(..., enable_thinking=...)`. |
| `strip_thinking(text)` | Remove `<think>...</think>` spans from a raw completion so the stored prediction is only the user-visible answer. |
| `DEFAULT_SYSTEM_PROMPT` | Benchmark-wide default system prompt for Qwen3. |
| `STOP_STRINGS` | Canonical stop strings for explicit early stopping. |

The [`example` inference backend](../inference/example.py) uses these
helpers directly; see `configs/models/qwen3.yaml` for how to toggle
thinking mode from a model YAML.

## Adding a new model-specific module

1. `src/models/<model>.py` — expose pure functions (no side effects at
   import time, no GPU allocation).
2. Re-export from `src/models/__init__.py` so the module is visible via
   `from models import <model>`.
3. Reference it from the backend that needs it (not from `main.py` — the
   backend owns per-model quirks).
