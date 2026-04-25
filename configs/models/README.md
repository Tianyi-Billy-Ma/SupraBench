# `configs/models/`

One YAML per evaluated model. Each file is consumed by `src/main.py` and
picks an inference backend plus its decoding parameters.

## Schema (minimum viable)

```yaml
name: qwen3-8b               # logical name; used in outputs/<task>_<model>/
backend: hf                  # registered via @register_backend in src/inference/
model_id: Qwen/Qwen3-8B      # HuggingFace repo id, OpenAI model id, etc.
generation:
  max_new_tokens: 1024
  temperature: 0.0
  top_p: 1.0
```

Backends pick up whatever keys they need from `self.config`; add API keys
via environment variables, **never** in the YAML.

## Adding a model

1. Pick the backend — reuse an existing entry in `src/inference/` if
   possible; only add a new backend when the delivery mechanism is new.
2. Copy an existing YAML and edit `name`, `model_id`, and `generation`.
3. If the model needs bespoke handling (odd chat template, custom stop
   tokens), drop a module into `src/models/` and reference it from the
   YAML.
