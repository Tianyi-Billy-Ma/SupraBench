"""OpenRouter inference backend.

Reads credentials from the environment — never hardcode keys here:
  export OPENROUTER_API_KEY=sk-or-v1-...

Model aliases allow reasoning-mode control without changing configs:

  model_id: or-deepseek-v4-pro-no-thinking  →  deepseek/deepseek-v4-pro + reasoning disabled
  model_id: or-qwen3.5-9b-no-thinking       →  qwen/qwen3.5-9b + reasoning disabled

Usage in a model YAML::

    backend: openrouter
    model_id: anthropic/claude-sonnet-4-6
    generation:
      max_completion_tokens: 8192

Concurrency is handled by ``main.py`` (``--concurrency`` flag), which
calls ``generate()`` from a ``ThreadPoolExecutor``.

The ``openai`` package is lazy-imported inside ``__init__`` so the
backend registry can be loaded in environments without the ``api``
extra installed (e.g. CRC training nodes that only have the ``hf``
extra).
"""

from __future__ import annotations

import os
import time
from typing import Any

from .base import InferenceBackend, register_backend

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Aliases: logical name → {model_id, extra_body}
# extra_body → merged into extra_body dict for OpenRouter reasoning params
MODEL_ALIASES: dict[str, dict[str, Any]] = {
    # ── GPT ──────────────────────────────────────────────────────────────────────
    "or-gpt-5.4-nano-xhigh": {
        "model_id":   "openai/gpt-5.4-nano",
        "extra_body": {"reasoning": {"effort": "xhigh"}},
    },
    # ── DeepSeek ────────────────────────────────────────────────────────────────
    "or-deepseek-v4-pro-no-thinking": {
        "model_id":   "deepseek/deepseek-v4-pro",
        "extra_body": {"reasoning": {"enabled": False}},
    },
    # ── Gemini ───────────────────────────────────────────────────────────────────
    "gemini-3-flash-no-thinking": {
        "model_id":   "google/gemini-3-flash-preview",
        "extra_body": {"reasoning": {"enabled": False}},
    },
    # ── Qwen3.5 ──────────────────────────────────────────────────────────────────
    "or-qwen3.5-9b-no-thinking": {
        "model_id":   "qwen/qwen3.5-9b",
        "extra_body": {"reasoning": {"enabled": False}},
    },
    "or-qwen3.5-27b-no-thinking": {
        "model_id":   "qwen/qwen3.5-27b",
        "extra_body": {"reasoning": {"enabled": False}},
    },
}


@register_backend("openrouter")
class OpenRouterBackend(InferenceBackend):
    """OpenRouter chat-completions backend (OpenAI-compatible API)."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        # Lazy import: keeps the registry import safe in envs without
        # the `api` extra (e.g. CRC nodes with only `hf` installed).
        from openai import OpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self._client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

        model_id = config["model_id"]
        alias = MODEL_ALIASES.get(model_id, {})
        self._model_id   = alias.get("model_id", model_id)
        self._extra_body: dict[str, Any] = alias.get("extra_body", {})

        gen = config.get("generation", {}) or {}
        self._max_tokens  = gen.get("max_completion_tokens", 8192)
        self._temperature = gen.get("temperature", None)

    def generate(self, prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model":                self._model_id,
            "messages":             [{"role": "user", "content": prompt}],
            "max_completion_tokens": self._max_tokens,
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body

        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return (resp.choices[0].message.content or "").strip() if resp.choices else ""
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise e
        return ""
