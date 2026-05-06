"""OpenAI / OpenRouter inference backend.

Reads credentials from environment variables — never hardcode keys here:
  export OPENAI_API_KEY=sk-...
  export OPENROUTER_API_KEY=sk-or-v1-...

Model aliases allow reasoning-mode control without changing configs:

  model_id: gpt-5.4-nano-xhigh   →  gpt-5.4-nano + reasoning_effort=xhigh
  model_id: or-deepseek-v4-pro-no-thinking  →  deepseek/deepseek-v4-pro + reasoning disabled

Usage in a model YAML::

    backend: openai
    model_id: gpt-5.4-mini
    provider: openai          # "openai" or "openrouter"
    generation:
      max_completion_tokens: 8192

Concurrency is handled by ``main.py`` (``--concurrency`` flag), which
calls ``generate()`` from a ``ThreadPoolExecutor``.
"""

from __future__ import annotations

import os
import time
from typing import Any

from openai import OpenAI

from .base import InferenceBackend, register_backend

OPENAI_BASE_URL     = "https://api.openai.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Aliases: logical name → {model_id, extra_kwargs, extra_body}
# extra_kwargs  → merged into top-level create() kwargs (e.g. reasoning_effort)
# extra_body    → merged into extra_body dict (OpenRouter reasoning params)
MODEL_ALIASES: dict[str, dict[str, Any]] = {
    # ── OpenAI direct ────────────────────────────────────────────────────────
    "gpt-5.4-nano-no-thinking": {
        "model_id":     "gpt-5.4-nano",
        "extra_kwargs": {"reasoning_effort": "none"},
    },
    "gpt-5.4-nano-xhigh": {
        "model_id":     "gpt-5.4-nano",
        "extra_kwargs": {"reasoning_effort": "xhigh", "max_completion_tokens": 32768},
    },
    # ── OpenRouter — GPT ─────────────────────────────────────────────────────
    "or-gpt-5.4-nano-no-thinking": {
        "model_id":   "openai/gpt-5.4-nano",
        "extra_body": {"reasoning": {"enabled": False}},
    },
    "or-gpt-5.4-nano-xhigh": {
        "model_id":   "openai/gpt-5.4-nano",
        "extra_body": {"reasoning": {"effort": "xhigh"}},
    },
    # ── OpenRouter — DeepSeek ────────────────────────────────────────────────
    "or-deepseek-v4-pro-no-thinking": {
        "model_id":   "deepseek/deepseek-v4-pro",
        "extra_body": {"reasoning": {"enabled": False}},
    },
    # ── OpenRouter — Gemini ──────────────────────────────────────────────────
    "gemini-3-flash-no-thinking": {
        "model_id":   "google/gemini-3-flash-preview",
        "extra_body": {"reasoning": {"enabled": False}},
    },
    # ── OpenRouter — Qwen3.5 ────────────────────────────────────────────────
    "or-qwen3.5-9b-no-thinking": {
        "model_id":   "qwen/qwen3.5-9b",
        "extra_body": {"reasoning": {"enabled": False}},
    },
    "or-qwen3.5-27b-no-thinking": {
        "model_id":   "qwen/qwen3.5-27b",
        "extra_body": {"reasoning": {"enabled": False}},
    },
}


@register_backend("openai")
class OpenAIBackend(InferenceBackend):
    """OpenAI or OpenRouter chat-completions backend."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        provider = config.get("provider", "openai")
        if provider == "openrouter":
            api_key  = os.environ.get("OPENROUTER_API_KEY", "")
            base_url = OPENROUTER_BASE_URL
        else:
            api_key  = os.environ.get("OPENAI_API_KEY", "")
            base_url = OPENAI_BASE_URL

        self._client = OpenAI(api_key=api_key, base_url=base_url)

        # Resolve alias
        model_id = config["model_id"]
        alias = MODEL_ALIASES.get(model_id, {})
        self._model_id    = alias.get("model_id", model_id)
        self._extra_kwargs: dict[str, Any] = alias.get("extra_kwargs", {})
        self._extra_body:   dict[str, Any] = alias.get("extra_body", {})

        gen = config.get("generation", {}) or {}
        self._max_tokens = self._extra_kwargs.pop("max_completion_tokens", None) \
                           or gen.get("max_completion_tokens", 8192)
        self._temperature = gen.get("temperature", None)

    def generate(self, prompt: str) -> str:
        kwargs: dict[str, Any] = {
            "model":    self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": self._max_tokens,
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        kwargs.update(self._extra_kwargs)

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
