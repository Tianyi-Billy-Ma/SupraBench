"""OpenRouter inference backend (OpenAI-compatible).

Routes requests to OpenRouter's OpenAI-compatible chat-completions API,
which fronts GPT-4o, Claude, Gemini, etc. The API key is read from a
``.env`` file at the repository root via ``python-dotenv``.

When ``example.fewshot_demos`` is set, demos are interleaved as
alternating user/assistant turns before the final query turn, with each
demo's images encoded as base64 ``image_url`` content items.

Requires the ``api`` extra (provides ``openai`` + ``python-dotenv``)::

    uv sync --extra api

Setup::

    cp .env.example .env
    # edit .env to set OPENROUTER_API_KEY=sk-or-...

Config (from ``configs/models/<model>.yaml``)::

    backend: openrouter
    model_id: openai/gpt-4o          # see https://openrouter.ai/models
    generation:
      max_new_tokens: 2048           # bumped for reasoning models — see below
      temperature: 0.0
    reasoning:                       # optional, OpenRouter reasoning controls
      effort: low                    # "low" | "medium" | "high"
      exclude: false                 # if true, OR omits reasoning from response

For **reasoning models** (Qwen3.5, GPT-5.x, Claude 4.6 with thinking,
DeepSeek-R1, etc.) the model spends tokens on hidden reasoning before
emitting ``message.content``. If ``max_new_tokens`` is too low the
answer never lands in ``content`` (``finish_reason=length``). Two
mitigations baked in here:

* ``reasoning`` config above lets you cap the reasoning budget via OR's
  ``reasoning.effort=low``, freeing tokens for the final answer.
* When ``message.content`` is empty/null, this backend falls back to the
  ``message.reasoning`` text — so you still get something to score from
  even if the model never finished its thinking.
"""

from __future__ import annotations

import base64
import io
import os
import time
from typing import TYPE_CHECKING, Any

from .base import InferenceBackend, register_backend

if TYPE_CHECKING:
    from datasets.base import Example


_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0


def _encode_image(img: Any) -> str:
    """PIL.Image -> data URI. Lazy-imports PIL so the module parses without it."""
    buf = io.BytesIO()
    fmt = (getattr(img, "format", None) or "PNG").upper()
    save_fmt = "PNG" if fmt not in {"PNG", "JPEG", "WEBP"} else fmt
    img.save(buf, format=save_fmt)
    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[save_fmt]
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _user_content(prompt: str, images: list[Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": _encode_image(img)}})
    return content


@register_backend("openrouter")
class OpenRouterBackend(InferenceBackend):
    """OpenAI-compatible client pointed at OpenRouter."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Copy .env.example to .env at the "
                "repo root and fill in your OpenRouter key, or export the "
                "variable in your shell."
            )

        self.client = OpenAI(
            base_url=config.get("base_url", _DEFAULT_BASE_URL),
            api_key=api_key,
        )
        self.model_id = config["model_id"]

        gen_cfg = config.get("generation", {}) or {}
        self.max_tokens = gen_cfg.get("max_new_tokens", 2048)
        self.temperature = gen_cfg.get("temperature", 0.0)
        self.top_p = gen_cfg.get("top_p")
        # Reasoning controls (OpenRouter-specific) — passed via extra_body.
        # Format: {"effort": "low"} or {"max_tokens": 1024} or {"exclude": True}.
        # See https://openrouter.ai/docs/use-cases/reasoning-tokens
        self.reasoning_cfg = config.get("reasoning")

    def generate(self, example: "Example") -> str:
        messages = self._build_messages(example)

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.reasoning_cfg is not None:
            kwargs["extra_body"] = {"reasoning": self.reasoning_cfg}

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                # Some reasoning models (Qwen3.5, GPT-5.x with thinking, etc.)
                # leave content=null and put output into the reasoning field;
                # fall back so we always get something parseable.
                content = msg.content or ""
                if not content:
                    msg_dump = msg.model_dump() if hasattr(msg, "model_dump") else {}
                    content = msg_dump.get("reasoning") or ""
                return content
            except Exception as exc:  # noqa: BLE001 — retry on any transient failure
                last_exc = exc
                if attempt == _MAX_RETRIES - 1:
                    break
                time.sleep(_BACKOFF_BASE ** attempt)
        raise RuntimeError(f"OpenRouter request failed after {_MAX_RETRIES} attempts") from last_exc

    @staticmethod
    def _build_messages(example: "Example") -> list[dict[str, Any]]:
        """Assemble OpenAI/OR-style chat messages, including any fewshot demos."""
        messages: list[dict[str, Any]] = []
        for demo in (example.fewshot_demos or []):
            demo_imgs = list(demo.get("images") or [])
            messages.append({
                "role": "user",
                "content": _user_content(example.prompt, demo_imgs),
            })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": f"<answer>{demo['answer']}</answer>"}],
            })
        messages.append({
            "role": "user",
            "content": _user_content(example.prompt, list(example.images or [])),
        })
        return messages
