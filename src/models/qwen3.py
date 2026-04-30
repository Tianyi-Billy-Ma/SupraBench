"""Qwen3 model-specific glue: chat templating and thinking-mode cleanup.

Qwen3 supports a toggleable "thinking" mode where the model emits a
``<think>...</think>`` reasoning block before its final answer. This
module exposes two helpers so backends can stay model-agnostic:

- :func:`build_messages` — assemble the ``messages`` list fed to
  ``tokenizer.apply_chat_template(..., enable_thinking=...)``.
- :func:`strip_thinking` — drop the ``<think>...</think>`` span from a
  raw completion so the stored prediction is only the user-visible answer.

Keep anything tokenizer- or weight-specific here; the inference backends
should not know that Qwen3 has thinking blocks.
"""

from __future__ import annotations

import re

DEFAULT_SYSTEM_PROMPT = "You are an expert in supramolecular chemistry."

# Canonical stop strings. Backends that want explicit stopping can feed
# these to their generation config; the default chat template already
# handles end-of-turn tokens so this is optional.
STOP_STRINGS: tuple[str, ...] = ("<|im_end|>",)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def build_messages(
    user_prompt: str,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Build the chat-messages list for ``tokenizer.apply_chat_template``."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from a Qwen3 completion.

    Safe to call on non-thinking output — non-matching text is returned
    unchanged (with whitespace trimmed).
    """
    return _THINK_BLOCK.sub("", text).strip()
