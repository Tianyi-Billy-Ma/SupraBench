"""Model-specific glue code (tokenizer quirks, chat templates, stop tokens).

Most models work with the generic inference backends in ``src/inference``.
Anything model-specific — non-standard chat templates, unusual stop tokens,
post-processing of raw completions — lives here. See :mod:`models.qwen3`
for the canonical reference.
"""

from . import qwen3

__all__ = ["qwen3"]
