import re


BASE_TEMPLATE = """
You are an expert in supramolecular chemistry.
Your task is to answer the following question.
Question: {query}
{fewshot}
{cot}

Put your final answer between <answer></answer>
"""

OPTION_TEMPLATE = """{idx}. {option}."""

FEWSHOT_TEMPLATE = """
Below are some examples that you should follow.

{examples}

"""

EXAMPLE_TEMPLATE = """
Question: {query}
<answer>{answer}</answer>
"""


COT_TEMPLATE = """
Let's think step by step.
"""


def generate_options(options: list[str]) -> str:
    """Render multiple-choice options as ``A. ...`` / ``B. ...`` lines.

    Args:
        options: The option texts, in order. Must contain at most 26 items,
            since each is labelled with a single uppercase letter A-Z.

    Returns:
        The rendered options joined by newlines, ready to concatenate into
        the ``query`` passed to ``generate_prompt``.
    """
    lines = [
        OPTION_TEMPLATE.format(idx=chr(ord("A") + i), option=option)
        for i, option in enumerate(options)
    ]
    return "\n".join(lines)


def generate_prompt(
    query: str,
    fewshot_examples: list[dict[str, str]] | None = None,
    thinking: bool = False,
) -> str:
    """Render the full benchmark prompt.

    Args:
        query: The question to ask the model. For multiple-choice items, render
            options with ``generate_options`` and concatenate them into ``query``
            before calling this function.
        fewshot_examples: Optional list of few-shot examples. Each element must
            be a dict with keys ``"query"`` (str) and ``"answer"`` (str).
        thinking: If True, append the chain-of-thought cue to the prompt.

    Returns:
        The final prompt string, stripped of leading/trailing whitespace.
    """
    if fewshot_examples:
        examples = [
            EXAMPLE_TEMPLATE.format(query=example["query"], answer=example["answer"]).strip()
            for example in fewshot_examples
        ]
        examples = "\n\n".join(examples)
        fewshot_prompt = FEWSHOT_TEMPLATE.format(examples=examples).strip()
    else:
        fewshot_prompt = ""

    cot_prompt = COT_TEMPLATE.strip() if thinking else ""
    prompt = BASE_TEMPLATE.format(query=query, fewshot=fewshot_prompt, cot=cot_prompt)
    return re.sub(r"\n{3,}", "\n\n", prompt).strip()
