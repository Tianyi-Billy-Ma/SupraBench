# SupraBench — Prompt Template

A small, dependency-free Python module for rendering consistent prompts across a supramolecular-chemistry benchmark. Teams call two helpers — `generate_options` and `generate_prompt` — to produce the exact same prompt layout for every model under test.

## Requirements

- Python 3.10+ (uses `list | None` union syntax)
- No third-party dependencies

## Project layout

```
SupraBench/
├── README.md
└── src/
    └── templates/
        ├── __init__.py
        └── template.py
```

The package lives under `src/templates/`. Put `src/` on your `PYTHONPATH` (or run Python from within `src/`) so the `templates` package can be imported:

```bash
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
```

## Quick start

```python
from templates import generate_options, generate_prompt

prompt = generate_prompt(
    "Which host-guest complex is more stable in aqueous solution: "
    "cucurbit[7]uril with adamantane, or beta-cyclodextrin with adamantane?"
)
print(prompt)
```

## API

### `generate_options(options: list[str]) -> str`

Renders multiple-choice options as `A. ...` / `B. ...` lines. Each option is labelled with a single uppercase letter, so pass at most 26 items. Concatenate the returned string into `query` before calling `generate_prompt`.

```python
options = generate_options([
    "beta-cyclodextrin",
    "cucurbit[7]uril",
    "calix[4]arene",
    "pillar[5]arene",
])
# A. beta-cyclodextrin.
# B. cucurbit[7]uril.
# C. calix[4]arene.
# D. pillar[5]arene.
```

### `generate_prompt(query, fewshot_examples=None, thinking=False) -> str`

Renders the full benchmark prompt.

| Argument | Type | Description |
| --- | --- | --- |
| `query` | `str` | The question. For multiple-choice items, concatenate `generate_options(...)` onto the question text before passing it in. |
| `fewshot_examples` | `list[dict[str, str]] \| None` | Optional list of few-shot examples. Each element must be a dict with keys `"query"` and `"answer"`. |
| `thinking` | `bool` | If `True`, appends a chain-of-thought cue. |

The final prompt is stripped of leading/trailing whitespace, and runs of 3+ blank lines are collapsed to a single blank line.

## Examples

### 1. Plain query (no options, no few-shot, no CoT)

```text
You are an expert in supramolecular chemistry.
Your task is to answer the following question.
Question: Which host-guest complex is more stable in aqueous solution: cucurbit[7]uril with adamantane, or beta-cyclodextrin with adamantane?

Put your final answer between <answer></answer>
```

### 2. Query + multiple-choice options

```text
You are an expert in supramolecular chemistry.
Your task is to answer the following question.
Question: Which macrocycle has the highest binding affinity for adamantane in water?
A. beta-cyclodextrin.
B. cucurbit[7]uril.
C. calix[4]arene.
D. pillar[5]arene.

Put your final answer between <answer></answer>
```

### 3. Query + few-shot examples

```text
You are an expert in supramolecular chemistry.
Your task is to answer the following question.
Question: What non-covalent interaction dominates the binding of ferrocene inside cucurbit[7]uril?
Below are some examples that you should follow.

Question: What drives the inclusion of adamantane in beta-cyclodextrin?
<answer>hydrophobic effect</answer>

Question: What interaction is most important for crown-ether/alkali-metal binding?
<answer>ion-dipole</answer>

Put your final answer between <answer></answer>
```

### 4. Query + chain-of-thought

```text
You are an expert in supramolecular chemistry.
Your task is to answer the following question.
Question: Rank the binding affinity of methylviologen with cucurbit[7]uril vs. cucurbit[8]uril.

Let's think step by step.

Put your final answer between <answer></answer>
```

### 5. Everything on: options + few-shot + CoT

```text
You are an expert in supramolecular chemistry.
Your task is to answer the following question.
Question: Which guest binds most tightly inside cucurbit[7]uril in water?
A. methane.
B. adamantane.
C. benzene.
D. ferrocene.
Below are some examples that you should follow.

Question: Which guest binds most tightly inside beta-cyclodextrin?
A. methane.
B. adamantane.
C. benzene.
<answer>B</answer>
Let's think step by step.

Put your final answer between <answer></answer>
```

### 6. Edge case: 6 options (beyond A–D)

```text
You are an expert in supramolecular chemistry.
Your task is to answer the following question.
Question: Which of the following is NOT a supramolecular macrocycle?
A. cucurbit[n]uril.
B. cyclodextrin.
C. calixarene.
D. pillararene.
E. crown ether.
F. polyethylene.

Put your final answer between <answer></answer>
```

