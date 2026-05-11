# Task 1 — representative failure modes (v2 adapter)

Three diagnostic examples pulled from
`outputs/task1_base_qwen3.5-27b-supra-v2-lora/predictions.jsonl`. Each
illustrates one of the three dominant failure modes.

## Mode 1 — Parse failure (no answer emitted at all)

- **Prompt:** Predict logKa for *tetra-endo-ethyl octa acid* + *4-Methylbenzoic acid*. "Put your final answer between `<answer></answer>`."
- **Reference:** 2.435
- **Parsed:** `None` (no number extractable)

```
Here's a thinking process that leads to the suggested answer:

1.  **Analyze the Request:**
    *   **Role:** Expert in supramolecular chemistry.
    *   **Task:** Predict the binding affinity (logKa) between a specific host and guest...
    *   **Format:** <answer>value</answer>.

2.  **Analyze the Host:**
    [...512 tokens of internal monologue, never produces <answer>...]
```

Output exhausts `max_new_tokens=512` without ever opening an `<answer>` tag.
~12 % of task-1 examples behave this way under the v2 adapter.

## Mode 2 — Far-wrong (parser grabs an irrelevant number)

- **Prompt:** Predict logKa for *Acyclic naphthalene-tetrameric Cucurbituril* + *Rocuronium*.
- **Reference:** 9.531
- **Parsed:** 4.0 → error 5.53 logKa units

```
The user wants me to predict the binding affinity (logKa) between a specific
host and a specific guest in aqueous solution.

Host: Acyclic ... Cucurbituril (tetrameric glycourilunit).
...
"Cucurbituril (tetrameric glycourilunit)" usually refers to Cucurbit[4]uril (CB[4]).
However, the prefix "Acyclic ..." suggests a modified cucurbituril...
[runs out of tokens still debating which CB[n] is being asked about]
```

The parser's fallback ("first numeric in the answer text after temperature
mentions are stripped") grabs the **4** from "Cucurbit[4]uril" — totally
unrelated to logKa.

## Mode 3 — Off-by-some (parser grabs a structural dimension)

- **Prompt:** Predict logKa for *Cucurbit[5]uril* + *Sanguinarine*.
- **Reference:** 4.083
- **Parsed:** 5.0 → error 0.92

```
The user wants me to predict the binding affinity (logKa) between
Cucurbit[5]uril (CB[5]) and Sanguinarine in aqueous solution.

1. Analyze the Host: Cucurbit[5]uril (CB[5]).
   * Cavity diameter is approximately 5.3 Å.
   * Typical binding constants (K_a) for CB[5] range from 10^4 to 10^7 M^-1...
[truncated mid-reasoning, never emits <answer>]
```

Parser picks up the **5** from "Cucurbit[5]uril" (or the 5.3 Å cavity size).
This particular example is close to truth (4.08) so contributes only 0.92 to
MAE — but the predicted value comes from the host name, not from any reasoning
about binding.

## Common signature

All three modes share:
1. Output opens with either `"The user wants to predict..."` or
   `"Here's a thinking process that leads to..."` (the internal-monologue
   artifact baked into base Qwen3.5-27B's pretraining; CPT on raw academic
   prose amplifies rather than suppresses it).
2. Body is elaborate, often-correct chemistry reasoning.
3. **No final `<answer>X</answer>` ever emitted.** Parser falls back to
   "first number in text" — which is structural (cavity size, the `N` in
   CB[N], etc.), not a logKa estimate.

This is format collapse, not knowledge collapse: the chemistry knowledge is
in the reasoning, but the model never compresses it into the required output
shape.
