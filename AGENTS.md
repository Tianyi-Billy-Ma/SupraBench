# SupraBench — Agent / Collaborator Guide

This file is the canonical contributor handbook for SupraBench, shared by
human collaborators and by AI coding agents (Claude Code reads it via the
top-level `CLAUDE.md` which points here).

## 1. What we are building

SupraBench is a benchmark of **seven supramolecular-chemistry tasks** used
to evaluate LLM performance (Qwen3, ChatGPT, and others). It is a
**multi-contributor project** — treat every file as something a teammate
will read next week.

## 2. Repository map

```
SupraBench/
├── configs/
│   ├── tasks/           # one YAML per benchmark task
│   └── models/          # one YAML per evaluated model
├── src/
│   ├── datasets/        # task-specific dataset loaders
│   ├── eval/            # task-specific evaluators
│   ├── inference/       # inference backends (OpenAI, HF, vLLM, ...)
│   ├── models/          # model-specific glue (chat templates, stop tokens)
│   ├── train/           # fine-tuning pipelines (placeholder)
│   ├── extras/          # code-level constants shared across modules
│   ├── templates/       # prompt rendering (generate_prompt / generate_options)
│   └── main.py          # entry point
├── scripts/
│   ├── crc/             # Anonymous HPC Cluster submission scripts
│   └── delta/           # NCSA Delta (UIUC) submission scripts
├── outputs/             # run artifacts (gitignored)
├── pyproject.toml       # uv-managed dependencies
└── AGENTS.md            # you are here
```

Every subdirectory has its own `README.md` — read it before editing that
area.

## 3. Environment

We use **uv** exclusively for dependency management.

```bash
# First-time setup
uv sync                    # base deps only
uv sync --extra api        # + OpenAI / Anthropic / httpx
uv sync --extra hf         # + torch / transformers / accelerate
uv sync --extra vllm       # + vLLM
uv sync --extra dev        # + pytest / ruff

# Run anything via uv so the pinned interpreter is used
uv run python src/main.py --task-config configs/tasks/bap.yaml \
                          --model-config configs/models/qwen3.yaml
```

- Python is pinned in `.python-version`.
- `uv.lock` is committed when present — do not edit it by hand; regenerate
  with `uv lock`.
- Never commit `.venv/` or API keys.

## 4. How a run is wired

`src/main.py` reads two YAMLs:

1. `configs/tasks/<task>.yaml` — picks a **dataset** key and an
   **evaluator** key.
2. `configs/models/<model>.yaml` — picks an **inference backend** key.

Each key is resolved through a string-based registry populated by the
`@register_dataset`, `@register_evaluator`, and `@register_backend`
decorators. Adding a new task or model therefore **never requires editing
`main.py`**.

Results are written to the flat path `outputs/<task>_<model>/`
(gitignored).

## 5. Extension checklists

### Add a new task

1. `configs/tasks/<task>.yaml` with `dataset:` and `evaluator:` keys.
2. `src/datasets/<task>.py` — subclass `SupraDataset`, decorate with
   `@register_dataset("<key>")`.
3. `src/eval/<task>.py` — subclass `Evaluator`, decorate with
   `@register_evaluator("<key>")`.
4. Import both from their package `__init__.py` so registration runs.
5. Smoke test: `uv run python src/main.py --task-config ... --limit 2`.

### Add a new model

1. `configs/models/<model>.yaml` with `backend:` + `model_id:` +
   `generation:`.
2. If the delivery mechanism is new, add a backend under
   `src/inference/<backend>.py` decorated with `@register_backend(...)`.
3. If the model has quirks (chat template, stop tokens, response scrubbing),
   add a helper under `src/models/<model>.py`.

### Add cluster scripts

1. Pick the right subdirectory (`scripts/crc/` or `scripts/delta/`).
2. Keep cluster-specific env setup inside that subdirectory; do not leak
   cluster paths into `src/`.
3. Every script ends in a `uv run python src/main.py ...` line so local
   and cluster runs match byte-for-byte.

## 6. Collaboration etiquette

- **Small PRs.** One task or one model per PR where possible.
- **Configs first.** When proposing a new task or model, land the YAML
  plus a stub loader/backend before writing heavy logic — it forces the
  interface discussion up front.
- **Don't rename registry keys** once data has been produced under them;
  the keys appear in `outputs/<task>_<model>/` paths.
- **Prompts go through `src/templates/`.** Never hand-format a prompt
  inline; divergence between models is the one thing a benchmark cannot
  tolerate.
- **Outputs are disposable.** Anyone may `rm -rf outputs/<task>_<model>/`
  and rerun. Treat anything inside `outputs/` as reproducible.
- **Secrets** (API keys, HF tokens) live in environment variables, never
  in YAML or code.

## 7. Running quality checks

```bash
uv run pytest              # once tests land
uv run ruff check src/     # lint
uv run ruff format src/    # auto-format
```

Please run these before opening a PR.


<claude-mem-context>
# Memory Context

# [SupraBench] recent context, 2026-05-24 8:52am EDT

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision 🚨security_alert 🔐security_note
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (19,884t read) | 438,629t work | 95% savings

### May 24, 2026
S672 Fix visual issues in molecular identification heatmap: font sizes too small, x-axis label arrow spacing, and unwanted ΔHeavy vertical separator line (May 24 at 5:52 AM)
S673 Fix molecular identification heatmap visuals: font sizes, x-axis label arrow spacing, ΔHeavy separator — then drop ΔHeavy column entirely per follow-up decision (May 24 at 5:59 AM)
S674 Rotate heatmap x-axis labels and align them to the end (right edge) of each cell, not the center (May 24 at 6:03 AM)
S675 Restructure molecular identification results: move vision table to appendix, make heatmap the main-body primary display, and rotate x-axis labels to right-edge alignment (May 24 at 6:07 AM)
S676 Overleaf paper asset audit and cleanup: remove unused figures/tables, move example_record to figures/, add rm command to sync.py (May 24 at 6:09 AM)
1689 6:17a ✅ 7 Stale Table Files Deleted from Overleaf Remote; 10 Figure Files Were Already Absent
1690 " 🔵 Overleaf Remote Now Clean: 0 Remote-Only Files, 38 Files in Sync
S677 Replace original evaluation metrics section in SupraBench paper appendix with Version 2 (May 24 at 6:17 AM)
1691 6:21a ✅ Evaluation Metric Version 2 Promoted to Replace Original in Appendix
1692 " 🔵 SupraBench Appendix Contains Two Evaluation Metric Sections Side-by-Side
1693 " 🔵 Evaluation Metric Labels Not Cross-Referenced Elsewhere in Paper
1694 6:22a ✅ Original Evaluation Metrics Section Marked Deprecated in appendix.tex
1695 " ✅ Evaluation Metrics v1 Deleted and v2 Promoted as Canonical Section in SupraBench Appendix
S678 Restructure Molecular Identification section in experiments.tex to match house style with numbered observations and italic takeaways (May 24 at 6:23 AM)
1696 6:26a ✅ Molecular Identification Section in experiments.tex Restructured with Numbered Observations
S679 Standardize domain adaptation terminology from CPT to DAPT across the SupraBench paper LaTeX source (May 24 at 6:27 AM)
1697 6:29a 🔵 SupraBench Paper CPT Findings in LaTeX Source
1698 6:30a 🔵 SupraBench CPT Training Recipe Details
1699 " ✅ Removed Metrics Description Sentence from Molecular Identification Section
1700 " ✅ Terminology Standardized from CPT to DAPT in Domain Adaptation Section
1701 " ✅ CPT→DAPT Terminology Rename Completed Across Both Paper Files
1702 6:31a 🔵 Two Remaining CPT References Found After DAPT Rename
1703 " ✅ domain_adaptation.tex Table Caption Updated from CPT to DAPT
1704 " ✅ CPT→DAPT Rename Fully Complete Across All Paper Files
1705 " ✅ CPT→DAPT Rename Verified Clean and Pushed to Overleaf
S680 SupraBench paper restructuring: replace three-application framing with four formal task names throughout all paper files (May 24 at 6:31 AM)
1706 6:40a ⚖️ Paper Restructuring: Tasks Decoupled from Applications, Formal Task Names Required
1707 6:41a ✅ Overleaf Sync and Pre-Restructure Backup Created
1708 " 🔵 Audit of Task-ID and Application-Framing Mentions Across Paper TeX Files
1709 " 🔵 Full Scope of Application-Framing and Task-ID Changes Identified Across Paper
1710 6:42a 🔵 main_results.tex Table Structure: Column Groups Use Application Labels, Caption References "Three Applications"
1711 6:44a 🔵 domain_adaptation.tex Table Structure: Same Application Labels as Main Results
1712 " ✅ abstract.tex: Replaced Application-Pillar Framing with Four Formal Task Names
1713 " ✅ introduction.tex: Task Names Replace Application Pillars; Task Count Corrected from Six to Five
1714 6:45a ✅ introduction.tex, conclusion.tex, appendix.tex: Application-Pillar Framing Removed from Three More Sections
1715 " ✅ benchmark.tex: Task Construction Section Fully Restructured from Three Application Paragraphs to Four Task Paragraphs
1716 " ✅ experiments.tex: Main Results Findings Updated to Use Formal Task Names
1717 " ✅ experiments.tex: DAPT Section, Subsection Title, and Figure Caption Updated with Formal Task Names
1718 " ✅ Both Result Tables Updated: Application Column Headers Replaced with Formal Task Names
1719 " 🔵 Verification Pass: All Task-ID and Application-Label References Successfully Removed
1720 6:46a 🔵 Final Verification: Remaining Application Mentions Are All Compliant Context-Setting Uses
1721 " ✅ All 8 Restructured Paper Files Pushed to Overleaf
S681 SupraBench paper task restructuring complete — four formal task names replace three-application framing throughout all paper files, Ultrawork mode cancelled (May 24 at 6:47 AM)
1722 6:58a 🔵 SupraBench Project Structure with Overleaf Paper Integration
1723 " 🔵 Overleaf-Sync Skill: Git Bridge Architecture for Two-Way Paper Sync
1724 " 🔵 paper-overleaf/ Has Significant Uncommitted Local Changes Not Yet Pushed to Overleaf
1725 6:59a 🔵 scripts/overleaf/sync.py Uses pyoverleaf Cookie API, Not Git Bridge
1726 " 🔵 paper-overleaf/ Is in True Diverged State — Both Local and Overleaf Remote Have Unpushed Commits
1727 " ✅ SupraBench Paper Task Taxonomy Restructured from Application Pillars to Standalone Task Names
1728 " 🔵 SupraBench Paper: Key Experimental Findings and Remaining TODOs
1729 7:00a 🔵 pyoverleaf pull --dry Confirms Local Working Tree Matches Overleaf Remote Content
1730 " 🔵 All Paper Cross-References Resolve Correctly; No Dangling Labels
1731 7:01a 🔵 All Evaluation Metrics Fully Defined in Appendix — benchmark.tex \todo{} Needs Only a Summary or Cross-Reference
1732 " 🔴 Filled benchmark.tex Evaluation Metrics \todo{} and Fixed Appendix Typo "Detaled → Detailed"
1733 " 🔵 Appendix Contains Full 7-Step Dataset Cleaning Pipeline with Van't Hoff Temperature Correction
1734 7:02a 🟣 Evaluation Metrics Fix and Appendix Typo Pushed to Overleaf Successfully
1735 8:50a 🔵 SupraBench Analysis Scripts Directory Contains Single File
1736 " 🔵 SupraBench Task1 Analysis Script: Results Stored in results/task1 and results/analysis
1737 8:51a 🟣 New Script: compute_extra_table_metrics.py for TBS Regret and SID Balanced Accuracy
1738 " 🔵 TBS Regret and SID Balanced Accuracy Metrics Computed for All Models

Access 439k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>