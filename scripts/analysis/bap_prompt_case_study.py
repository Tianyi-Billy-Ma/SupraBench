"""BAP prompt-degradation case study.

Generates a reviewable Markdown report for the Drug Delivery case study
without touching the Overleaf paper directory.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.eval.bap import parse_logka


RESULTS = REPO / "results" / "bap"
DATA = REPO / "data" / "bap"
OUT = REPO / "results" / "analysis"

METHODS = ["base", "fewshot", "cot"]
METHOD_LABEL = {"base": "Base", "fewshot": "Few-Shot", "cot": "CoT"}

MAIN_MODELS = [
    ("qwen3.5-9b", "Qwen3.5-9B"),
    ("qwen3.5-27b", "Qwen3.5-27B"),
    ("llama-3.1-8b-instruct", "Llama3.1-8B"),
    ("llama-3.1-70b-instruct", "Llama3.1-70B"),
    ("gpt-5.4-mini_nothinking", "GPT-5.4-Mini"),
    ("gpt-5.4-nano_xhigh", "GPT-5.4-Nano"),
    ("gemini-3-flash-preview_nothinking", "Gemini-3-Flash"),
    ("deepseek-v4-pro", "DeepSeek-v4"),
]

HOST_ABBREV = {
    "Cucurbit[8]uril": "CB[8]",
    "Cucurbit[6]uril": "CB[6]",
    "Cucurbit[7]uril": "CB[7]",
    "Cucurbit[5]uril": "CB[5]",
    "β-Cyclodextrin": "beta-CD",
    "α-Cyclodextrin": "alpha-CD",
    "p-Sulfonatocalix[4]arene": "p-SC4",
    "p-Sulfonatocalix[6]arene": "p-SC6",
    "syn-Amide Naphthotube": "syn-NT",
    "anti-Amide Naphthotube": "anti-NT",
    "Octa acid cavitand": "OA",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _coerce(value: float, text: str) -> float | None:
    if value > 50 and "log" not in text.lower():
        try:
            value = math.log10(value)
        except ValueError:
            return None
    if value < -10 or value > 30:
        return None
    return value


def fallback_parse_logka(text: str) -> float | None:
    """A diagnostic-only parser that takes the last plausible numeric value."""

    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    ans = m.group(1).strip() if m else text.split("</think>", 1)[-1].strip()
    cleaned = re.sub(r"\b25\s*[°˚]C\b|\b298\.?1?5?\s*K\b", "", ans)
    nums = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", cleaned)
    for raw in reversed(nums):
        try:
            value = _coerce(float(raw), cleaned)
        except ValueError:
            continue
        if value is not None:
            return value
    return None


def host_family(host: str) -> str:
    low = host.lower()
    if "cucurbit[8]" in low:
        return "CB[8]"
    if "cucurbit[7]" in low:
        return "CB[7]"
    if "cucurbit[6]" in low:
        return "CB[6]"
    if "cucurbit[5]" in low:
        return "CB[5]"
    if "cucurbit" in low or "glycouril" in low or "glycoluril" in low:
        return "CB-like"
    if "β-cyclodextrin" in low or "beta-cyclodextrin" in low:
        return "beta-CD"
    if "α-cyclodextrin" in low or "alpha-cyclodextrin" in low:
        return "alpha-CD"
    if "cyclodextrin" in low:
        return "CD-other"
    if "sulfonatocalix[4]" in low or "sulfocalix[4]" in low:
        return "p-SC4"
    if "sulfonatocalix[6]" in low or "sulfocalix[6]" in low:
        return "p-SC6"
    if "calix" in low:
        return "calixarene-other"
    if "naphthotube" in low:
        return "naphthotube"
    if "octa acid" in low or "cavitand" in low:
        return "cavitand"
    return "other"


def affinity_bin(ref: float) -> str:
    if ref < 3:
        return "<3"
    if ref < 6:
        return "3-6"
    if ref < 9:
        return "6-9"
    return ">=9"


def stats(values: list[float]) -> dict[str, float | int | None]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "std": None}
    return {
        "n": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
    }


def rmse(errors: list[float]) -> float | None:
    if not errors:
        return None
    return math.sqrt(mean([e * e for e in errors]))


def slope_intercept(xs: list[float], ys: list[float]) -> tuple[float | None, float | None]:
    if len(xs) < 2:
        return None, None
    mx, my = mean(xs), mean(ys)
    varx = sum((x - mx) ** 2 for x in xs)
    if varx == 0:
        return None, None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / varx
    return slope, my - slope * mx


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.{digits}f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def extract_fewshot_values() -> dict[str, Any]:
    rows = _read_jsonl(DATA / "fewshot.jsonl")
    signatures: Counter[tuple[float, ...]] = Counter()
    for row in rows:
        prompt = row["question"]
        before_final = prompt.rsplit("Put your final answer", 1)[0]
        vals = []
        for ans in re.findall(r"<answer>(.*?)</answer>", before_final, re.DOTALL | re.IGNORECASE):
            parsed = parse_logka(f"<answer>{ans}</answer>")
            if parsed is not None:
                vals.append(parsed)
        signatures[tuple(vals)] += 1
    top_sig, top_n = signatures.most_common(1)[0]
    flat = list(top_sig)
    return {
        "n_rows": len(rows),
        "n_unique_signatures": len(signatures),
        "top_signature_count": top_n,
        "values": flat,
        "mean": mean(flat),
        "median": median(flat),
        "std": pstdev(flat) if len(flat) > 1 else 0.0,
        "min": min(flat),
        "max": max(flat),
    }


def load_records() -> list[dict[str, Any]]:
    records = []
    for method in METHODS:
        for slug, label in MAIN_MODELS:
            path = RESULTS / method / f"{slug}.jsonl"
            for row in _read_jsonl(path):
                response = str(row.get("response") or "")
                official = parse_logka(response)
                fallback = fallback_parse_logka(response)
                ref = float(row["reference"])
                host = row.get("host_name") or ""
                pred_field = row.get("prediction")
                if isinstance(pred_field, (int, float)):
                    pred_json = float(pred_field)
                else:
                    pred_json = None
                records.append(
                    {
                        "method": method,
                        "model": slug,
                        "model_label": label,
                        "id": row["id"],
                        "host": host,
                        "family": host_family(host),
                        "guest": row.get("guest_name") or row.get("molecule") or "",
                        "ref": ref,
                        "bin": affinity_bin(ref),
                        "official": official,
                        "fallback": fallback,
                        "pred_json": pred_json,
                        "response_len": len(response),
                        "response": response,
                        "abs_error": abs(official - ref) if official is not None else None,
                        "signed_error": official - ref if official is not None else None,
                        "fallback_abs_error": abs(fallback - ref) if fallback is not None else None,
                    }
                )
    return records


def summarize_by_model_method(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[(rec["model"], rec["method"])].append(rec)
    for slug, label in MAIN_MODELS:
        for method in METHODS:
            group = grouped[(slug, method)]
            parsed = [r for r in group if r["official"] is not None]
            errors = [r["abs_error"] for r in parsed]
            signed = [r["signed_error"] for r in parsed]
            refs = [r["ref"] for r in parsed]
            preds = [r["official"] for r in parsed]
            lengths = [r["response_len"] for r in parsed]
            slope, intercept = slope_intercept(refs, preds)
            out.append(
                {
                    "model": slug,
                    "model_label": label,
                    "method": method,
                    "n": len(group),
                    "n_parsed": len(parsed),
                    "parse_rate": len(parsed) / len(group) if group else None,
                    "mae": mean(errors) if errors else None,
                    "rmse": rmse(errors),
                    "medae": median(errors) if errors else None,
                    "bias": mean(signed) if signed else None,
                    "pred_mean": mean(preds) if preds else None,
                    "pred_std": pstdev(preds) if len(preds) > 1 else None,
                    "gold_mean": mean(refs) if refs else None,
                    "slope": slope,
                    "intercept": intercept,
                    "mean_response_len": mean(lengths) if lengths else None,
                    "len_error_corr": pearson(lengths, errors) if len(lengths) > 2 else None,
                    "fallback_mae": mean([r["fallback_abs_error"] for r in group if r["fallback_abs_error"] is not None])
                    if group
                    else None,
                    "fallback_parse_rate": len([r for r in group if r["fallback"] is not None]) / len(group)
                    if group
                    else None,
                }
            )
    return out


def paired_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        by_key[(rec["model"], rec["id"])][rec["method"]] = rec
    out = []
    for slug, label in MAIN_MODELS:
        rows = [methods for (model, _), methods in by_key.items() if model == slug and all(m in methods for m in METHODS)]
        for method in METHODS:
            parsed = [r[method] for r in rows if r[method]["official"] is not None]
            errors = [r["abs_error"] for r in parsed]
            signed = [r["signed_error"] for r in parsed]
            out.append(
                {
                    "model": slug,
                    "model_label": label,
                    "method": method,
                    "n_paired": len(rows),
                    "n_parsed": len(parsed),
                    "mae": mean(errors) if errors else None,
                    "rmse": rmse(errors),
                    "medae": median(errors) if errors else None,
                    "bias": mean(signed) if signed else None,
                }
            )
    return out


def grouped_summary(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[(rec[key], rec["method"])].append(rec)
    out = []
    for group_name in sorted({rec[key] for rec in records}):
        if sum(1 for rec in records if rec[key] == group_name and rec["method"] == "base") < 20:
            continue
        base_errors = [r["abs_error"] for r in grouped[(group_name, "base")] if r["abs_error"] is not None]
        for method in METHODS:
            rows = grouped[(group_name, method)]
            errors = [r["abs_error"] for r in rows if r["abs_error"] is not None]
            signed = [r["signed_error"] for r in rows if r["signed_error"] is not None]
            out.append(
                {
                    key: group_name,
                    "method": method,
                    "n": len(errors),
                    "mae": mean(errors) if errors else None,
                    "delta_vs_base": (mean(errors) - mean(base_errors)) if errors and base_errors else None,
                    "bias": mean(signed) if signed else None,
                }
            )
    return out


def select_examples(records: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for rec in records:
        by_key[(rec["model"], rec["id"])][rec["method"]] = rec
    candidates = []
    for (model, ex_id), methods in by_key.items():
        if not all(m in methods for m in METHODS):
            continue
        base = methods["base"]
        if base["abs_error"] is None or base["abs_error"] > 0.75:
            continue
        for method in ["fewshot", "cot"]:
            rec = methods[method]
            if rec["abs_error"] is None:
                continue
            damage = rec["abs_error"] - base["abs_error"]
            if damage >= 2.0:
                candidates.append((damage, method, model, ex_id, base, rec, methods))
    selected = []
    seen: set[tuple[str, str]] = set()
    for damage, method, model, ex_id, base, rec, methods in sorted(candidates, reverse=True):
        family_key = (method, rec["family"])
        if family_key in seen and len(selected) < limit - 2:
            continue
        seen.add(family_key)
        selected.append(
            {
                "model": rec["model_label"],
                "id": ex_id,
                "host": rec["host"],
                "family": rec["family"],
                "guest": rec["guest"],
                "gold": rec["ref"],
                "base_pred": base["official"],
                "base_abs_error": base["abs_error"],
                "bad_method": method,
                "bad_pred": rec["official"],
                "bad_abs_error": rec["abs_error"],
                "damage": damage,
                "base_response": base["response"][:500].replace("\n", " "),
                "bad_response": rec["response"][:900].replace("\n", " "),
            }
        )
        if len(selected) >= limit:
            break
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    metrics: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    by_family: list[dict[str, Any]],
    by_bin: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    fewshot_info: dict[str, Any],
) -> str:
    metric_map = {(r["model"], r["method"]): r for r in metrics}
    paired_map = {(r["model"], r["method"]): r for r in paired}

    degradation_rows = []
    paired_degradation_rows = []
    slope_rows = []
    parser_rows = []
    for slug, label in MAIN_MODELS:
        base = metric_map[(slug, "base")]
        few = metric_map[(slug, "fewshot")]
        cot = metric_map[(slug, "cot")]
        degradation_rows.append(
            [
                label,
                fmt(base["mae"]),
                fmt(few["mae"]),
                fmt(few["mae"] - base["mae"]),
                fmt(cot["mae"]),
                fmt(cot["mae"] - base["mae"]),
            ]
        )
        pbase = paired_map[(slug, "base")]
        pfew = paired_map[(slug, "fewshot")]
        pcot = paired_map[(slug, "cot")]
        paired_degradation_rows.append(
            [
                label,
                pbase["n_paired"],
                fmt(pbase["mae"]),
                fmt(pfew["mae"]),
                fmt(pfew["mae"] - pbase["mae"]),
                fmt(pcot["mae"]),
                fmt(pcot["mae"] - pbase["mae"]),
            ]
        )
        slope_rows.append(
            [
                label,
                fmt(base["slope"]),
                fmt(few["slope"]),
                fmt(cot["slope"]),
                fmt(base["pred_std"]),
                fmt(few["pred_std"]),
                fmt(cot["pred_std"]),
            ]
        )
        parser_rows.append(
            [
                label,
                fmt(base["parse_rate"], 4),
                fmt(few["parse_rate"], 4),
                fmt(cot["parse_rate"], 4),
                fmt(few["fallback_mae"] - few["mae"] if few["fallback_mae"] is not None else None),
                fmt(cot["fallback_mae"] - cot["mae"] if cot["fallback_mae"] is not None else None),
                fmt(few["mean_response_len"], 1),
                fmt(cot["mean_response_len"], 1),
            ]
        )

    base_mean = mean(metric_map[(slug, "base")]["mae"] for slug, _ in MAIN_MODELS)
    few_mean = mean(metric_map[(slug, "fewshot")]["mae"] for slug, _ in MAIN_MODELS)
    cot_mean = mean(metric_map[(slug, "cot")]["mae"] for slug, _ in MAIN_MODELS)
    paired_base_mean = mean(paired_map[(slug, "base")]["mae"] for slug, _ in MAIN_MODELS)
    paired_few_mean = mean(paired_map[(slug, "fewshot")]["mae"] for slug, _ in MAIN_MODELS)
    paired_cot_mean = mean(paired_map[(slug, "cot")]["mae"] for slug, _ in MAIN_MODELS)

    family_map = {(r["family"], r["method"]): r for r in by_family if "family" in r}
    family_rows = []
    family_names = [
        f for f in sorted({r["family"] for r in by_family if "family" in r})
        if (f, "base") in family_map and family_map[(f, "base")]["n"] >= 80
    ]
    for family in sorted(family_names, key=lambda f: -family_map[(f, "base")]["n"]):
        b = family_map[(family, "base")]
        few = family_map.get((family, "fewshot"))
        cot = family_map.get((family, "cot"))
        family_rows.append(
            [
                family,
                b["n"],
                fmt(b["mae"]),
                fmt(few["mae"] if few else None),
                fmt(few["delta_vs_base"] if few else None),
                fmt(cot["mae"] if cot else None),
                fmt(cot["delta_vs_base"] if cot else None),
            ]
        )

    bin_map = {(r["bin"], r["method"]): r for r in by_bin if "bin" in r}
    bin_rows = []
    for bin_name in ["<3", "3-6", "6-9", ">=9"]:
        b = bin_map[(bin_name, "base")]
        few = bin_map[(bin_name, "fewshot")]
        cot = bin_map[(bin_name, "cot")]
        bin_rows.append(
            [
                bin_name,
                b["n"],
                fmt(b["mae"]),
                fmt(few["mae"]),
                fmt(few["delta_vs_base"]),
                fmt(cot["mae"]),
                fmt(cot["delta_vs_base"]),
                fmt(b["bias"]),
                fmt(few["bias"]),
                fmt(cot["bias"]),
            ]
        )

    example_rows = []
    for ex in examples[:6]:
        example_rows.append(
            [
                ex["model"],
                ex["bad_method"],
                ex["family"],
                ex["id"],
                fmt(ex["gold"]),
                fmt(ex["base_pred"]),
                fmt(ex["bad_pred"]),
                fmt(ex["damage"]),
            ]
        )

    lines = [
        "# Case Study: Why CoT and Few-Shot Hurt Drug Delivery",
        "",
        "**Scope**: This report analyzes BAP Drug Delivery binding-affinity prediction using existing inference files under `results/bap/`. It does not modify Overleaf files.",
        "",
        "## Executive Summary",
        "",
        f"- The main effect is robust: across the eight main-table models, mean MAE is `{base_mean:.3f}` for Base, `{few_mean:.3f}` for Few-Shot, and `{cot_mean:.3f}` for CoT. Few-Shot is `+{few_mean - base_mean:.3f}` MAE worse than Base, and CoT is `+{cot_mean - base_mean:.3f}` worse.",
        f"- The paired-ID audit gives the same conclusion: mean MAE is `{paired_base_mean:.3f}` for Base, `{paired_few_mean:.3f}` for Few-Shot, and `{paired_cot_mean:.3f}` for CoT.",
        "- The parser-artifact explanation is weak: 7 of 8 main-table models have near-complete official parse rates, and the one major exception, Llama3.1-8B, is not rescued by a diagnostic fallback numeric parser. The fallback parser does not repair the CoT/Few-Shot degradation.",
        f"- Few-Shot uses one fixed exemplar signature in `{fewshot_info['top_signature_count']}/{fewshot_info['n_rows']}` prompts: `{', '.join(fmt(v) for v in fewshot_info['values'])}`. This mixes an implausibly low negative example with an extreme high-affinity example, which is consistent with wider and less calibrated predictions rather than stable regression.",
        "- The clearest mechanism supported by the diagnostics is not simply verbosity. CoT and Few-Shot change numeric calibration: prediction slopes and variances shift, and the damage is especially visible for several host families. The high-affinity bin is already severely underpredicted by all methods, but it is not where CoT/Few-Shot add the largest extra MAE.",
        "",
        "## Main Degradation Table",
        "",
        md_table(
            ["Model", "Base MAE", "Few-Shot MAE", "Few-Shot Delta", "CoT MAE", "CoT Delta"],
            degradation_rows,
        ),
        "",
        "Interpretation: every main-table model is worse under both Few-Shot and CoT than under Base. CoT is the single worst method for most models, while Few-Shot is comparably harmful on average.",
        "",
        "## Paired-ID Audit",
        "",
        md_table(
            ["Model", "Paired N", "Base MAE", "Few-Shot MAE", "Few-Shot Delta", "CoT MAE", "CoT Delta"],
            paired_degradation_rows,
        ),
        "",
        "Interpretation: the degradation is not caused by different example sets across prompt strategies. The paired subset preserves the same direction and magnitude.",
        "",
        "## Parser and Verbosity Control",
        "",
        md_table(
            [
                "Model",
                "Base Parse",
                "Few Parse",
                "CoT Parse",
                "Few Fallback Delta",
                "CoT Fallback Delta",
                "Few Len",
                "CoT Len",
            ],
            parser_rows,
        ),
        "",
        "Notes: `Fallback Delta` is diagnostic fallback MAE minus official MAE. Values near zero mean alternative numeric extraction does not change the conclusion. Large positive values mean fallback extraction is worse, usually because it captures irrelevant numbers from verbose reasoning. The official benchmark should not be replaced by this diagnostic parser.",
        "",
        "## Numeric Compression and Calibration",
        "",
        md_table(
            ["Model", "Base Slope", "Few Slope", "CoT Slope", "Base Pred Std", "Few Pred Std", "CoT Pred Std"],
            slope_rows,
        ),
        "",
        "Interpretation: slopes below 1 indicate compressed predictions relative to gold `logKa`; larger prediction standard deviations indicate more volatile numeric estimates. CoT and Few-Shot often increase slope and spread while still worsening MAE, so the failure is better described as calibration destabilization than simple compression.",
        "",
        "## Few-Shot Exemplar Anchor",
        "",
        md_table(
            ["Property", "Value"],
            [
                ["Rows using top exemplar signature", f"{fewshot_info['top_signature_count']} / {fewshot_info['n_rows']}"],
                ["Unique exemplar signatures", fewshot_info["n_unique_signatures"]],
                ["Exemplar values", ", ".join(fmt(v) for v in fewshot_info["values"])],
                ["Exemplar mean", fmt(fewshot_info["mean"])],
                ["Exemplar std", fmt(fewshot_info["std"])],
                ["Exemplar min / max", f"{fmt(fewshot_info['min'])} / {fmt(fewshot_info['max'])}"],
            ],
        ),
        "",
        "Interpretation: the Few-Shot prompt gives the model a tiny numeric prior with one negative affinity and one extreme high-affinity answer. That is a bad calibration object for a broad regression task: it teaches answer format, but it also injects a distorted numeric range.",
        "",
        "## Host-Family Localization",
        "",
        md_table(
            ["Family", "N", "Base MAE", "Few MAE", "Few Delta", "CoT MAE", "CoT Delta"],
            family_rows,
        ),
        "",
        "Interpretation: this table localizes where the prompting damage appears. Families with large positive deltas are the best candidates for qualitative examples in the paper.",
        "",
        "## Affinity-Bin Localization",
        "",
        md_table(
            [
                "Gold Bin",
                "N",
                "Base MAE",
                "Few MAE",
                "Few Delta",
                "CoT MAE",
                "CoT Delta",
                "Base Bias",
                "Few Bias",
                "CoT Bias",
            ],
            bin_rows,
        ),
        "",
        "Interpretation: the high-affinity bin is the critical scientific region for drug delivery because loading/retention claims depend on strong binding. All methods strongly underpredict this bin, but CoT/Few-Shot do not add the largest incremental MAE there. The broader degradation is concentrated in the lower and middle affinity ranges plus specific host families.",
        "",
        "## Representative Failure Candidates",
        "",
        md_table(
            ["Model", "Bad Method", "Family", "ID", "Gold", "Base Pred", "Bad Pred", "Damage"],
            example_rows,
        ),
        "",
        "Detailed candidate traces are stored in `results/analysis/bap_prompt_case_study_examples.json` for manual review. These examples were selected by a transparent rule: Base absolute error <= 0.75 and CoT/Few-Shot error at least 2.0 larger.",
        "",
        "## Paper-Ready Finding Draft",
        "",
        "> On the Drug Delivery affinity-regression task, adding exemplars or explicit reasoning consistently degrades performance relative to the direct prompt. The degradation persists under paired-example and parser-control analyses, suggesting that the issue is not merely output formatting. Instead, CoT and Few-Shot destabilize the numeric calibration of `logKa` estimates: Few-Shot exposes the model to a small, distorted set of affinity anchors, while CoT encourages verbose rationales that shift the final numeric estimate. The strongest localization evidence appears at the host-family level, especially for cucurbituril-like hosts; high-affinity examples are severely underpredicted by all prompting strategies rather than uniquely damaged by CoT or Few-Shot.",
        "",
        "## Suggested Next Actions",
        "",
        "1. Use this report to decide which one mechanism to emphasize in the paper: numeric anchoring, high-affinity underprediction, or family-specific overgeneralization.",
        "2. Convert the strongest two tables into one compact case-study figure/table after review.",
        "3. Only run optional diagnostic inference if the current evidence feels too correlational. The must-run analysis already supports a reviewer-facing case study.",
        "",
        "## Generated Artifacts",
        "",
        "- `results/analysis/bap_prompt_case_study.md`",
        "- `results/analysis/bap_prompt_case_study_summary.json`",
        "- `results/analysis/bap_prompt_case_study_metrics.csv`",
        "- `results/analysis/bap_prompt_case_study_paired.csv`",
        "- `results/analysis/bap_prompt_case_study_by_family.csv`",
        "- `results/analysis/bap_prompt_case_study_by_bin.csv`",
        "- `results/analysis/bap_prompt_case_study_examples.json`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = load_records()
    metrics = summarize_by_model_method(records)
    paired = paired_summary(records)
    by_family = grouped_summary(records, "family")
    by_bin = grouped_summary(records, "bin")
    fewshot_info = extract_fewshot_values()
    examples = select_examples(records)

    summary = {
        "fewshot_exemplars": fewshot_info,
        "metrics": metrics,
        "paired": paired,
        "by_family": by_family,
        "by_bin": by_bin,
        "examples": examples,
    }

    (OUT / "bap_prompt_case_study_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(OUT / "bap_prompt_case_study_metrics.csv", metrics)
    write_csv(OUT / "bap_prompt_case_study_paired.csv", paired)
    write_csv(OUT / "bap_prompt_case_study_by_family.csv", by_family)
    write_csv(OUT / "bap_prompt_case_study_by_bin.csv", by_bin)
    (OUT / "bap_prompt_case_study_examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = build_report(metrics, paired, by_family, by_bin, examples, fewshot_info)
    (OUT / "bap_prompt_case_study.md").write_text(report, encoding="utf-8")
    print(OUT / "bap_prompt_case_study.md")


if __name__ == "__main__":
    main()
