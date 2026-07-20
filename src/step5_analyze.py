"""
step5_analyze.py — Step 5: Analysis & Visualization.

Turns the label files produced by Steps 3 (human) and 4 (judge) into (a) a single reproducible
metrics report at `data/reports/step5_analysis.json` and (b) the three charts that tell the
project's story in `visualizations/*.png`. It computes nothing new about the models — it
AGGREGATES the labels we already have — so it makes no API calls and is safe to re-run anytime.

Three questions it answers, each backed by one chart:

  1. Can we TRUST the judge?  → human-vs-judge agreement per dimension (Phase A calibration).
     Chart: agreement_by_dimension.png (bars + the 80% threshold line).

  2. Did the correction WORK?  → before/after overall + per-dimension failure rate.
     Chart: failure_before_after.png (paired bars; the climax metric in the title).

  3. WHERE does quality break down?  → category×dimension failure heatmap for the weak run.
     Chart: weak_segment_heatmap.png (shows the D3/tool-realism concentration).

"before" = the deliberately-weakened run (run-label `weak`, judged); "after"/"corrected" = the
clean run (run-label `baseline`, judged). Agreement is measured on the human/judge overlap of
the clean run (the only items with BOTH label types).

Run it:
    python -m src.step5_analyze                                   # defaults: weak vs baseline
    python -m src.step5_analyze --before weak --after baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.schemas import DIMENSION_KEYS, DIMENSION_LABELS
from src.step1_generate import GeneratedRecord

LABELS_DIR = Path("data/labels")
GENERATED_DIR = Path("data/generated")
REPORTS_DIR = Path("data/reports")
VIZ_DIR = Path("visualizations")

# Short D1–D6 tick labels for chart axes (the full DIMENSION_LABELS are too long on an axis).
SHORT_LABELS = {key: DIMENSION_LABELS[key].split(" ", 1)[0] for key in DIMENSION_KEYS}  # "D1".."D6"

# Spec thresholds we draw as reference lines.
AGREEMENT_THRESHOLD = 0.80
IMPROVEMENT_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Loading.
# ---------------------------------------------------------------------------
def _load_labels(kind: str, run_label: str) -> list[dict]:
    """Load a label file's `labels` list. kind is 'human' or 'judge'."""
    path = LABELS_DIR / f"{kind}_{run_label}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the {kind} labeling step for '{run_label}' first.")
    return json.loads(path.read_text(encoding="utf-8"))["labels"]


def _load_categories(run_label: str) -> dict[str, str]:
    """Map trace_id -> category by reading the gated JSONL (labels don't carry the category)."""
    path = GENERATED_DIR / f"{run_label}_gated.jsonl"
    cats: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = GeneratedRecord.model_validate_json(line)
            cats[rec.trace_id] = rec.category.value
    return cats


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------
def compute_agreement(human: list[dict], judge: list[dict]) -> dict:
    """Per-dimension + overall human/judge agreement on the items BOTH labeled."""
    hj = {r["trace_id"]: r for r in human}
    jj = {r["trace_id"]: r for r in judge}
    overlap = sorted(set(hj) & set(jj))
    n = len(overlap)
    per_dim = {}
    for dim in DIMENSION_KEYS:
        agree = sum(1 for t in overlap if hj[t][dim] == jj[t][dim])
        per_dim[dim] = {"agree": agree, "n": n, "rate": (agree / n if n else 0.0)}
    overall_agree = sum(1 for t in overlap if hj[t]["overall_pass"] == jj[t]["overall_pass"])
    return {
        "n": n,
        "per_dimension": per_dim,
        "overall_pass": {"agree": overall_agree, "n": n, "rate": (overall_agree / n if n else 0.0)},
        "meets_threshold": all(d["rate"] >= AGREEMENT_THRESHOLD for d in per_dim.values()),
    }


def compute_quality(judge: list[dict], categories: dict[str, str]) -> dict:
    """Overall + per-dimension + per-segment (category×dim) FAILURE rates for one judged run."""
    n = len(judge)
    overall_fail = sum(1 for r in judge if not r["overall_pass"])
    per_dim = {
        dim: {
            "fail": sum(1 for r in judge if r[dim] == 0),
            "n": n,
            "fail_rate": (sum(1 for r in judge if r[dim] == 0) / n if n else 0.0),
        }
        for dim in DIMENSION_KEYS
    }
    # Segment table: fail COUNT per (category, dimension). A DataFrame makes the heatmap trivial.
    rows = []
    for r in judge:
        cat = categories.get(r["trace_id"], "unknown")
        for dim in DIMENSION_KEYS:
            rows.append({"category": cat, "dimension": SHORT_LABELS[dim], "fail": 1 - r[dim]})
    seg = (
        pd.DataFrame(rows)
        .groupby(["category", "dimension"])["fail"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=[SHORT_LABELS[d] for d in DIMENSION_KEYS])
    )
    return {
        "n": n,
        "overall_fail": overall_fail,
        "overall_fail_rate": (overall_fail / n if n else 0.0),
        "per_dimension": per_dim,
        "_segment_frame": seg,  # kept out of JSON (leading underscore stripped on save)
    }


def compute_improvement(before: dict, after: dict) -> dict:
    """The climax metric: (before_fail_rate − after_fail_rate) / before_fail_rate."""
    b, a = before["overall_fail_rate"], after["overall_fail_rate"]
    improvement = (b - a) / b if b > 0 else 0.0
    return {
        "before_fail_rate": b,
        "after_fail_rate": a,
        "before_n": before["n"],
        "after_n": after["n"],
        "improvement": improvement,
        "threshold": IMPROVEMENT_THRESHOLD,
        "passes": improvement >= IMPROVEMENT_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Charts.
# ---------------------------------------------------------------------------
def chart_agreement(agreement: dict, out: Path) -> None:
    dims = [SHORT_LABELS[d] for d in DIMENSION_KEYS] + ["Overall"]
    rates = [agreement["per_dimension"][d]["rate"] for d in DIMENSION_KEYS]
    rates.append(agreement["overall_pass"]["rate"])
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(dims, rates, color=sns.color_palette("crest", len(dims)))
    ax.axhline(AGREEMENT_THRESHOLD, ls="--", color="crimson",
               label=f"threshold {AGREEMENT_THRESHOLD:.0%}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Human ↔ Judge agreement")
    ax.set_title(f"Step 6 Phase A — judge calibration (n={agreement['n']} human/judge overlap)")
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.02, f"{r:.0%}", ha="center", fontsize=9)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def chart_before_after(before: dict, after: dict, improvement: dict, out: Path) -> None:
    dims = [SHORT_LABELS[d] for d in DIMENSION_KEYS] + ["Overall"]
    before_r = [before["per_dimension"][d]["fail_rate"] for d in DIMENSION_KEYS] + [before["overall_fail_rate"]]
    after_r = [after["per_dimension"][d]["fail_rate"] for d in DIMENSION_KEYS] + [after["overall_fail_rate"]]
    x = np.arange(len(dims))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w / 2, before_r, w, label=f"before / weak (n={before['n']})", color="#c44e52")
    ax.bar(x + w / 2, after_r, w, label=f"after / corrected (n={after['n']})", color="#55a868")
    ax.set_xticks(x, dims)
    ax.set_ylabel("Failure rate (LLM judge)")
    ax.set_title(
        f"Before vs After — overall failure {improvement['before_fail_rate']:.0%} → "
        f"{improvement['after_fail_rate']:.0%}  "
        f"({improvement['improvement']:.0%} reduction, "
        f"{'PASS' if improvement['passes'] else 'FAIL'} ≥ {IMPROVEMENT_THRESHOLD:.0%})"
    )
    for i, (bv, av) in enumerate(zip(before_r, after_r)):
        if bv > 0:
            ax.text(i - w / 2, bv + 0.01, f"{bv:.0%}", ha="center", fontsize=8)
        if av > 0:
            ax.text(i + w / 2, av + 0.01, f"{av:.0%}", ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def chart_segment_heatmap(before_quality: dict, out: Path) -> None:
    seg = before_quality["_segment_frame"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(seg, annot=True, fmt="d", cmap="Reds", linewidths=0.5,
                cbar_kws={"label": "fail count"}, ax=ax)
    ax.set_title(f"Weak run — where quality breaks down (category × dimension, n={before_quality['n']})")
    ax.set_xlabel("dimension")
    ax.set_ylabel("category")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
def _strip_private(obj):
    """Recursively drop keys starting with '_' (e.g. the DataFrame) so the report is JSON-safe."""
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items() if not k.startswith("_")}
    return obj


def run_analysis(before_label: str, after_label: str) -> dict:
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Agreement uses the human labels + judge labels of the AFTER run (the clean one Edwin labeled).
    human = _load_labels("human", after_label)
    judge_after = _load_labels("judge", after_label)
    judge_before = _load_labels("judge", before_label)
    cats_after = _load_categories(after_label)
    cats_before = _load_categories(before_label)

    agreement = compute_agreement(human, judge_after)
    quality_before = compute_quality(judge_before, cats_before)
    quality_after = compute_quality(judge_after, cats_after)
    improvement = compute_improvement(quality_before, quality_after)

    # Charts.
    chart_agreement(agreement, VIZ_DIR / "agreement_by_dimension.png")
    chart_before_after(quality_before, quality_after, improvement, VIZ_DIR / "failure_before_after.png")
    chart_segment_heatmap(quality_before, VIZ_DIR / "weak_segment_heatmap.png")

    report = {
        "before_run": before_label,
        "after_run": after_label,
        "agreement": agreement,
        "quality_before": _strip_private(quality_before),
        "quality_after": _strip_private(quality_after),
        "improvement": improvement,
    }
    (REPORTS_DIR / "step5_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def _print_summary(report: dict) -> None:
    ag = report["agreement"]
    imp = report["improvement"]
    print("\nStep 5 — Analysis & Visualization")
    print(f"  agreement (n={ag['n']}): "
          + ", ".join(f"{SHORT_LABELS[d]} {ag['per_dimension'][d]['rate']:.0%}" for d in DIMENSION_KEYS)
          + f", overall {ag['overall_pass']['rate']:.0%}"
          + f"   [{'meets ≥80%' if ag['meets_threshold'] else 'BELOW 80%'}]")
    print(f"  failure rate: before {imp['before_fail_rate']:.0%} (n={imp['before_n']}) "
          f"→ after {imp['after_fail_rate']:.0%} (n={imp['after_n']})")
    print(f"  improvement:  {imp['improvement']:.0%}  "
          f"[{'PASS' if imp['passes'] else 'FAIL'} ≥ {IMPROVEMENT_THRESHOLD:.0%}]")
    print(f"  report -> {REPORTS_DIR / 'step5_analysis.json'}")
    print(f"  charts -> {VIZ_DIR}/ (agreement_by_dimension, failure_before_after, weak_segment_heatmap).png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5 — analysis & visualization.")
    parser.add_argument("--before", default="weak", help="Run-label for the 'before' (weak) run.")
    parser.add_argument("--after", default="baseline", help="Run-label for the 'after' (corrected) run.")
    args = parser.parse_args()
    report = run_analysis(args.before, args.after)
    _print_summary(report)


if __name__ == "__main__":
    main()
