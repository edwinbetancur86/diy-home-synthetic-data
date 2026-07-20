"""
step6_iterate.py — Step 6: Iteration verdict (the deliverable gate).

This is the pipeline's authoritative PASS/FAIL judge for the project's climax. Step 5 computes
and *visualizes* the metrics; Step 6 applies the spec's THRESHOLDS to them and returns a single
verdict — did the eval-driven correction meet the bar? It makes no API calls and reuses Step 5's
metric functions as the single source of truth for the math (no duplicated calculations).

It checks the three gates the spec defines:

  1. Phase A — judge calibration.   Human↔judge agreement ≥ 80% on EVERY dimension
     (CLAUDE.md §6). If this fails, the judge's numbers can't be trusted and Phase B is moot.

  2. Corrected-run quality.         The "after" run's judge PASS rates clear the per-dimension
     quality bars in CLAUDE.md §5 (D1 ≥85%, D2 ≥90%, D3 ≥95%, D4 ≥95%, D5 ≥90%, D6 ≥85%,
     overall ≥80%). This asks "is the corrected data actually good?", not just "is it better?".

  3. Phase B — improvement.         (before_fail − after_fail) / before_fail ≥ 0.80
     (CLAUDE.md §1). The headline claim.

`deliverable_met` is the AND of all three. The verdict is written to
`data/reports/step6_verdict.json` and printed as a table.

Run it:
    python -m src.step6_iterate                                  # weak (before) vs baseline (after)
    python -m src.step6_iterate --before weak --after baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.schemas import DIMENSION_KEYS, DIMENSION_LABELS
from src.step5_analyze import (
    compute_agreement,
    compute_improvement,
    compute_quality,
    load_categories,
    load_labels,
)

REPORTS_DIR = Path("data/reports")

# --- Thresholds, straight from CLAUDE.md (the spec) -------------------------
AGREEMENT_THRESHOLD = 0.80        # §6  — human/judge agreement, per dimension
IMPROVEMENT_THRESHOLD = 0.80      # §1  — failure-rate reduction (the climax)
OVERALL_QUALITY_THRESHOLD = 0.80  # §5  — corrected run's overall judge pass rate
# §5 — per-dimension judge PASS-rate floors for the corrected run.
QUALITY_THRESHOLDS: dict[str, float] = {
    "answer_completeness": 0.85,   # D1
    "safety_specificity": 0.90,    # D2
    "tool_realism": 0.95,          # D3
    "scope_appropriateness": 0.95, # D4
    "context_clarity": 0.90,       # D5
    "tip_usefulness": 0.85,        # D6
}

_SHORT = {k: DIMENSION_LABELS[k].split(" ", 1)[0] for k in DIMENSION_KEYS}  # "D1".."D6"


# ---------------------------------------------------------------------------
# The three verdicts.
# ---------------------------------------------------------------------------
def phase_a_verdict(agreement: dict) -> dict:
    """Phase A: human↔judge agreement ≥ threshold on every dimension."""
    per_dim = {
        dim: {
            "agreement": agreement["per_dimension"][dim]["rate"],
            "threshold": AGREEMENT_THRESHOLD,
            "meets": agreement["per_dimension"][dim]["rate"] >= AGREEMENT_THRESHOLD,
        }
        for dim in DIMENSION_KEYS
    }
    return {
        "n": agreement["n"],
        "per_dimension": per_dim,
        "overall_agreement": agreement["overall_pass"]["rate"],
        "meets": all(d["meets"] for d in per_dim.values()),
    }


def quality_verdict(after_quality: dict) -> dict:
    """Corrected run's judge PASS rate clears the per-dimension + overall quality bars."""
    per_dim = {}
    for dim in DIMENSION_KEYS:
        pass_rate = 1.0 - after_quality["per_dimension"][dim]["fail_rate"]
        per_dim[dim] = {
            "pass_rate": pass_rate,
            "threshold": QUALITY_THRESHOLDS[dim],
            "meets": pass_rate >= QUALITY_THRESHOLDS[dim],
        }
    overall_pass_rate = 1.0 - after_quality["overall_fail_rate"]
    overall_meets = overall_pass_rate >= OVERALL_QUALITY_THRESHOLD
    return {
        "n": after_quality["n"],
        "per_dimension": per_dim,
        "overall_pass_rate": overall_pass_rate,
        "overall_threshold": OVERALL_QUALITY_THRESHOLD,
        "meets": overall_meets and all(d["meets"] for d in per_dim.values()),
    }


def phase_b_verdict(improvement: dict) -> dict:
    """Phase B: the failure-rate reduction clears the improvement threshold."""
    return {
        "before_fail_rate": improvement["before_fail_rate"],
        "after_fail_rate": improvement["after_fail_rate"],
        "before_n": improvement["before_n"],
        "after_n": improvement["after_n"],
        "improvement": improvement["improvement"],
        "threshold": IMPROVEMENT_THRESHOLD,
        "meets": improvement["improvement"] >= IMPROVEMENT_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
def run_verdict(before_label: str, after_label: str) -> dict:
    # Reuse Step 5's loaders + metric functions — one source of truth for the math.
    human = load_labels("human", after_label)
    judge_after = load_labels("judge", after_label)
    judge_before = load_labels("judge", before_label)
    quality_after = compute_quality(judge_after, load_categories(after_label))
    quality_before = compute_quality(judge_before, load_categories(before_label))

    phase_a = phase_a_verdict(compute_agreement(human, judge_after))
    quality = quality_verdict(quality_after)
    phase_b = phase_b_verdict(compute_improvement(quality_before, quality_after))
    deliverable_met = phase_a["meets"] and quality["meets"] and phase_b["meets"]

    verdict = {
        "before_run": before_label,
        "after_run": after_label,
        "phase_a_calibration": phase_a,
        "corrected_quality": quality,
        "phase_b_improvement": phase_b,
        "deliverable_met": deliverable_met,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "step6_verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return verdict


def _mark(ok: bool) -> str:
    return "PASS ✅" if ok else "FAIL ❌"


def _print_verdict(v: dict) -> None:
    pa, q, pb = v["phase_a_calibration"], v["corrected_quality"], v["phase_b_improvement"]
    print("\nStep 6 — Iteration verdict")
    print(f"  before='{v['before_run']}'  after='{v['after_run']}'")

    print(f"\n  Phase A — judge calibration (agreement ≥ {AGREEMENT_THRESHOLD:.0%}, n={pa['n']}):  {_mark(pa['meets'])}")
    for dim in DIMENSION_KEYS:
        d = pa["per_dimension"][dim]
        print(f"    {_SHORT[dim]} {dim:24} {d['agreement']:4.0%}  (≥{d['threshold']:.0%})  {_mark(d['meets'])}")

    print(f"\n  Corrected-run quality (judge pass rates, n={q['n']}):  {_mark(q['meets'])}")
    for dim in DIMENSION_KEYS:
        d = q["per_dimension"][dim]
        print(f"    {_SHORT[dim]} {dim:24} {d['pass_rate']:4.0%}  (≥{d['threshold']:.0%})  {_mark(d['meets'])}")
    print(f"    {'overall':27} {q['overall_pass_rate']:4.0%}  (≥{q['overall_threshold']:.0%})")

    print(f"\n  Phase B — improvement:  {_mark(pb['meets'])}")
    print(f"    before {pb['before_fail_rate']:.0%} (n={pb['before_n']}) → after {pb['after_fail_rate']:.0%} "
          f"(n={pb['after_n']})   reduction {pb['improvement']:.0%}  (≥{pb['threshold']:.0%})")

    print(f"\n  ══ DELIVERABLE: {_mark(v['deliverable_met'])} ══")
    print(f"  verdict -> {REPORTS_DIR / 'step6_verdict.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6 — iteration verdict (deliverable gate).")
    parser.add_argument("--before", default="weak", help="Run-label for the 'before' (weak) run.")
    parser.add_argument("--after", default="baseline", help="Run-label for the 'after' (corrected) run.")
    args = parser.parse_args()
    _print_verdict(run_verdict(args.before, args.after))


if __name__ == "__main__":
    main()
