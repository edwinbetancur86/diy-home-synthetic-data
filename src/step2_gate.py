"""
step2_gate.py — Step 2: Data Quality Gate.

Takes the raw JSONL from Step 1 and decides which items are allowed to advance to
human labeling (Step 3) and the LLM judge (Step 4). It does four cheap, deterministic
things — NO LLM calls, so it costs nothing and is fully reproducible:

  1. Structural check   — re-parse every line against the schema. Anything that no
                          longer validates is dropped (should be rare; Step 1 already
                          validated, but re-checking guards against hand-edited/corrupt files).
  2. Per-dimension pre-checks — fast keyword/overlap heuristics that estimate D2–D6
                          quality. Most are ADVISORY (recorded, non-blocking); only the
                          safety-scope check (D4) can hard-reject.
  3. Batch dedup        — normalize each question and drop near-duplicates so the set
                          isn't padded with the same item twice (Jaccard on tokens).
  4. Distribution check — compare the surviving category mix against the benchmark
                          (uniform 20% per category) and flag any under-represented one.

WHY are most per-dimension checks advisory instead of hard rejections?
  The binding quality judgment is the human (Step 3) + judge (Step 4). If this gate
  hard-failed items on noisy D2/D6 heuristics it would (a) risk throwing out good data
  on a false match and (b) destroy the very failure signal the improvement experiment
  (Step 6) is designed to measure. So the gate only hard-stops on things that are
  unambiguous or dangerous: broken structure, exact duplicates, and DIY steps that walk
  someone through genuinely pro-only work (gas lines, the main service panel, the meter).
  Everything else is surfaced as a flag so we can eyeball cheap quality signals before
  spending money on the judge — but the item still advances.

Run it:
    python -m src.step2_gate --run-label baseline
    python -m src.step2_gate --run-label baseline --live-benchmark   # verify vs. HF dataset
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from src.schemas import Category
from src.step1_generate import GeneratedRecord, _append_jsonl

# ---------------------------------------------------------------------------
# Paths. Input is Step 1's dataset; outputs are the gated dataset + a JSON report.
# ---------------------------------------------------------------------------
GENERATED_DIR = Path("data/generated")
REPORTS_DIR = Path("data/reports")

# The benchmark (dipenbhuva/home-diy-repair-qa) is a documented uniform 20%/category.
# We keep that as a constant so the gate runs offline and reproducibly; --live-benchmark
# can pull the real dataset to re-confirm the numbers haven't drifted.
BENCHMARK_SHARE = 1.0 / len(Category)  # 0.20
MIN_CATEGORY_SHARE = 0.20              # spec: every category must be >= 20% of the set

# Two questions are "the same" if their word sets overlap this much (Jaccard). 0.85 is
# strict enough to catch reworded duplicates without merging two genuinely different Qs.
DUPLICATE_JACCARD = 0.85

# ---------------------------------------------------------------------------
# Heuristic vocabularies for the cheap per-dimension pre-checks.
# These are intentionally SMALL and conservative — a rough smoke detector, not the judge.
# ---------------------------------------------------------------------------

# D4 (scope) HARD stop: DIY steps should never walk a homeowner through these. We use
# specific dangerous phrases (e.g. "main panel", not bare "breaker") so we don't false-flag
# the allowed action of resetting a tripped breaker.
SCOPE_HARD_KEYWORDS: tuple[str, ...] = (
    "gas line", "gas pipe", "gas valve", "gas supply", "gas fitting",
    "natural gas", "propane line", "main panel", "service panel",
    "main breaker panel", "service entrance", "electric meter", "electrical meter",
    "meter base", "sewer main", "load-bearing", "load bearing",
)
# If a step MENTIONS out-of-scope work only to redirect to a pro, it's fine — don't flag.
PRO_REDIRECT_TERMS: tuple[str, ...] = (
    "professional", "licensed", "electrician", "plumber", "technician",
    "call a", "hire a", "do not attempt", "don't attempt", "qualified", "certified",
)

# D2 (safety specificity) advisory: generic filler vs. a concrete hazard/precaution.
GENERIC_SAFETY: tuple[str, ...] = (
    "be careful", "use caution", "be cautious", "stay safe", "be safe",
    "safety first", "use common sense", "take precautions", "wear protective gear",
    "wear safety gear", "exercise caution",
)
SPECIFIC_SAFETY_SIGNALS: tuple[str, ...] = (
    "turn off", "shut off", "unplug", "disconnect", "shock", "voltage", "breaker",
    "water supply", "goggles", "gloves", "ventilat", "fumes", "hot", "sharp", "burn",
    "eye protection", "circuit", "power to",
)

# D3 (tool realism) advisory: clearly pro/specialty gear a typical homeowner won't own.
PRO_TOOLS: tuple[str, ...] = (
    "manifold gauge", "refrigerant", "vacuum pump", "recovery machine", "brazing torch",
    "oxy-acetylene", "megohmmeter", "oscilloscope", "hydro jetter", "drain camera",
    "thermal imaging", "core drill", "pipe threading machine", "sheet metal brake",
)

# D6 (tip usefulness) advisory: empty encouragement that teaches nothing.
GENERIC_TIPS: tuple[str, ...] = (
    "take your time", "be patient", "good luck", "you can do it", "don't rush",
    "have fun", "stay safe", "be careful", "read the instructions", "measure twice",
)

# Common words to ignore when comparing questions/answers for overlap.
_STOPWORDS: frozenset[str] = frozenset(
    "a an the is are my i to of in on and or for with it this that how do i my "
    "why what when can should does keep my me you your it's its from at be have has".split()
)


# ---------------------------------------------------------------------------
# Small text utilities used by dedup and the context-overlap check.
# ---------------------------------------------------------------------------
def _tokens(text: str) -> set[str]:
    """Lowercase content words (letters/digits only), minus stopwords. Set = order-free."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Overlap ratio of two token sets: |intersection| / |union|. 0..1."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# The cheap per-dimension pre-checks. Each returns True when the item looks like it
# FAILS that dimension. `advisory` flags are recorded but do not block.
# ---------------------------------------------------------------------------
def _scope_violation(steps: list[str]) -> bool:
    """D4 HARD: a step performs pro-only work (gas/main panel/meter) without redirecting."""
    for step in steps:
        low = step.lower()
        if any(k in low for k in SCOPE_HARD_KEYWORDS) and not any(
            r in low for r in PRO_REDIRECT_TERMS
        ):
            return True
    return False


def _weak_safety(safety_info: str) -> bool:
    """D2 advisory: generic filler with no concrete hazard, or suspiciously short."""
    low = safety_info.lower()
    too_short = len(safety_info.split()) < 6
    generic = any(p in low for p in GENERIC_SAFETY)
    specific = any(s in low for s in SPECIFIC_SAFETY_SIGNALS)
    return too_short or (generic and not specific)


def _unrealistic_tools(tools: list[str], steps: list[str]) -> bool:
    """D3 advisory: any tool/step references clearly professional or specialty equipment."""
    haystack = " ".join(tools + steps).lower()
    return any(t in haystack for t in PRO_TOOLS)


def _context_mismatch(equipment_problem: str, answer: str) -> bool:
    """D5 advisory: none of the problem's key words show up in the answer at all."""
    problem_terms = _tokens(equipment_problem)
    if not problem_terms:
        return False
    answer_terms = _tokens(answer)
    return len(problem_terms & answer_terms) == 0


def _weak_tips(tips: list[str], steps: list[str]) -> bool:
    """D6 advisory: every tip is generic encouragement, or a tip just restates a step."""
    if not tips:
        return True
    step_token_sets = [_tokens(s) for s in steps]
    generic_count = 0
    for tip in tips:
        low = tip.lower()
        if any(g in low for g in GENERIC_TIPS):
            generic_count += 1
            continue
        # A tip that heavily overlaps any step is a restatement, not a new insight.
        tip_tokens = _tokens(tip)
        if any(_jaccard(tip_tokens, s) > 0.6 for s in step_token_sets):
            generic_count += 1
    return generic_count == len(tips)


# ---------------------------------------------------------------------------
# Benchmark distribution: constant by default, live from HF only when asked.
# ---------------------------------------------------------------------------
def _benchmark_distribution(live: bool) -> dict[str, float]:
    """Category shares to compare against. Uniform constant unless --live-benchmark."""
    if not live:
        return {c.value: BENCHMARK_SHARE for c in Category}

    # Optional: confirm the real dataset is still ~uniform. Heavy + needs network, so
    # it's opt-in. Falls back to the constant if `datasets` isn't installed / offline.
    try:
        from datasets import load_dataset  # noqa: PLC0415 — optional, imported on demand

        ds = load_dataset("dipenbhuva/home-diy-repair-qa", split="train")
        counts = Counter(row["category"] for row in ds)
        total = sum(counts.values())
        return {cat: n / total for cat, n in counts.items()}
    except Exception as exc:  # noqa: BLE001 — never let a benchmark fetch crash the gate
        print(f"  (live benchmark unavailable: {type(exc).__name__}; using uniform 20%)")
        return {c.value: BENCHMARK_SHARE for c in Category}


# ---------------------------------------------------------------------------
# The gate itself.
# ---------------------------------------------------------------------------
def run_gate(run_label: str, live_benchmark: bool = False, min_share: float = MIN_CATEGORY_SHARE) -> dict:
    """Gate one run's dataset. Writes the gated JSONL + a JSON report and returns the report."""
    in_path = GENERATED_DIR / f"{run_label}.jsonl"
    if not in_path.exists():
        raise FileNotFoundError(
            f"No generated data at {in_path}. Run Step 1 first:  "
            f"python -m src.step1_generate --run-label {run_label}"
        )
    out_path = GENERATED_DIR / f"{run_label}_gated.jsonl"
    report_path = REPORTS_DIR / f"step2_gate_{run_label}.json"
    out_path.unlink(missing_ok=True)  # start clean so re-runs don't append to old output

    per_item: list[dict] = []
    kept_question_tokens: list[set[str]] = []  # token sets of items already accepted
    advisory_counter: Counter[str] = Counter()
    hard_counter: Counter[str] = Counter()
    passed_by_category: Counter[str] = Counter()
    passed_records: list[GeneratedRecord] = []
    input_count = 0

    for line in in_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        input_count += 1

        # (1) Structural re-validation.
        try:
            record = GeneratedRecord.model_validate_json(line)
        except ValidationError as exc:
            hard_counter["structural"] += 1
            per_item.append({"trace_id": None, "decision": "reject",
                             "hard_reasons": ["structural"], "advisory_flags": [],
                             "detail": str(exc.error_count()) + " schema error(s)"})
            continue

        qa = record.item
        hard_reasons: list[str] = []
        advisory_flags: list[str] = []

        # (2a) HARD per-dimension check: safety scope (D4).
        if _scope_violation(qa.steps):
            hard_reasons.append("scope_safety")

        # (3) HARD: near-duplicate question vs. anything already accepted.
        q_tokens = _tokens(qa.question)
        if any(_jaccard(q_tokens, kept) >= DUPLICATE_JACCARD for kept in kept_question_tokens):
            hard_reasons.append("duplicate")

        # (2b) ADVISORY per-dimension pre-checks (recorded, non-blocking).
        if _weak_safety(qa.safety_info):
            advisory_flags.append("safety_specificity")
        if _unrealistic_tools(qa.tools_required, qa.steps):
            advisory_flags.append("tool_realism")
        if _context_mismatch(qa.equipment_problem, qa.answer):
            advisory_flags.append("context_clarity")
        if _weak_tips(qa.tips, qa.steps):
            advisory_flags.append("tip_usefulness")

        for flag in advisory_flags:
            advisory_counter[flag] += 1

        decision = "reject" if hard_reasons else "pass"
        if decision == "pass":
            passed_records.append(record)
            kept_question_tokens.append(q_tokens)
            passed_by_category[record.category.value] += 1
            _append_jsonl(out_path, record.model_dump(mode="json"))
        else:
            for reason in hard_reasons:
                hard_counter[reason] += 1

        per_item.append({"trace_id": record.trace_id, "decision": decision,
                         "hard_reasons": hard_reasons, "advisory_flags": advisory_flags})

    # (4) Category distribution of the SURVIVING items vs. the benchmark.
    passed_count = len(passed_records)
    benchmark = _benchmark_distribution(live_benchmark)
    observed = {}
    below_threshold: list[str] = []
    for c in Category:
        n = passed_by_category.get(c.value, 0)
        share = (n / passed_count) if passed_count else 0.0
        observed[c.value] = {"count": n, "share": round(share, 4)}
        if share < min_share:
            below_threshold.append(c.value)

    report = {
        "run_label": run_label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_count": input_count,
        "passed_count": passed_count,
        "rejected_count": input_count - passed_count,
        "pass_rate": round(passed_count / input_count, 4) if input_count else 0.0,
        "hard_rejections": dict(hard_counter),
        "advisory_flag_counts": dict(advisory_counter),
        "category_distribution": {
            "observed": observed,
            "benchmark": {k: round(v, 4) for k, v in benchmark.items()},
            "min_share_threshold": min_share,
            "below_threshold": below_threshold,
            "distribution_ok": not below_threshold and passed_count >= 30,
        },
        "per_item": per_item,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_summary(report, out_path, report_path)
    return report


def _print_summary(report: dict, out_path: Path, report_path: Path) -> None:
    """Human-readable recap so you don't have to open the JSON to see how a run went."""
    print(f"\nStep 2 gate — {report['run_label']}")
    print(f"  input:    {report['input_count']}")
    print(f"  passed:   {report['passed_count']}  (pass rate {report['pass_rate']:.0%})")
    print(f"  rejected: {report['rejected_count']}  {report['hard_rejections'] or ''}")
    if report["advisory_flag_counts"]:
        print(f"  advisory flags (non-blocking): {report['advisory_flag_counts']}")
    dist = report["category_distribution"]
    if dist["below_threshold"]:
        print(f"  ⚠ under {dist['min_share_threshold']:.0%}: {dist['below_threshold']}")
    if report["passed_count"] < 30:
        print("  ⚠ fewer than 30 passing items — rates below this are noisy (spec rule #9).")
    print(f"  distribution_ok: {dist['distribution_ok']}")
    print(f"  gated data -> {out_path}")
    print(f"  report     -> {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2 — data quality gate.")
    parser.add_argument("--run-label", default="baseline", help="Which Step 1 run to gate.")
    parser.add_argument("--live-benchmark", action="store_true",
                        help="Verify category shares against the live HF dataset (needs network).")
    parser.add_argument("--min-share", type=float, default=MIN_CATEGORY_SHARE,
                        help="Minimum share each category must hold (default 0.20).")
    args = parser.parse_args()

    # Windows consoles default to cp1252, which can't encode the ⚠ glyphs in the
    # summary. Force UTF-8 (with a safe no-op fallback) so a run never dies on print.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # non-reconfigurable stream (e.g. some redirects)
        pass
    run_gate(args.run_label, live_benchmark=args.live_benchmark, min_share=args.min_share)


if __name__ == "__main__":
    main()
