"""
step3_label.py — Step 3: Human Labeling (CLI).

Walks a human reviewer through the GATED items (Step 2 output) and records a binary
pass/fail on all 6 quality dimensions for each one. The result is the ground-truth
`data/labels/human_<run_label>.json` that Step 6 uses to calibrate the LLM judge — so
this is the human half of the human-vs-judge agreement measurement.

This module makes NO LLM calls. It is pure I/O + the reviewer's judgment. Three design
choices matter and are worth defending:

  1. SAME RUBRIC AS THE JUDGE. The "fails when…" text shown for each dimension is
     `DIMENSION_GUIDANCE` from schemas.py — the exact rubric the Step 4 judge prompt is
     built from. If the human and the judge scored against different definitions, their
     agreement number would be meaningless.

  2. RESUMABLE + CRASH-SAFE. Labeling 20+ items is tedious and easily interrupted. We
     save the label file after EVERY item and, on restart, skip items already labeled.
     You can quit ('q') at any prompt and pick up exactly where you left off.

  3. COMPUTED overall_pass. The reviewer never sets "overall" — LabelRecord derives it
     from the 6 scores, so it can't contradict them.

Run it (interactive — run it yourself in a real terminal):
    python -m src.step3_label --run-label baseline            # label all gated items
    python -m src.step3_label --run-label baseline --limit 25 # stop after 25
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from src.schemas import (
    DIMENSION_GUIDANCE,
    DIMENSION_KEYS,
    DIMENSION_LABELS,
    LabelRecord,
)
from src.step1_generate import GeneratedRecord

GENERATED_DIR = Path("data/generated")
LABELS_DIR = Path("data/labels")

# Spec: a human must label at least this many items for the ground truth to be usable.
MIN_ITEMS = 20
_WRAP = 96  # console wrap width for long text fields


class _Quit(Exception):
    """Raised when the reviewer types 'q' — unwind cleanly and save what we have."""


# ---------------------------------------------------------------------------
# Loading: gated dataset in, existing labels (for resume) in.
# ---------------------------------------------------------------------------
def _load_gated(run_label: str) -> list[GeneratedRecord]:
    """Read the Step 2 gated JSONL. Loud error if it's missing — don't label ungated data."""
    path = GENERATED_DIR / f"{run_label}_gated.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"No gated data at {path}. Run Step 2 first:  "
            f"python -m src.step2_gate --run-label {run_label}"
        )
    records: list[GeneratedRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(GeneratedRecord.model_validate_json(line))
    return records


def _labels_path(run_label: str) -> Path:
    return LABELS_DIR / f"human_{run_label}.json"


def _load_existing(run_label: str) -> dict:
    """Load prior labels for resume, or start a fresh container. Keyed by trace_id."""
    path = _labels_path(run_label)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        # Normalize the labels list into a dict for O(1) 'already done?' checks.
        data["_by_trace"] = {rec["trace_id"]: rec for rec in data.get("labels", [])}
        return data
    return {
        "run_label": run_label,
        "labeler": "human",
        "created_at": _now_iso(),
        "updated_at": None,
        "labels": [],
        "_by_trace": {},
    }


def _save(run_label: str, container: dict) -> None:
    """Persist labels. Called after every item so a crash never loses more than nothing."""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    container["updated_at"] = _now_iso()
    out = {k: v for k, v in container.items() if not k.startswith("_")}
    out["count"] = len(out["labels"])
    _labels_path(run_label).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Rendering one item and collecting the 6 scores.
# ---------------------------------------------------------------------------
def _wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(text, width=_WRAP, initial_indent=indent, subsequent_indent=indent)


def _show_item(record: GeneratedRecord, position: int, total: int, done: int) -> None:
    qa = record.item
    print("\n" + "=" * _WRAP)
    print(f" Item {position}/{total}   (labeled so far: {done})   trace_id={record.trace_id}")
    print(f" Category: {record.category.value}")
    print("=" * _WRAP)
    print(f"\nQUESTION:\n{_wrap(qa.question)}")
    print(f"\nEQUIPMENT PROBLEM:\n{_wrap(qa.equipment_problem)}")
    print(f"\nANSWER:\n{_wrap(qa.answer)}")
    print(f"\nTOOLS REQUIRED ({len(qa.tools_required)}):")
    for t in qa.tools_required:
        print(_wrap(f"- {t}", indent="    "))
    print(f"\nSTEPS ({len(qa.steps)}):")
    for i, s in enumerate(qa.steps, 1):
        print(_wrap(f"{i}. {s}", indent="    "))
    print("\nSAFETY INFO:")
    print(_wrap(qa.safety_info))
    print(f"\nTIPS ({len(qa.tips)}):")
    for tip in qa.tips:
        print(_wrap(f"- {tip}", indent="    "))
    print("-" * _WRAP)


def _ask_binary(dim_key: str) -> int:
    """Prompt for one dimension. Returns 1 (pass) or 0 (fail); 'q' aborts the session."""
    label = DIMENSION_LABELS[dim_key]
    hint = DIMENSION_GUIDANCE[dim_key]
    while True:
        print(f"\n  {label}")
        print(f"    {hint}")
        raw = input("    pass/fail?  [p]=pass  [f]=fail  [q]=save & quit : ").strip().lower()
        if raw in ("p", "pass", "1", "y", "yes"):
            return 1
        if raw in ("f", "fail", "0", "n", "no"):
            return 0
        if raw in ("q", "quit"):
            raise _Quit
        print("    ↳ didn't catch that — type p, f, or q.")


def _label_one(record: GeneratedRecord) -> LabelRecord | None:
    """Collect all 6 scores for one item. Returns None if the reviewer skips it."""
    choice = input(
        "\nLabel this item?  [Enter]=yes  [s]=skip  [q]=save & quit : "
    ).strip().lower()
    if choice in ("q", "quit"):
        raise _Quit
    if choice in ("s", "skip"):
        return None

    scores = {dim: _ask_binary(dim) for dim in DIMENSION_KEYS}
    return LabelRecord(trace_id=record.trace_id, labeler="human", **scores)


# ---------------------------------------------------------------------------
# The session loop.
# ---------------------------------------------------------------------------
def run_labeling(run_label: str, limit: int | None = None) -> None:
    gated = _load_gated(run_label)
    container = _load_existing(run_label)
    done_ids = set(container["_by_trace"])

    todo = [r for r in gated if r.trace_id not in done_ids]
    if limit is not None:
        todo = todo[:limit]

    total_available = len(gated)
    print(f"\nStep 3 — human labeling: {run_label}")
    print(f"  gated items:      {total_available}")
    print(f"  already labeled:  {len(done_ids)}")
    print(f"  to label now:     {len(todo)}"
          + (f"  (--limit {limit})" if limit is not None else ""))
    if not todo:
        print("\nNothing left to label. ✔")
        _summary(container)
        return
    print("  Tip: you can quit any time with 'q' — progress is saved after every item.")

    newly = 0
    try:
        for i, record in enumerate(todo, 1):
            _show_item(record, i, len(todo), len(container["labels"]))
            label = _label_one(record)
            if label is None:
                print("  (skipped)")
                continue
            container["labels"].append(label.model_dump(mode="json"))
            container["_by_trace"][record.trace_id] = container["labels"][-1]
            _save(run_label, container)  # crash-safe: persist immediately
            newly += 1
            verdict = "PASS" if label.overall_pass else "FAIL"
            print(f"  ✔ saved — overall: {verdict}")
    except _Quit:
        print("\n[quit] saving and exiting…")
    except (KeyboardInterrupt, EOFError):
        print("\n[interrupted] saving and exiting…")

    _save(run_label, container)
    print(f"\nSession done. Labeled {newly} new item(s) this session.")
    _summary(container)


def _summary(container: dict) -> None:
    """Recap totals + per-dimension fail counts, and whether we've hit the ≥20 minimum."""
    labels = container["labels"]
    n = len(labels)
    print(f"\n  total labeled: {n}")
    if n == 0:
        return
    overall_pass = sum(1 for r in labels if r["overall_pass"])
    print(f"  overall pass:  {overall_pass}/{n}  ({overall_pass / n:.0%})")
    print("  per-dimension fail counts:")
    for dim in DIMENSION_KEYS:
        fails = sum(1 for r in labels if r[dim] == 0)
        print(f"    {DIMENSION_LABELS[dim]:26} {fails} fail(s)")
    if n < MIN_ITEMS:
        print(f"  ⚠ need ≥{MIN_ITEMS} labeled for a usable ground truth (have {n}).")
    else:
        print(f"  ✔ meets the ≥{MIN_ITEMS}-item minimum.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3 — human labeling CLI (6 dimensions).")
    parser.add_argument("--run-label", default="baseline", help="Which gated run to label.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Label at most this many NEW items this session.")
    args = parser.parse_args()

    # Generated safety_info contains emoji (⚠️); force UTF-8 so a Windows console
    # (cp1252) doesn't crash mid-item while printing.
    for stream in (sys.stdout, sys.stdin):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    run_labeling(args.run_label, limit=args.limit)


if __name__ == "__main__":
    main()
