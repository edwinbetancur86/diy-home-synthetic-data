"""
step1_generate.py — Step 1: Generation.

Uses the Instructor-patched Claude client to produce schema-valid QAItems, one per
call, cycling through all 5 categories so the batch is naturally ~20% per category
(which sets up the Step 2 distribution gate to pass).

Design highlights (each maps to a spec rule):
  - create_with_completion(): also returns the raw Anthropic response so we can log it
    (Step 1 audit requirement).
  - tenacity retry on transient API/rate-limit errors; Instructor retries on schema
    mismatch → a single bad output never crashes the run.
  - a small delay between calls → basic rate limiting.
  - clean records go to data/generated/*.jsonl (the dataset); verbose raw goes to
    logs/*.jsonl (the audit trail).

Run it:
    python -m src.step1_generate --count 10 --run-label baseline
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import cycle
from pathlib import Path

import anthropic
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from prompts.generator import DEFAULT_GENERATOR_VERSION, build_generation_prompt, subtopic_for
from src.config import get_client, get_settings
from src.schemas import Category, QAItem

# Project directories (relative to the repo root).
DATA_DIR = Path("data/generated")
LOG_DIR = Path("logs")
# Ceiling for one Q&A item. 1500 was too tight once sub-topic seeding made answers
# fuller — detailed items (long answer + ≥3 steps + safety + tips) truncated mid-JSON
# and failed to parse (IncompleteOutputException). 4000 gives comfortable headroom.
MAX_TOKENS = 4000


class GeneratedRecord(BaseModel):
    """A generated QAItem wrapped with the pipeline metadata later steps need."""

    trace_id: str
    category: Category
    prompt_version: str
    model: str
    temperature: float
    generated_at: str  # ISO-8601 UTC timestamp
    item: QAItem


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Retry only on transient, retail-able API errors. Schema/validation retries are handled
# separately by Instructor's own max_retries, so we don't double-handle those here.
@retry(
    retry=retry_if_exception_type(
        (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError)
    ),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _call_claude(system_prompt: str, user_prompt: str) -> tuple[QAItem, object]:
    """One generation call. Returns the validated QAItem plus the raw completion."""
    settings = get_settings()
    client = get_client()
    item, completion = client.messages.create_with_completion(
        model=settings.generator_model,
        max_tokens=MAX_TOKENS,
        temperature=settings.generator_temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        response_model=QAItem,
        max_retries=2,  # Instructor re-asks the model if the output fails validation
    )
    return item, completion


def generate_item(
    category: Category, trace_id: str, version: str, run_label: str, focus: str | None = None
) -> GeneratedRecord | None:
    """Generate one record for a category. Returns None (and logs) on unrecoverable failure.

    `focus` seeds a specific sub-problem so batches stay diverse (see prompts.generator).
    """
    settings = get_settings()
    system_prompt, user_prompt = build_generation_prompt(category, version, focus=focus)
    log_path = LOG_DIR / f"step1_{run_label}.jsonl"
    try:
        item, completion = _call_claude(system_prompt, user_prompt)
    except Exception as exc:  # noqa: BLE001 — we want to log and continue, never crash the run
        _append_jsonl(
            log_path,
            {
                "trace_id": trace_id,
                "category": category.value,
                "prompt_version": version,
                "focus": focus,
                "timestamp": _now_iso(),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return None

    record = GeneratedRecord(
        trace_id=trace_id,
        category=category,
        prompt_version=version,
        model=settings.generator_model,
        temperature=settings.generator_temperature,
        generated_at=_now_iso(),
        item=item,
    )

    # Verbose audit log, including the raw model response.
    _append_jsonl(
        log_path,
        {
            "trace_id": trace_id,
            "category": category.value,
            "prompt_version": version,
            "focus": focus,
            "model": settings.generator_model,
            "timestamp": record.generated_at,
            "status": "ok",
            "raw_response": completion.model_dump(mode="json"),
        },
    )
    return record


def generate_batch(count: int, run_label: str, version: str) -> list[GeneratedRecord]:
    """Generate `count` items, round-robin across the 5 categories for even coverage."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    categories = cycle(Category)  # appliance, plumbing, electrical, hvac, general, repeat
    out_path = DATA_DIR / f"{run_label}.jsonl"

    # Per-category counter so each item in a category draws the NEXT sub-topic (keeps a
    # 10-per-category batch from converging on the same duplicate question).
    per_category_index: Counter[Category] = Counter()

    records: list[GeneratedRecord] = []
    ok, failed = 0, 0
    for i in range(count):
        category = next(categories)
        focus = subtopic_for(category, per_category_index[category])
        per_category_index[category] += 1
        trace_id = f"{run_label}_{i:03d}"
        print(f"[{i + 1}/{count}] generating {category.value} ({trace_id})...", flush=True)

        record = generate_item(category, trace_id, version, run_label, focus=focus)
        if record is None:
            failed += 1
        else:
            records.append(record)
            _append_jsonl(out_path, record.model_dump(mode="json"))
            ok += 1

        # Basic rate limiting between calls.
        if i < count - 1:
            time.sleep(settings.request_delay_seconds)

    print(f"\nDone. {ok} generated, {failed} failed. Data -> {out_path}")
    return records


def _next_trace_index(run_label: str) -> int:
    """Next free numeric suffix for this run's trace ids.

    Scans the Step 1 LOG (which records every attempt — successes AND errors) so a
    top-up never reuses an index from an earlier attempt, even one that failed and so
    never made it into the dataset file. Returns 0 if the run has no log yet.
    """
    log_path = LOG_DIR / f"step1_{run_label}.jsonl"
    if not log_path.exists():
        return 0
    max_idx = -1
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            trace_id = json.loads(line).get("trace_id", "") or ""
        except json.JSONDecodeError:
            continue
        match = re.search(r"_(\d+)$", trace_id)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def topup_category(
    category: Category,
    target_successes: int,
    run_label: str,
    version: str,
    max_attempts: int | None = None,
) -> list[GeneratedRecord]:
    """Generate items for ONE category until `target_successes` of them succeed.

    Used to rebalance a run whose category mix fell short of the 20% floor (e.g. HVAC).
    Because generation can fail, we loop on *successes*, not attempts — but cap total
    attempts (default 3x) so a persistently failing category can't spin forever. New
    records append to the same {run_label}.jsonl, with trace ids continuing past the
    highest existing index, so downstream steps see one combined, collision-free set.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    out_path = DATA_DIR / f"{run_label}.jsonl"
    start = _next_trace_index(run_label)
    cap = max_attempts if max_attempts is not None else target_successes * 3

    records: list[GeneratedRecord] = []
    ok, attempt = 0, 0
    while ok < target_successes and attempt < cap:
        trace_id = f"{run_label}_{start + attempt:03d}"
        focus = subtopic_for(category, start + attempt)  # keep top-ups diverse too
        print(f"[top-up {ok + 1}/{target_successes}] {category.value} ({trace_id})...", flush=True)
        record = generate_item(category, trace_id, version, run_label, focus=focus)
        attempt += 1
        if record is not None:
            records.append(record)
            _append_jsonl(out_path, record.model_dump(mode="json"))
            ok += 1
        if ok < target_successes and attempt < cap:
            time.sleep(settings.request_delay_seconds)

    if ok < target_successes:
        print(f"\n⚠ top-up incomplete: {ok}/{target_successes} {category.value} after {attempt} attempts.")
    else:
        print(f"\nDone. Added {ok} {category.value} item(s) -> {out_path}")
    return records


def _append_jsonl(path: Path, obj: dict) -> None:
    """Append one JSON object as a line. JSONL is append-friendly and stream-readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1 — generate DIY repair Q&A items.")
    parser.add_argument("--count", type=int, default=10,
                        help="How many items to generate. With --category, how many must SUCCEED.")
    parser.add_argument("--run-label", default="baseline", help="Output filename / trace prefix.")
    parser.add_argument("--version", default=DEFAULT_GENERATOR_VERSION, help="Generator prompt version.")
    parser.add_argument("--category", choices=[c.value for c in Category], default=None,
                        help="Top-up mode: generate ONLY this category until --count items succeed, "
                             "appending to an existing run (used to rebalance the 20%%/category floor).")
    args = parser.parse_args()

    if args.category:
        topup_category(
            Category(args.category), target_successes=args.count,
            run_label=args.run_label, version=args.version,
        )
    else:
        generate_batch(count=args.count, run_label=args.run_label, version=args.version)


if __name__ == "__main__":
    main()
