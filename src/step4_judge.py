"""
step4_judge.py — Step 4: LLM-as-Judge.

Runs an independent LLM judge over EVERY gated item (Step 2 output) and scores the same 6
quality dimensions the human scored in Step 3, producing a `data/labels/judge_<run_label>.json`
in the identical shape as the human file. Step 5 then measures human-vs-judge agreement (the
Step 6 Phase A calibration target) and per-dimension quality on the full set.

Why this mirrors Step 1 (`step1_generate.py`) so closely — same reasons, same spec rules:
  - Instructor-patched Claude client + `create_with_completion()` → schema-safe structured
    output (a `JudgeVerdict`) AND the raw response for the audit log.
  - tenacity retry on transient API errors; Instructor `max_retries` on schema mismatch →
    a single bad response never crashes the run (rule #6).
  - a small delay between calls → basic rate limiting (rule #7).
  - clean labels go to data/labels/; verbose per-item traces (scores + rationale + raw
    response) go to logs/ (the audit trail).

Two choices specific to the judge:
  1. INDEPENDENCE. The judge never sees the human labels — it scores from the item alone,
     against the SAME rubric (`DIMENSION_GUIDANCE`, via prompts/judge.py). That is what makes
     their agreement a meaningful calibration number rather than a circular one.
  2. RESUMABLE + CRASH-SAFE + IDEMPOTENT. Like the Step 3 CLI, we save after every item and
     skip already-judged trace_ids on restart. Given a small API-credit budget and a machine
     that has shut down mid-run before, never re-spending on work already done matters.

Judge determinism (rule #3): the judge runs at `JUDGE_TEMPERATURE` (near 0), which must be
below the generator's temperature. We assert that at runtime rather than trusting config.

Run it:
    python -m src.step4_judge --run-label baseline            # judge all gated items
    python -m src.step4_judge --run-label baseline --limit 5  # judge at most 5 new items
    python -m src.step4_judge --run-label baseline --dry-run  # print one prompt, no API call
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from prompts.judge import DEFAULT_JUDGE_VERSION, build_judge_prompt
from src.config import get_client, get_settings
from src.schemas import DIMENSION_KEYS, BinaryScore, LabelRecord
from src.step1_generate import GeneratedRecord

GENERATED_DIR = Path("data/generated")
LABELS_DIR = Path("data/labels")
LOG_DIR = Path("logs")

# The judge's output is tiny (6 scores + a short rationale); 1500 is comfortable headroom.
JUDGE_MAX_TOKENS = 1500


# ---------------------------------------------------------------------------
# JudgeVerdict — the judge's RAW structured output for one item.
#
# Field names are the exact 6 dimension keys, so converting to a LabelRecord is a clean
# splat. Like QAItem, this is only the model's job — pipeline metadata (trace_id, labeler,
# timestamps) is wrapped around it by our code, not invented by the LLM.
# ---------------------------------------------------------------------------
class JudgeVerdict(BaseModel):
    """6 binary scores (1 = pass, 0 = fail) plus a brief rationale."""

    answer_completeness: BinaryScore     # D1
    safety_specificity: BinaryScore      # D2
    tool_realism: BinaryScore            # D3
    scope_appropriateness: BinaryScore   # D4
    context_clarity: BinaryScore         # D5
    tip_usefulness: BinaryScore          # D6
    rationale: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, obj: dict) -> None:
    """Append one JSON object as a line (the per-item audit trail)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Loading: gated dataset in, existing judge labels (for resume) in.
# ---------------------------------------------------------------------------
def _load_gated(run_label: str) -> list[GeneratedRecord]:
    """Read the Step 2 gated JSONL. Loud error if missing — don't judge ungated data."""
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
    return LABELS_DIR / f"judge_{run_label}.json"


def _load_existing(run_label: str, version: str) -> dict:
    """Load prior judge labels for resume, or start a fresh container. Keyed by trace_id."""
    path = _labels_path(run_label)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_by_trace"] = {rec["trace_id"]: rec for rec in data.get("labels", [])}
        return data
    settings = get_settings()
    return {
        "run_label": run_label,
        "labeler": "llm_judge",
        "model": settings.judge_model,
        "temperature": settings.judge_temperature,
        "judge_version": version,
        "created_at": _now_iso(),
        "updated_at": None,
        "labels": [],
        "_by_trace": {},
    }


def _save(run_label: str, container: dict) -> None:
    """Persist judge labels. Called after every item so a crash never loses more than nothing."""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    container["updated_at"] = _now_iso()
    out = {k: v for k, v in container.items() if not k.startswith("_")}
    out["count"] = len(out["labels"])
    _labels_path(run_label).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# One judge call.
# ---------------------------------------------------------------------------
# Retry only on transient, retriable API errors. Schema/validation retries are Instructor's
# job (max_retries below), so we don't double-handle those here.
@retry(
    retry=retry_if_exception_type(
        (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError)
    ),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _call_judge(system_prompt: str, user_prompt: str) -> tuple[JudgeVerdict, object]:
    """One judging call. Returns the validated JudgeVerdict plus the raw completion."""
    settings = get_settings()
    client = get_client()
    verdict, completion = client.messages.create_with_completion(
        model=settings.judge_model,
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=settings.judge_temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        response_model=JudgeVerdict,
        max_retries=2,  # Instructor re-asks the model if the output fails validation
    )
    return verdict, completion


def judge_item(
    record: GeneratedRecord, version: str, run_label: str
) -> LabelRecord | None:
    """Judge one item. Returns a LabelRecord, or None (and logs) on unrecoverable failure."""
    settings = get_settings()
    system_prompt, user_prompt = build_judge_prompt(record.item, version)
    log_path = LOG_DIR / f"step4_judge_{run_label}.jsonl"
    try:
        verdict, completion = _call_judge(system_prompt, user_prompt)
    except Exception as exc:  # noqa: BLE001 — log and continue, never crash the run (rule #6)
        _append_jsonl(
            log_path,
            {
                "trace_id": record.trace_id,
                "category": record.category.value,
                "judge_version": version,
                "timestamp": _now_iso(),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return None

    scores = {dim: getattr(verdict, dim) for dim in DIMENSION_KEYS}
    label = LabelRecord(trace_id=record.trace_id, labeler="llm_judge", **scores)

    # Verbose audit log: scores, the judge's rationale, and the raw model response.
    _append_jsonl(
        log_path,
        {
            "trace_id": record.trace_id,
            "category": record.category.value,
            "judge_version": version,
            "model": settings.judge_model,
            "temperature": settings.judge_temperature,
            "timestamp": _now_iso(),
            "status": "ok",
            "scores": scores,
            "overall_pass": label.overall_pass,
            "rationale": verdict.rationale,
            "raw_response": completion.model_dump(mode="json"),
        },
    )
    return label


# ---------------------------------------------------------------------------
# The run loop.
# ---------------------------------------------------------------------------
def run_judging(run_label: str, version: str, limit: int | None = None) -> None:
    settings = get_settings()

    # Rule #3: the judge must be at least as deterministic as the generator. Guard it here
    # rather than trusting whatever is in .env.
    if settings.judge_temperature > settings.generator_temperature:
        raise RuntimeError(
            f"judge_temperature ({settings.judge_temperature}) must not exceed "
            f"generator_temperature ({settings.generator_temperature}) — the judge has to be "
            f"the more deterministic of the two (spec rule #3). Fix JUDGE_TEMPERATURE in .env."
        )

    gated = _load_gated(run_label)
    container = _load_existing(run_label, version)
    done_ids = set(container["_by_trace"])

    todo = [r for r in gated if r.trace_id not in done_ids]
    if limit is not None:
        todo = todo[:limit]

    print(f"\nStep 4 — LLM-as-Judge: {run_label}")
    print(f"  judge model:      {settings.judge_model}  (temp {settings.judge_temperature})")
    print(f"  judge prompt:     {version}")
    print(f"  gated items:      {len(gated)}")
    print(f"  already judged:   {len(done_ids)}")
    print(f"  to judge now:     {len(todo)}"
          + (f"  (--limit {limit})" if limit is not None else ""))
    if not todo:
        print("\nNothing left to judge. ✔")
        _summary(container)
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, 0
    for i, record in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] judging {record.category.value} ({record.trace_id})...",
              flush=True)
        label = judge_item(record, version, run_label)
        if label is None:
            failed += 1
        else:
            container["labels"].append(label.model_dump(mode="json"))
            container["_by_trace"][record.trace_id] = container["labels"][-1]
            _save(run_label, container)  # crash-safe: persist immediately
            ok += 1
            verdict = "PASS" if label.overall_pass else "FAIL"
            print(f"   ✔ {verdict}")

        # Basic rate limiting between calls.
        if i < len(todo):
            time.sleep(settings.request_delay_seconds)

    _save(run_label, container)
    print(f"\nDone. Judged {ok} new item(s), {failed} failed. Labels -> {_labels_path(run_label)}")
    _summary(container)


def _summary(container: dict) -> None:
    """Recap totals + per-dimension fail counts for the judge's labels."""
    labels = container["labels"]
    n = len(labels)
    print(f"\n  total judged: {n}")
    if n == 0:
        return
    overall_pass = sum(1 for r in labels if r["overall_pass"])
    print(f"  overall pass: {overall_pass}/{n}  ({overall_pass / n:.0%})")
    print("  per-dimension fail counts:")
    for dim in DIMENSION_KEYS:
        fails = sum(1 for r in labels if r[dim] == 0)
        print(f"    {dim:24} {fails} fail(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4 — LLM-as-Judge (6 dimensions).")
    parser.add_argument("--run-label", default="baseline", help="Which gated run to judge.")
    parser.add_argument("--version", default=DEFAULT_JUDGE_VERSION, help="Judge prompt version.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Judge at most this many NEW items this run.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the judge prompt for the first un-judged item and exit "
                             "(no API call, no cost).")
    args = parser.parse_args()

    if args.dry_run:
        gated = _load_gated(args.run_label)
        container = _load_existing(args.run_label, args.version)
        done = set(container["_by_trace"])
        todo = [r for r in gated if r.trace_id not in done]
        if not todo:
            print("Nothing left to judge — no prompt to show.")
            return
        system_prompt, user_prompt = build_judge_prompt(todo[0].item, args.version)
        print(f"=== DRY RUN — judge prompt for {todo[0].trace_id} "
              f"({todo[0].category.value}) ===")
        print("\n----- SYSTEM -----\n" + system_prompt)
        print("\n----- USER -----\n" + user_prompt)
        return

    run_judging(args.run_label, args.version, limit=args.limit)


if __name__ == "__main__":
    main()
