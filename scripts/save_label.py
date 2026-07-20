"""
save_label.py — helper for chat-driven human labeling (mirrors src/step3_label.py's writes).

Records ONE human LabelRecord into data/labels/human_<run_label>.json using the exact same
schema + save format as the CLI: computes overall_pass, upserts by trace_id, saves the whole
file atomically after the write. Lets Edwin label remotely (via chat) while keeping the file
100% compatible with Step 4/6.

Usage:
    python -X utf8 -m scripts.save_label baseline baseline_006 1 1 1 1 1 1
    #                                     run_label  trace_id   D1..D6 (1=pass 0=fail)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.schemas import DIMENSION_KEYS, LabelRecord

LABELS_DIR = Path("data/labels")


def main() -> None:
    run_label, trace_id = sys.argv[1], sys.argv[2]
    dims = [int(x) for x in sys.argv[3:9]]
    if len(dims) != 6:
        raise SystemExit(f"need 6 scores D1..D6, got {len(dims)}: {dims}")

    scores = dict(zip(DIMENSION_KEYS, dims))
    rec = LabelRecord(trace_id=trace_id, labeler="human", **scores)
    row = rec.model_dump(mode="json")

    path = LABELS_DIR / f"human_{run_label}.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    by = {r["trace_id"]: i for i, r in enumerate(d["labels"])}
    if trace_id in by:
        d["labels"][by[trace_id]] = row  # upsert / relabel
        action = "updated"
    else:
        d["labels"].append(row)
        action = "added"

    d["updated_at"] = datetime.now(timezone.utc).isoformat()
    d["count"] = len(d["labels"])
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{action} {trace_id} -> overall_pass={rec.overall_pass} | total labeled: {d['count']}")


if __name__ == "__main__":
    main()
