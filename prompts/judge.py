"""
judge.py — Versioned LLM-as-Judge prompt templates for Step 4.

Prompts are DATA, not logic (rule #4) — each version is an entry in JUDGE_PROMPTS keyed by
a version string, exactly like prompts/generator.py. `build_judge_prompt()` assembles the
(system, user) messages for one item.

The single most important design choice here: the judge's rubric is built from the SAME
`DIMENSION_GUIDANCE` that the Step 3 human CLI shows the reviewer. If the human and the judge
scored against different definitions of "pass", their agreement number (the Step 6 calibration
target) would be meaningless. Building both from one source keeps them honest.

The baseline judge prompt (v1) is intentionally strict and gives no benefit of the doubt —
we want it to actually detect the baseline's quality failures so Step 6 can prove they shrink.
Step 6 Phase A may add a recalibrated version (e.g. "judge_v2_calibrated") if human/judge
agreement comes in below the 80% threshold.
"""

from __future__ import annotations

from src.schemas import (
    DIMENSION_GUIDANCE,
    DIMENSION_KEYS,
    DIMENSION_LABELS,
    QAItem,
)

# Which judge prompt version to use by default. Phase A will flip this if we recalibrate.
DEFAULT_JUDGE_VERSION = "judge_v1_baseline"


def _rubric_block() -> str:
    """Render the 6 dimensions + their 'fails when…' definitions, in canonical D1–D6 order."""
    return "\n".join(
        f"- {DIMENSION_LABELS[key]} ({key}): {DIMENSION_GUIDANCE[key]}"
        for key in DIMENSION_KEYS
    )


# The baseline judge system prompt. Defines the role, the 1/0 convention, and the shared
# rubric. It deliberately tells the judge NOT to invent criteria or give benefit of the doubt,
# so its scores are comparable to a careful human's rather than being systematically lenient.
_BASELINE_JUDGE_SYSTEM = (
    "You are a strict, impartial quality evaluator for Home DIY Repair Q&A items. "
    "For ONE item you will independently score each of the 6 quality dimensions as pass or "
    "fail, using ONLY the rubric below. Do not invent new criteria and do not give the item "
    "the benefit of the doubt.\n\n"
    "Scoring convention (this is the opposite of the failure wording, so read carefully):\n"
    "  1 = PASS — the item does NOT have the described flaw.\n"
    "  0 = FAIL — the item HAS the described flaw.\n\n"
    "A dimension FAILS when…\n"
    f"{_rubric_block()}\n\n"
    "Judge each dimension on its own — a strong answer can still fail a single dimension. "
    "Be consistent and deterministic: an identical item must always receive identical scores. "
    "Give a brief rationale (one short sentence for each dimension you fail; if all 6 pass, "
    "say so briefly)."
)

JUDGE_PROMPTS: dict[str, str] = {
    "judge_v1_baseline": _BASELINE_JUDGE_SYSTEM,
}


def render_item(item: QAItem) -> str:
    """Render the 7 content fields of a QAItem as plain text for the judge to read."""
    tools = "\n".join(f"  - {t}" for t in item.tools_required)
    steps = "\n".join(f"  {i}. {s}" for i, s in enumerate(item.steps, 1))
    tips = "\n".join(f"  - {t}" for t in item.tips)
    return (
        f"QUESTION:\n{item.question}\n\n"
        f"EQUIPMENT PROBLEM:\n{item.equipment_problem}\n\n"
        f"ANSWER:\n{item.answer}\n\n"
        f"TOOLS REQUIRED ({len(item.tools_required)}):\n{tools}\n\n"
        f"STEPS ({len(item.steps)}):\n{steps}\n\n"
        f"SAFETY INFO:\n{item.safety_info}\n\n"
        f"TIPS ({len(item.tips)}):\n{tips}"
    )


def build_judge_prompt(
    item: QAItem, version: str = DEFAULT_JUDGE_VERSION
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for judging one item.

    Raises KeyError if the version is unknown — a loud failure beats silently scoring with
    the wrong rubric.
    """
    system_prompt = JUDGE_PROMPTS[version]
    user_prompt = (
        "Evaluate this Home DIY Repair Q&A item on all 6 dimensions.\n\n"
        f"{render_item(item)}\n\n"
        "Return a pass/fail score (1 = pass, 0 = fail) for each of the 6 dimensions "
        "and a brief rationale."
    )
    return system_prompt, user_prompt
