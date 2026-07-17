"""
schemas.py — The data contracts for the whole pipeline.

Everything that flows between the 6 steps is one of these Pydantic models.
Two ideas drive the design:

  1. `QAItem` is EXACTLY the 7 content fields the LLM must generate — nothing more.
     Instructor uses it as the contract: Claude's output must match it or retry.

  2. `LabelRecord` is the 6-dimension pass/fail scoring record. The human reviewer
     (Step 3) and the LLM-as-Judge (Step 4) both produce this identical shape, which
     is what lets us compute per-dimension human-vs-LLM agreement later.

Pipeline metadata (trace_id, category, timestamps) is wrapped AROUND these models by
our own code — it is not the LLM's job to invent it.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# The 5 repair categories. Using an Enum (instead of loose strings) means a typo
# like "plumbng" becomes an immediate, loud error instead of a silent bad segment.
# ---------------------------------------------------------------------------
class Category(str, Enum):
    APPLIANCE = "appliance_repair"
    PLUMBING = "plumbing_repair"
    ELECTRICAL = "electrical_repair"
    HVAC = "hvac_maintenance"
    GENERAL = "general_home_repair"


# A binary quality score: 1 = pass, 0 = fail. `ge/le` reject anything outside {0,1}.
BinaryScore = Annotated[int, Field(ge=0, le=1)]


# ---------------------------------------------------------------------------
# QAItem — the 7-field schema the generator (Step 1) must produce.
# The validation rules here ARE the structural half of the Step 2 gate.
# ---------------------------------------------------------------------------
class QAItem(BaseModel):
    # str_strip_whitespace: trims every string field automatically, so "   " is
    # treated as empty and correctly fails the min_length check below.
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, description="A realistic DIY repair question from a homeowner")
    answer: str = Field(min_length=1, description="A clear, actionable, step-by-step answer")
    equipment_problem: str = Field(min_length=1, description="The specific problem, e.g. 'dripping faucet'")
    tools_required: list[str] = Field(min_length=1, description="Tools a typical homeowner would own")
    steps: list[str] = Field(min_length=3, description="Ordered repair steps (at least 3)")
    safety_info: str = Field(min_length=1, description="Hazard-specific safety warnings and precautions")
    tips: list[str] = Field(min_length=1, description="Non-obvious, task-specific practical tips")

    @field_validator("tools_required", "steps", "tips")
    @classmethod
    def _clean_string_lists(cls, items: list[str]) -> list[str]:
        """Strip each list entry and drop any that are blank.

        Without this, the LLM could return ["", "  ", "real step"] and pass the
        min_length count while carrying junk. We clean first, then the count rule
        (min_length) is enforced against the *real* content.
        """
        cleaned = [s.strip() for s in items if s and s.strip()]
        return cleaned


# ---------------------------------------------------------------------------
# LabelRecord — the 6-dimension pass/fail record (Steps 3 & 4).
# Identical shape for human and LLM judge → enables per-dimension agreement.
# ---------------------------------------------------------------------------
class LabelRecord(BaseModel):
    trace_id: str = Field(description="Links this label back to its QA item")
    labeler: Literal["human", "llm_judge"]

    # The 6 quality dimensions (D1–D6). 1 = pass, 0 = fail.
    answer_completeness: BinaryScore     # D1
    safety_specificity: BinaryScore      # D2
    tool_realism: BinaryScore            # D3
    scope_appropriateness: BinaryScore   # D4
    context_clarity: BinaryScore         # D5
    tip_usefulness: BinaryScore          # D6

    # An item passes overall only if ALL 6 dimensions pass. We COMPUTE this rather
    # than trust a labeler to set it, so it can never contradict the 6 scores.
    overall_pass: bool = False

    @model_validator(mode="after")
    def _compute_overall_pass(self) -> "LabelRecord":
        self.overall_pass = all(
            score == 1
            for score in (
                self.answer_completeness,
                self.safety_specificity,
                self.tool_realism,
                self.scope_appropriateness,
                self.context_clarity,
                self.tip_usefulness,
            )
        )
        return self

    @property
    def dimension_scores(self) -> dict[str, int]:
        """The 6 scores as a dict, in D1–D6 order. Handy for tables and agreement math."""
        return {
            "answer_completeness": self.answer_completeness,
            "safety_specificity": self.safety_specificity,
            "tool_realism": self.tool_realism,
            "scope_appropriateness": self.scope_appropriateness,
            "context_clarity": self.context_clarity,
            "tip_usefulness": self.tip_usefulness,
        }


# The 6 dimension keys in canonical D1–D6 order — imported wherever we loop over
# dimensions (CLI prompts, judge rubric, charts) so the order is defined in ONE place.
DIMENSION_KEYS: tuple[str, ...] = (
    "answer_completeness",   # D1
    "safety_specificity",    # D2
    "tool_realism",          # D3
    "scope_appropriateness", # D4
    "context_clarity",       # D5
    "tip_usefulness",        # D6
)

# Human-readable labels for the same dimensions (for CLI prompts and chart axes).
DIMENSION_LABELS: dict[str, str] = {
    "answer_completeness": "D1 Answer Completeness",
    "safety_specificity": "D2 Safety Specificity",
    "tool_realism": "D3 Tool Realism",
    "scope_appropriateness": "D4 Scope Appropriateness",
    "context_clarity": "D5 Context Clarity",
    "tip_usefulness": "D6 Tip Usefulness",
}

# The binding "fails when…" definition of each dimension, straight from the spec. This is
# the ONE rubric both graders use: the Step 3 human CLI shows it to the reviewer, and the
# Step 4 judge prompt is built from it — so "pass/fail" means the same thing to both,
# which is what makes their per-dimension agreement (Step 6 calibration) meaningful.
DIMENSION_GUIDANCE: dict[str, str] = {
    "answer_completeness": "FAIL if the answer skips key stages needed to actually finish the repair.",
    "safety_specificity": "FAIL if safety is generic ('be careful') instead of the specific hazard + precaution.",
    "tool_realism": "FAIL if it needs a pro/specialty tool (>$50 or trade-only) a typical homeowner won't own.",
    "scope_appropriateness": "FAIL if it gives amateur steps for a job that truly needs a pro (gas, main panel).",
    "context_clarity": "FAIL if the answer doesn't actually address the stated equipment_problem.",
    "tip_usefulness": "FAIL if the tips just restate a step or give generic encouragement.",
}
