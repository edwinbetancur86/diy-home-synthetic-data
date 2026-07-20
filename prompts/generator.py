"""
generator.py — Versioned generation prompt templates for Step 1.

Prompts are DATA, not logic. Each version is an entry in GENERATOR_PROMPTS keyed by a
version string. Step 6 Phase B adds a new version (e.g. "generator_v2_corrected") so we
can compare baseline vs. corrected runs. `build_generation_prompt()` assembles the
system + user messages for a given category and version.

The baseline (v1) is intentionally competent but not exhaustive — it should leave a
real failure rate (>=15%) for the evaluation loop to detect and later fix, per the spec.
"""

from __future__ import annotations

from src.schemas import Category

# Which prompt version to use by default. Phase B will flip this (or pass an override).
DEFAULT_GENERATOR_VERSION = "generator_v1_baseline"

# Per-category flavor so the single template produces diverse, category-appropriate
# questions instead of homogeneous output (spec: "Prompt Diversity Without Repetition").
CATEGORY_GUIDANCE: dict[Category, str] = {
    Category.APPLIANCE: (
        "Home appliances: refrigerators, washing machines, dryers, dishwashers, ovens. "
        "Common issues like not cooling, not draining, not heating, strange noises."
    ),
    Category.PLUMBING: (
        "Household plumbing: leaks, clogs, running toilets, dripping faucets, "
        "low water pressure, pipe problems under sinks."
    ),
    Category.ELECTRICAL: (
        "Homeowner-level electrical only: replacing an outlet or switch, installing a "
        "light fixture, resetting a tripped breaker. NEVER main panel or service-line work."
    ),
    Category.HVAC: (
        "HVAC maintenance: changing filters, thermostat problems, cleaning vents, "
        "basic airflow and no-heat / no-cool troubleshooting."
    ),
    Category.GENERAL: (
        "General home repair: patching drywall, sticking doors/windows, squeaky floors, "
        "basic carpentry, caulking, hanging things securely."
    ),
}

# The baseline system prompt. Defines the role and the 7-field structure. It asks for
# safety and tips but does NOT yet aggressively enforce hazard-specificity or
# non-obvious tips — those are the gaps Phase B will target if the data shows failures.
_BASELINE_SYSTEM = (
    "You are a helpful home-repair expert writing Q&A training data for a DIY "
    "assistant. Produce ONE realistic homeowner repair question and a complete, "
    "practical answer.\n\n"
    "Requirements:\n"
    "- The question sounds like a real homeowner (first person, describes a symptom).\n"
    "- Provide at least 3 concrete, ordered steps.\n"
    "- List only tools a typical homeowner owns or can buy at a hardware store.\n"
    "- Include safety information relevant to the repair.\n"
    "- Include at least one practical tip.\n"
    "- Keep everything within safe DIY scope; if a job truly needs a professional, "
    "say so in the answer."
)

# ---------------------------------------------------------------------------
# Step 6 Phase B — the DELIBERATELY WEAKENED baseline (the experiment's "before").
#
# WHY this exists: the v1 prompt turned out too good — the calibrated LLM judge fails only
# ~2% of its output, which is too clean to demonstrate a measurable, data-driven improvement
# (the climax needs (before-after)/before >= 0.80, impossible off a 2% floor). So we craft a
# prompt that produces a realistic quality-failure rate to serve as the "before"; the existing
# v1 prompt is the "after". This turns "my baseline was already good" into a controlled
# experiment proving the eval harness can DETECT and QUANTIFY a regression and its repair.
#
# WHAT it degrades — and why exactly these dimensions: Step 2's gate HARD-rejects only D4
# (scope). D2/D3/D6 (and D1) are ADVISORY there (non-blocking), so failures on those SURVIVE
# the gate and reach the judge. This prompt therefore degrades the ADVISORY dimensions only:
#   - D2 Safety Specificity → asks for a brief, generic safety note (no specific hazard).
#   - D3 Tool Realism       → invites professional-grade / specialty tools.
#   - D6 Tip Usefulness      → allows tips that merely restate the steps.
#   - D1 Answer Completeness → tells it to be concise and skip stages.
# It deliberately does NOT push out-of-scope (gas/panel) DIY — that would trip the D4 hard
# reject and remove the very failures we want the judge to score. It also keeps the STRUCTURAL
# asks (>=3 steps, >=1 tool, non-empty safety, >=1 tip) so items stay schema-valid and don't
# just error out of Step 1. Same SUBTOPICS seeding as v1 — only the QUALITY guidance changes,
# so the before/after difference is attributable to the prompt, not to different topics.
# ---------------------------------------------------------------------------
_WEAK_BASELINE_SYSTEM = (
    "You are writing Q&A training data for a home-repair DIY assistant. Produce ONE realistic "
    "homeowner repair question and a quick, practical answer.\n\n"
    "Requirements:\n"
    "- The question sounds like a real homeowner (first person, describes a symptom).\n"
    "- Provide at least 3 ordered steps.\n"
    "- List the tools needed to do the job well — feel free to include professional-grade or "
    "specialty tools if they give the best result; don't restrict yourself to only basic "
    "homeowner tools.\n"
    "- Include a short, general safety note — a brief reminder to be careful is enough; you "
    "don't need to spell out the specific hazard or precaution.\n"
    "- Include at least one tip; tips can simply reinforce or restate the key steps.\n"
    "- Keep the answer concise and move quickly — you don't need to exhaustively cover every "
    "stage of the repair."
)

GENERATOR_PROMPTS: dict[str, str] = {
    "generator_v1_baseline": _BASELINE_SYSTEM,
    "generator_v0_weak": _WEAK_BASELINE_SYSTEM,
}

# ---------------------------------------------------------------------------
# Per-category sub-topic bank (DATA, not logic — rule #4).
#
# WHY this exists: the baseline sends one identical request per category. At temp 0.8
# the model still converges on each category's single most common problem (plumbing →
# "dripping faucet", electrical → "dead outlet"), so a batch of 10 calls produced many
# near-duplicate questions that Step 2's dedup then removed — starving those categories
# below the 20% floor. Seeding each call with a DISTINCT problem spreads the questions
# out so the surviving set stays balanced.
#
# This only steers the TOPIC of the question. It deliberately does NOT add quality
# guidance (hazard specificity, non-obvious tips), so the baseline keeps the real
# quality-failure rate that Step 6's improvement experiment needs to detect and fix.
# ≥10 topics per category so a 10-per-category batch gets a unique focus each time.
# ---------------------------------------------------------------------------
SUBTOPICS: dict[Category, tuple[str, ...]] = {
    Category.APPLIANCE: (
        "Refrigerator not cooling even though the freezer still works",
        "Dishwasher won't drain and leaves standing water in the bottom",
        "Clothes dryer runs but doesn't produce any heat",
        "Washing machine won't spin so clothes come out soaking wet",
        "Oven won't heat up to the temperature it's set to",
        "Refrigerator leaking water onto the kitchen floor",
        "Dishwasher leaves a white film and spots on the dishes",
        "Washing machine shakes violently during the spin cycle",
        "Microwave runs but doesn't actually heat the food",
        "Ice maker in the freezer stopped making ice",
        "Freezer keeps building up thick frost",
        "Dryer takes two or three cycles to dry a load",
    ),
    Category.PLUMBING: (
        "Toilet keeps running long after it's flushed",
        "Bathroom sink drains very slowly",
        "Kitchen faucet drips constantly even when fully off",
        "Weak, low water pressure in the shower",
        "Leak under the kitchen sink at the P-trap",
        "Garbage disposal is jammed and just hums",
        "Bathtub is draining slower and slower",
        "Toilet is clogged and a plunger isn't clearing it",
        "Showerhead is clogged with mineral buildup and sprays weakly",
        "Toilet won't seal because of a worn flapper",
        "Rusty, discolored water coming from a faucet",
        "Faucet aerator is clogged and the stream has gone weak",
    ),
    Category.ELECTRICAL: (
        "Wall outlet suddenly stopped working",
        "Light switch no longer turns on the light",
        "Breaker keeps tripping whenever a certain appliance runs",
        "Replacing a worn-out light switch",
        "Installing a new ceiling light fixture",
        "GFCI outlet in the bathroom won't reset",
        "Lights in one room keep flickering",
        "Replacing a cracked outlet receptacle",
        "Dimmer switch buzzes and feels warm",
        "Ceiling fan stopped working on every speed",
        "Under-cabinet light won't turn on",
        "Outlet has no ground and needs testing",
    ),
    Category.HVAC: (
        "Central AC runs but the house isn't cooling",
        "Furnace blows cold air instead of heat",
        "Thermostat screen is blank and unresponsive",
        "Airflow from the vents is weak",
        "AC coils keep freezing up with ice",
        "When and how to replace the HVAC air filter",
        "One room stays much hotter than the rest of the house",
        "Furnace short-cycles, turning on and off every few minutes",
        "Musty smell every time the AC kicks on",
        "Programmable thermostat won't follow its schedule",
        "Outdoor condenser unit won't turn on",
        "Excessive dust blowing out of the vents",
    ),
    Category.GENERAL: (
        "Patching a small hole in drywall",
        "Interior door sticks and won't close properly",
        "Hardwood floor squeaks in one spot",
        "Re-caulking around the bathtub",
        "Hanging a heavy mirror securely on drywall",
        "Window is painted shut and won't open",
        "Filling nail holes and touching up the paint",
        "Cabinet door hinge is loose and the door sags",
        "Draft coming in under the exterior door",
        "Wobbly stair railing that needs tightening",
        "Loose, wobbly interior doorknob",
        "Peeling caulk around a window frame",
    ),
}


def subtopic_for(category: Category, index: int) -> str:
    """Pick a sub-topic for the `index`-th item of a category (wraps round-robin)."""
    topics = SUBTOPICS[category]
    return topics[index % len(topics)]


def build_generation_prompt(
    category: Category,
    version: str = DEFAULT_GENERATOR_VERSION,
    focus: str | None = None,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for one item in the given category.

    `focus` seeds a specific sub-problem so a batch doesn't converge on duplicates.
    Raises KeyError if the version is unknown — a loud failure beats silently
    generating with the wrong prompt.
    """
    system_prompt = GENERATOR_PROMPTS[version]
    focus_line = (
        f"Focus on this SPECIFIC problem and write a distinct question about it: {focus}\n"
        if focus
        else ""
    )
    user_prompt = (
        f"Generate one Home DIY Repair Q&A item for this category:\n"
        f"Category: {category.value}\n"
        f"Category scope: {CATEGORY_GUIDANCE[category]}\n"
        f"{focus_line}\n"
        f"Return all 7 fields: question, answer, equipment_problem, tools_required, "
        f"steps, safety_info, tips."
    )
    return system_prompt, user_prompt
