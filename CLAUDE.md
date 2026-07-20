# CLAUDE.md — Project Context for Claude Code

> This file is loaded automatically into Claude Code's context every session.
> It is the project's "single source of truth" so the assistant always knows what
> we're building, how it's structured, and the rules it must follow.

---

## 1. What this project is

**diy-home-synthetic-data** is an automated pipeline that uses an LLM to **generate**
synthetic Home DIY Repair Q&A data, then **evaluates** that data's quality across
**6 dimensions**, **calibrates** an LLM judge against human labels, and **iteratively
corrects** the generation prompt to prove a measurable, data-driven improvement.

It is a bootcamp mini-project and a **portfolio piece** evaluated by instructors and
potential employers. Code clarity, documentation, and reproducibility matter as much
as correctness.

**The climax deliverable** is a before/after comparison proving:
```
improvement = (baseline_failure_rate - corrected_failure_rate) / baseline_failure_rate  >= 0.80
```
A pipeline that runs cleanly but shows no improvement does NOT pass.

---

## 2. Tech stack

- **Python 3.10+**
- **Anthropic (Claude)** as the LLM provider
- **Instructor + Pydantic** for schema-safe structured output (both generator and judge)
- **pandas / numpy** for aggregation
- **matplotlib / seaborn** for charts
- **datasets** (Hugging Face) to load the benchmark for the Step 2 distribution check
- **python-dotenv** for config, **tenacity** for retries

Config lives in `.env` (git-ignored). Models/temperatures are read from env vars —
see `.env.example`.

---

## 3. The 6-step architecture

Each step is independently runnable and produces output that feeds the next.

| Step | Name | Output |
|------|------|--------|
| 1 | **Generation** — one parameterized prompt → Claude (via Instructor) → Q&A items across all 5 categories | `data/generated/*.jsonl` + raw-response log |
| 2 | **Data Quality Gate** — per-item schema + cheap per-dimension pre-checks; batch dedup + category-distribution vs. benchmark | gate report; only passing items advance |
| 3 | **Human Labeling (CLI)** — reviewer gives binary pass/fail on all 6 dims for ≥ 20 items | `data/labels/human_*.json` |
| 4 | **LLM-as-Judge** — independent judge scores the same 6 dims for every item (low temp, structured) | `data/labels/judge_*.json` |
| 5 | **Analysis & Visualization** — segment-level metrics + all charts | `data/reports/*.json` + `visualizations/*.png` |
| 6 | **Iteration** — Phase A: calibrate judge to ≥80% agreement; Phase B: fix worst segment×dim, re-run, prove >80% improvement | revised prompts + iteration log |

---

## 4. The 7-field Q&A schema

Every generated item must have: `question`, `answer`, `equipment_problem`,
`tools_required` (list), `steps` (list, ≥3), `safety_info`, `tips` (list, ≥1).
Validation rules: `question`/`answer` non-empty; `steps` ≥ 3; `tools_required` ≥ 1;
`tips` ≥ 1; `safety_info` non-empty.

## 5. The 6 quality dimensions (binary pass/fail, used identically by human + judge)

| Code | Dimension | Fails when… |
|------|-----------|-------------|
| D1 | Answer Completeness | answer skips key stages |
| D2 | Safety Specificity | generic ("be careful") instead of the specific hazard + precaution |
| D3 | Tool Realism | needs a pro/specialty tool (>$50 or trade-only) |
| D4 | Scope Appropriateness | gives amateur steps for a job that needs a pro (gas, panel) |
| D5 | Context Clarity | answer doesn't address the stated `equipment_problem` |
| D6 | Tip Usefulness | tips just restate a step or give generic encouragement |

**Overall pass = all 6 dimensions pass.**

**LLM-judge quality thresholds:** D1 ≥85%, D2 ≥90%, D3 ≥95%, D4 ≥95%, D5 ≥90%,
D6 ≥85%, overall ≥80%, human/LLM agreement ≥80% per dim.

## 6. The 5 repair categories (each must be ≥ 20% of the set, benchmark-aligned)

Appliance Repair · Plumbing Repair · Electrical Repair · HVAC Maintenance · General Home Repair

Benchmark dataset (distribution reference ONLY, not for labeling):
`dipenbhuva/home-diy-repair-qa` on Hugging Face (5,000 items, uniform 20%/category).

---

## 7. Repository layout

```
.
├── CLAUDE.md            # this file — project context for Claude Code
├── AGENT.md             # the "AI Senior Engineer" mentor charter
├── README.md            # human-facing run instructions
├── .env.example         # config template (real .env is git-ignored)
├── requirements.txt
├── src/                 # pipeline code (one module per step)
├── prompts/             # generator + judge prompt templates (versioned, NOT hardcoded)
├── data/
│   ├── generated/       # Step 1 output (JSONL)
│   ├── labels/          # Step 3 human + Step 4 judge labels
│   └── reports/         # Step 5 metrics, before/after comparison
├── visualizations/      # Step 5 chart PNGs
├── logs/                # per-step logs + per-item trace records
└── docs/                # iteration log, design notes
```

---

## 8. Hard rules (do not violate)

1. **No hardcoded repair answers.** All Q&A content is LLM-generated at runtime.
2. **Never commit secrets.** The API key lives only in `.env`.
3. **Judge temperature < generator temperature** (judge near 0 for determinism).
4. **Prompts are not hardcoded in logic** — keep them in `prompts/` (config/templates) so variants are swappable.
5. **Every prompt correction must be data-driven** and logged in the iteration log
   (hypothesis → change → before/after metric → decision). No intuition-only tweaks.
6. **Handle malformed LLM responses gracefully** — retry, log, never crash the run.
7. **Rate-limit**: small delay between API calls.
8. **Baseline and corrected outputs get separate filenames** (needed for comparison).
9. **Sample size ≥ 30** for any reported rate (failure rates are noisy below that).

---

## 9. Common commands

```bash
pip install -r requirements.txt                                          # install deps

# Pipeline (per run-label). Steps 1 & 4 hit the Claude API; 2, 3, 5 do not.
python -m src.step1_generate --count 50 --run-label baseline             # 1. generate (API)
python -m src.step2_gate                --run-label baseline             # 2. quality gate
python -m src.step3_label               --run-label baseline             # 3. human labeling (interactive)
python -m src.step4_judge               --run-label baseline             # 4. LLM-as-judge (API)
python -m src.step5_analyze --before weak --after baseline               # 5. metrics + charts

# Step 6 Phase B — the deliberately-weakened "before" run (see docs/iteration_log.md)
python -m src.step1_generate --count 50 --run-label weak --version generator_v0_weak   # (API)
python -m src.step2_gate                --run-label weak
python -m src.step4_judge               --run-label weak                 # (API)

# Useful flags
python -m src.step4_judge    --run-label baseline --dry-run              # print a judge prompt, no API cost
python -m src.step1_generate --run-label baseline --category electrical_repair --count 1   # top up one category
python -m scripts.save_label baseline <trace_id> 1 1 1 1 1 1            # record one human label non-interactively
```

---

## 10. Working style

See **AGENT.md** — the owner (Edwin, an AI Engineering student) wants decisions,
code, and architecture **explained as we go**, senior-engineer style, to learn and
to be able to defend every choice in an interview.
