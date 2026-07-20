# diy-home-synthetic-data

An automated pipeline that **generates** synthetic Home DIY Repair Q&A data with an
LLM, **evaluates** it across 6 quality dimensions with both a human reviewer and an
independent LLM-as-Judge, **calibrates** the judge against the human, and **iteratively
corrects** the generation prompt to prove a measurable improvement in data quality.

> Bootcamp mini-project (AI Accelerator Bootcamp). Built with Python, Anthropic Claude,
> Instructor + Pydantic, and matplotlib/seaborn.

## The core result

The pipeline must prove a data-driven improvement:

```
improvement = (baseline_failure_rate - corrected_failure_rate) / baseline_failure_rate  >= 0.80
```

## Architecture (6 steps)

1. **Generation** — parameterized prompt → Claude → 7-field Q&A items across 5 categories
2. **Data Quality Gate** — schema + per-dimension pre-checks; dedup; category distribution vs. benchmark
3. **Human Labeling (CLI)** — binary pass/fail on 6 dimensions for ≥ 20 items
4. **LLM-as-Judge** — same 6 dimensions scored on every item (low temperature)
5. **Analysis & Visualization** — segment metrics + charts
6. **Iteration** — Phase A: calibrate judge to ≥80% agreement · Phase B: correct generator prompt

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
copy .env.example .env         # Windows  (cp on macOS/Linux)
#   then edit .env and paste your real ANTHROPIC_API_KEY
```

## Running the pipeline

Every step is a standalone module run with `python -m`. Steps 1 and 4 call the Claude API
(cost money); Steps 2, 3, and 5 do not. A `--run-label` names the dataset a step reads/writes,
which keeps the **weak** (before) and **baseline** (after) runs in separate files.

```bash
# ── The "after" / corrected run (the clean generator) ───────────────────────
python -m src.step1_generate --count 50 --run-label baseline            # 1. generate  (API)
python -m src.step2_gate                --run-label baseline            # 2. quality gate
python -m src.step3_label               --run-label baseline            # 3. human labeling (interactive CLI, ≥20 items)
python -m src.step4_judge               --run-label baseline            # 4. LLM-as-judge  (API)

# ── The "before" / weak run (deliberately degraded prompt — see docs/iteration_log.md) ──
python -m src.step1_generate --count 50 --run-label weak --version generator_v0_weak   # (API)
python -m src.step2_gate                --run-label weak
python -m src.step4_judge               --run-label weak                # (API)

# ── Analysis + charts (no API calls; reads the committed label files) ───────
python -m src.step5_analyze --before weak --after baseline              # 5. metrics + visualizations
python -m src.step6_iterate --before weak --after baseline              # 6. deliverable verdict (PASS/FAIL)
```

Handy flags: `step4_judge --dry-run` prints a judge prompt without spending credits;
`step1_generate --category electrical_repair --count 1` tops up one category to hold the
20 %/category floor; both label steps are resumable (they skip already-done items).

## Results

Scored by the calibrated LLM judge (Haiku 4.5 @ temp 0.0), n = 50 per run:

| | before (`generator_v0_weak`) | after (`generator_v1_baseline`) |
|---|---|---|
| Overall failure rate | **30 %** (15/50) | **2 %** (1/50) |

```
improvement = (0.30 − 0.02) / 0.30 = 0.93   →   93 %   ≥ 0.80   ✅ PASS
```

Human ↔ judge agreement (Phase A calibration, 30-item overlap) is **≥ 93 % on every dimension**,
so no judge recalibration was needed. Charts in [`visualizations/`](./visualizations/):
`failure_before_after.png`, `agreement_by_dimension.png`, `weak_segment_heatmap.png`. Full
decision record in [`docs/iteration_log.md`](./docs/iteration_log.md); metrics JSON in
[`data/reports/step5_analysis.json`](./data/reports/step5_analysis.json).

## Repository layout

See [`CLAUDE.md`](./CLAUDE.md) for the full project context and
[`AGENT.md`](./AGENT.md) for the mentoring/working style used to build it.
