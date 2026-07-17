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

*(Commands are added here as each step's module is built.)*

## Repository layout

See [`CLAUDE.md`](./CLAUDE.md) for the full project context and
[`AGENT.md`](./AGENT.md) for the mentoring/working style used to build it.
