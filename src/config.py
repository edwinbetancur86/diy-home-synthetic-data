"""
config.py — Single source of truth for configuration.

Loads settings from `.env` exactly once and exposes them as a typed `Settings`
object, plus factory functions for the Instructor-patched Anthropic clients used by
the generator (Step 1) and the judge (Step 4).

Why centralize this?
  - One place reads secrets, so we fail loudly and early if the key is missing.
  - Generator and judge get DIFFERENT models/temperatures (spec rule: the judge must
    be more deterministic than the generator). Defining that split here keeps every
    other module from having to know the details.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import instructor
from anthropic import Anthropic
from dotenv import load_dotenv

# Read the .env file into environment variables (no-op if already loaded).
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of everything the pipeline needs to talk to Claude."""

    anthropic_api_key: str
    generator_model: str
    judge_model: str
    generator_temperature: float
    judge_temperature: float
    # A tiny pause between API calls so we don't trip rate limits (spec: "Don't skip
    # rate limiting"). Kept here so it's tunable in one spot.
    request_delay_seconds: float = 1.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build the Settings once and cache it. Raises a clear error if the key is absent."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key or api_key == "your-anthropic-api-key-here":
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and paste your "
            "real key into it (never commit .env)."
        )

    return Settings(
        anthropic_api_key=api_key,
        generator_model=os.getenv("GENERATOR_MODEL", "claude-sonnet-4-6"),
        judge_model=os.getenv("JUDGE_MODEL", "claude-haiku-4-5-20251001"),
        generator_temperature=float(os.getenv("GENERATOR_TEMPERATURE", "0.8")),
        judge_temperature=float(os.getenv("JUDGE_TEMPERATURE", "0.0")),
    )


@lru_cache(maxsize=1)
def get_client() -> instructor.Instructor:
    """Return an Instructor-patched Anthropic client.

    `instructor.from_anthropic(...)` wraps the normal Anthropic client so that
    `client.messages.create(..., response_model=SomeModel)` forces the LLM's output
    to validate against `SomeModel` (retrying automatically if it doesn't). This is
    what turns free-form text into schema-safe structured output.
    """
    settings = get_settings()
    return instructor.from_anthropic(Anthropic(api_key=settings.anthropic_api_key))
