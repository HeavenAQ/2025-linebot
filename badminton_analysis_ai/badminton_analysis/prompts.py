"""Load GPT prompt text from `badminton_analysis_ai/prompts/`."""

from __future__ import annotations

from functools import cache
from pathlib import Path

PROMPT_DIRECTORY = Path(__file__).resolve().parents[1] / "prompts"


@cache
def _text(name: str) -> str:
    text = (PROMPT_DIRECTORY / f"{name}.txt").read_text(encoding="utf-8")
    return text[:-1] if text.endswith("\n") else text


def prompt(name: str, /, **fields: object) -> str:
    """The prompt `<name>.txt` with its `{placeholders}` filled in."""
    text = _text(name)
    return text.format(**fields) if fields else text
