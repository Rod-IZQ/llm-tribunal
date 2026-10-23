"""
Data shapes for the council.

WorkerAnswer  what every witness must return (JSON)
Testimony     one witness's full record for a run (answer, timing, errors)
Exhibit       an anonymized answer as the judge sees it (Answer A, B, C...)
Verdict       what the judge must return (JSON)
RunRecord     everything about one run, saved to runs/<timestamp>.json

parse_worker_answer() / parse_verdict() turn raw model text into these shapes,
tolerating the usual model quirks (code fences, extra text, 85 instead of 0.85).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, ValidationError, field_validator


# ---------------------------------------------------------------- shared helpers

def _normalize_confidence(value) -> float:
    """Accept 0.85, '0.85', 85, '85%' and return a float in [0, 1]."""
    if isinstance(value, str):
        value = value.strip().rstrip("%")
    v = float(value)
    if v > 1:          # model answered on a 0-100 scale
        v = v / 100
    return max(0.0, min(1.0, v))


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (fenced or not)."""
    if not text or not text.strip():
        raise ValueError("empty reply")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in reply")
        candidate = text[start : end + 1]
    # strict=False: tolerate raw line breaks/tabs inside strings (Ministral does this)
    return json.loads(candidate, strict=False)


def _flatten_text(v):
    """Some models return a field as an object or list instead of a string."""
    if isinstance(v, dict):
        return "\n".join(f"{k}: {_flatten_text(x)}" if not str(k).startswith("(") else f"{k} {_flatten_text(x)}"
                         for k, x in v.items()).strip()
    if isinstance(v, list):
        return "\n".join(str(_flatten_text(x)) for x in v).strip()
    return v


# ---------------------------------------------------------------- witness side

class WorkerAnswer(BaseModel):
    answer: str = Field(min_length=1)
    reasoning: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    caveats: list[str] = Field(default_factory=list)

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v):
        return _normalize_confidence(v)

    @field_validator("answer", "reasoning", mode="before")
    @classmethod
    def _text(cls, v):
        return _flatten_text(v)

    @field_validator("caveats", mode="before")
    @classmethod
    def _caveats(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v


class Testimony(BaseModel):
    """One witness's record for a run. `answer` is None if it failed."""
    witness: str                      # short id: groq, local, gemini...
    family: str                       # model family, for the diversity rule
    model: str
    answer: Optional[WorkerAnswer] = None
    raw_text: str = ""                # untouched reply, kept for debugging
    latency_ms: Optional[float] = None
    error: Optional[str] = None       # plain-language reason if it failed
    late: bool = False                # arrived after the soft deadline

    @property
    def usable(self) -> bool:
        return self.answer is not None and not self.late


# ---------------------------------------------------------------- judge side

class Exhibit(BaseModel):
    """What the judge sees: a letter and the answer. No witness name."""
    letter: str
    answer: WorkerAnswer


class Verdict(BaseModel):
    final_answer: str = Field(min_length=1)
    chosen: list[str] = Field(default_factory=list)   # e.g. ["B", "D"]; empty = judge overrode all
    why: str
    disagreements: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    unresolved: bool = False          # judge says the evidence can't settle it

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v):
        return _normalize_confidence(v)

    @field_validator("chosen", mode="before")
    @classmethod
    def _letters(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = re.findall(r"[A-Za-z]", v)
        out = []
        for item in v:
            letter = str(item).strip().upper().removeprefix("ANSWER").strip()
            if len(letter) != 1 or not letter.isalpha():
                raise ValueError(f"'{item}' is not an answer letter")
            out.append(letter)
        return out

    def check_letters(self, valid: set[str]) -> None:
        """Raise if the judge picked a letter that wasn't on the table."""
        bad = [c for c in self.chosen if c not in valid]
        if bad:
            raise ValueError(f"judge chose unknown answers: {bad}")


# ---------------------------------------------------------------- run log

class RunRecord(BaseModel):
    question: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    seed: int                                   # shuffle seed, for replay
    letter_map: dict[str, str] = Field(default_factory=dict)   # "A" -> "groq"
    testimonies: list[Testimony] = Field(default_factory=list)
    judge_model: str
    verdict: Optional[Verdict] = None
    judge_error: Optional[str] = None


# ---------------------------------------------------------------- parsers

class ParseError(ValueError):
    pass


def parse_worker_answer(text: str) -> WorkerAnswer:
    try:
        return WorkerAnswer.model_validate(_extract_json(text))
    except (ValueError, ValidationError) as exc:
        raise ParseError(f"bad worker JSON: {exc}") from exc


def parse_verdict(text: str, valid_letters: set[str]) -> Verdict:
    try:
        verdict = Verdict.model_validate(_extract_json(text))
        verdict.check_letters(valid_letters)
        return verdict
    except (ValueError, ValidationError) as exc:
        raise ParseError(f"bad verdict JSON: {exc}") from exc
