"""
Offline tests for core/schemas.py and core/prompts.py. No API calls, costs nothing.

Run from the AI_orchestrator folder:
    python tests/test_core.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from core.prompts import JUDGE_SYSTEM, WORKER_SYSTEM, judge_user, manual_judge_prompt
from core.schemas import (
    Exhibit, ParseError, RunRecord, Testimony, parse_verdict, parse_worker_answer,
)

console = Console()
passed = failed = 0


def check(name, fn):
    global passed, failed
    try:
        fn()
        console.print(f"[green]✓[/green] {name}")
        passed += 1
    except Exception as exc:
        console.print(f"[red]✗ {name}[/red]  ->  {exc}")
        failed += 1


def expect_reject(text, letters=None):
    try:
        parse_verdict(text, letters) if letters else parse_worker_answer(text)
    except ParseError:
        return
    raise AssertionError("should have been rejected")


# ---------------------------------------------------------------- witness replies

console.print("\n[bold]Witness replies[/bold]")

def clean_json():
    a = parse_worker_answer('{"answer":"Paris","reasoning":"capital","confidence":0.9,"caveats":[]}')
    assert a.answer == "Paris" and a.confidence == 0.9

def fenced_with_chatter():
    a = parse_worker_answer('Sure! Here you go:\n```json\n{"answer":"Paris","confidence":0.8}\n```\nHope that helps')
    assert a.answer == "Paris" and a.caveats == []

def percent_confidence():
    assert parse_worker_answer('{"answer":"x","confidence":"85%"}').confidence == 0.85
    assert parse_worker_answer('{"answer":"x","confidence":70}').confidence == 0.70

def caveat_as_string():
    assert parse_worker_answer('{"answer":"x","confidence":0.5,"caveats":"not sure"}').caveats == ["not sure"]

def line_breaks_inside_strings():
    a = parse_worker_answer('{"answer":"line one\nline two","confidence":0.5}')
    assert "line two" in a.answer

def answer_as_object():
    a = parse_worker_answer('{"answer":{"(1) Overconfidence":"dates","(2) Framing":"neutral"},"confidence":0.5}')
    assert "Overconfidence" in a.answer and "Framing" in a.answer

def unusual_characters():
    a = parse_worker_answer('{"answer":"non‑breaking hyphen and What’s","confidence":0.5}')
    assert "‑" in a.answer

check("clean JSON", clean_json)
check("raw line breaks inside text (Ministral)", line_breaks_inside_strings)
check("answer given as an object instead of text", answer_as_object)
check("unusual characters like ‑ survive", unusual_characters)
check("JSON in code fence with extra text", fenced_with_chatter)
check("confidence as 85% or 70", percent_confidence)
check("caveats as a single string", caveat_as_string)
check("rejects empty reply", lambda: expect_reject(""))
check("rejects reply with no JSON", lambda: expect_reject("The answer is Paris."))
check("rejects missing answer", lambda: expect_reject('{"reasoning":"x","confidence":0.5}'))
check("rejects missing confidence", lambda: expect_reject('{"answer":"Paris"}'))

# ---------------------------------------------------------------- judge replies

console.print("\n[bold]Judge replies[/bold]")
LETTERS = {"A", "B", "C"}

def normal_verdict():
    v = parse_verdict('{"final_answer":"Paris","chosen":["A","C"],"why":"w","confidence":0.9}', LETTERS)
    assert v.chosen == ["A", "C"] and not v.unresolved

def messy_letters():
    v = parse_verdict('{"final_answer":"Paris","chosen":["answer b","c"],"why":"w","confidence":0.9}', LETTERS)
    assert v.chosen == ["B", "C"]

def override_all():
    v = parse_verdict('{"final_answer":"Lisbon","chosen":[],"why":"all wrong","confidence":0.7}', LETTERS)
    assert v.chosen == []

def unresolved():
    v = parse_verdict('{"final_answer":"No consensus","chosen":[],"why":"w","confidence":0.3,"unresolved":true}', LETTERS)
    assert v.unresolved

check("normal verdict", normal_verdict)
check("letters written as 'answer b' or lowercase", messy_letters)
check("judge overrides every witness", override_all)
check("judge marks question unresolved", unresolved)
check("rejects a letter that wasn't on the table",
      lambda: expect_reject('{"final_answer":"x","chosen":["D"],"why":"w","confidence":0.5}', LETTERS))

# ---------------------------------------------------------------- prompts and logs

console.print("\n[bold]Prompts and run log[/bold]")

def judge_prompt_is_anonymous():
    a = parse_worker_answer('{"answer":"Paris","confidence":0.9}')
    text = judge_user("Capital of France?", [Exhibit(letter="A", answer=a), Exhibit(letter="B", answer=a)])
    assert "Answer A" in text and "Answer B" in text
    for name in ("groq", "gemini", "llama", "mistral", "nemotron"):
        assert name not in text.lower(), f"witness name '{name}' leaked into judge prompt"

def prompts_ask_for_json():
    assert '"confidence"' in WORKER_SYSTEM and '"unresolved"' in JUDGE_SYSTEM

def manual_prompt_has_everything():
    a = parse_worker_answer('{"answer":"Paris","confidence":0.9}')
    text = manual_judge_prompt("Q?", [Exhibit(letter="A", answer=a)])
    assert JUDGE_SYSTEM in text and "Answer A" in text

def run_log_saves():
    a = parse_worker_answer('{"answer":"Paris","confidence":0.9}')
    rec = RunRecord(question="Q?", seed=42, judge_model="claude-haiku-5-5", letter_map={"A": "groq"},
                    testimonies=[Testimony(witness="groq", family="openai", model="gpt-oss-120b", answer=a),
                                 Testimony(witness="mistral", family="mistral", model="m", error="rate-limited")])
    back = RunRecord.model_validate_json(rec.model_dump_json())
    assert back.testimonies[0].usable and not back.testimonies[1].usable

check("judge prompt hides witness names", judge_prompt_is_anonymous)
check("both prompts describe the JSON shape", prompts_ask_for_json)
check("--manual-judge prompt is complete", manual_prompt_has_everything)
check("run log saves and reloads", run_log_saves)

# ---------------------------------------------------------------- summary

console.print()
if failed:
    console.print(f"[red bold]{failed} failed[/red bold], {passed} passed")
    sys.exit(1)
console.print(f"[green bold]All {passed} tests passed.[/green bold]")
