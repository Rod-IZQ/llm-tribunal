"""
The judge: Claude reads the anonymized testimonies and returns a Verdict.

Exhibit order follows TESTIMONY_ORDER.md: usable answers are shuffled with a
logged seed, then lettered A, B, C... so position carries no meaning.

Try the full council from the AI_orchestrator folder:
    python -m core.judge "Is a tomato a fruit or a vegetable?"
    python -m core.judge --manual "..."     # no API call; writes a prompt to paste into claude.ai
    python -m core.judge --no-txt "..."     # skip the .txt report (written to runs/ by default)
    python -m core.judge --docx "..."       # also write a Word document (needs python-docx)
"""

from __future__ import annotations

import asyncio
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

from .prompts import JUDGE_SYSTEM, judge_user, manual_judge_prompt
from .schemas import Exhibit, ParseError, Testimony, Verdict, parse_verdict

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
JUDGE_MAX_TOKENS = 2000
REPAIR = ("Your reply was not valid JSON in the required shape, or it chose a letter that "
          "wasn't on the table. Reply again with ONLY the JSON object.")


def default_model() -> str:
    return os.getenv("JUDGE_MODEL", "claude-haiku-5-5").strip() or "claude-haiku-5-5"


def escalation_model() -> str:
    """Stronger judge for hard cases. Empty in .env = escalation off."""
    return os.getenv("JUDGE_ESCALATE_MODEL", "").strip()


# ---------------------------------------------------------------- exhibits

def make_exhibits(testimonies: list[Testimony], seed: Optional[int] = None
                  ) -> tuple[list[Exhibit], dict[str, str], int]:
    """Shuffle usable answers and letter them. Returns (exhibits, letter->witness, seed)."""
    usable = [t for t in testimonies if t.usable]
    seed = seed if seed is not None else random.randrange(1_000_000)
    random.Random(seed).shuffle(usable)
    exhibits, letter_map = [], {}
    for letter, t in zip(LETTERS, usable):
        exhibits.append(Exhibit(letter=letter, answer=t.answer))
        letter_map[letter] = t.witness
    return exhibits, letter_map, seed


# ---------------------------------------------------------------- the call

async def _ask_claude(model: str, question: str, exhibits: list[Exhibit]) -> Verdict:
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "").strip())
    valid = {e.letter for e in exhibits}
    messages = [{"role": "user", "content": judge_user(question, exhibits)}]

    for attempt in range(2):                       # first try + one repair
        msg = await client.messages.create(
            model=model, max_tokens=JUDGE_MAX_TOKENS, system=JUDGE_SYSTEM, messages=messages,
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        try:
            return parse_verdict(text, valid)
        except ParseError:
            if attempt == 1:
                raise
            messages += [{"role": "assistant", "content": text}, {"role": "user", "content": REPAIR}]
    raise RuntimeError("unreachable")


def _needs_escalation(v: Verdict) -> bool:
    return v.unresolved or v.confidence < 0.5


async def judge(question: str, testimonies: list[Testimony], seed: Optional[int] = None,
                model: Optional[str] = None) -> dict:
    """
    Returns a dict with: verdict (Verdict or None), letter_map, seed, judge_model,
    escalated (bool), error (str or None).
    """
    exhibits, letter_map, seed = make_exhibits(testimonies, seed)
    model = model or default_model()
    result = dict(verdict=None, letter_map=letter_map, seed=seed, judge_model=model,
                  escalated=False, error=None)

    if len(exhibits) < 2:
        result["error"] = f"only {len(exhibits)} usable answer(s); need at least 2"
        return result

    try:
        verdict = await _ask_claude(model, question, exhibits)
        stronger = escalation_model()
        if stronger and stronger != model and _needs_escalation(verdict):
            verdict = await _ask_claude(stronger, question, exhibits)
            result.update(judge_model=stronger, escalated=True)
        result["verdict"] = verdict
    except anthropic.AuthenticationError:
        result["error"] = "Anthropic key rejected"
    except anthropic.BadRequestError as exc:
        result["error"] = "no credits left" if "credit" in str(exc).lower() else f"bad request: {exc}"
    except ParseError as exc:
        result["error"] = f"judge reply unusable: {exc}"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {str(exc)[:150]}"
    return result


def write_manual_prompt(question: str, testimonies: list[Testimony], seed: Optional[int] = None) -> Path:
    """--manual-judge: save a ready-to-paste prompt instead of calling the API."""
    exhibits, letter_map, seed = make_exhibits(testimonies, seed)
    out_dir = ROOT / "runs"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"manual_{datetime.now():%Y%m%d_%H%M%S}.txt"
    key = ", ".join(f"{k}={v}" for k, v in letter_map.items())
    path.write_text(manual_judge_prompt(question, exhibits)
                    + f"\n\n<!-- seed={seed}; key (don't paste): {key} -->\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- try it from the terminal

async def _demo(question: str, manual: bool, save_txt: bool = True, docx: bool = False) -> None:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from .workers import ask, build_witnesses

    console = Console()
    witnesses = [w for w in build_witnesses().values() if w.configured]
    console.print(f"[bold]Questioning {len(witnesses)} witnesses…[/bold]")
    testimonies = await asyncio.gather(*(ask(w, question) for w in witnesses))

    if manual:
        path = write_manual_prompt(question, testimonies)
        console.print(f"\nPrompt saved to [bold]{path}[/bold]. Paste everything above the last line into claude.ai.")
        return

    console.print("[bold]The judge is deliberating…[/bold]\n")
    r = await judge(question, testimonies)
    witness_to_letter = {v: k for k, v in r["letter_map"].items()}
    chosen = set(r["verdict"].chosen) if r["verdict"] else set()

    table = Table(title="Testimony")
    for col in ("Exhibit", "Witness", "Time", "Conf.", "Answer"):
        table.add_column(col)
    for t in sorted(testimonies, key=lambda t: witness_to_letter.get(t.witness, "~")):
        letter = witness_to_letter.get(t.witness, "–")
        mark = f"[green]{letter} ✓[/green]" if letter in chosen else letter
        took = f"{t.latency_ms / 1000:.1f}s" if t.latency_ms else "-"
        if t.answer:
            table.add_row(mark, t.witness, took, f"{t.answer.confidence:.0%}", t.answer.answer[:90])
        else:
            table.add_row("–", t.witness, took, "", f"[red]{t.error}[/red]")
    console.print(table)

    if save_txt:
        try:
            from .report_txt import write_txt
            path = write_txt(question, testimonies, r)
            console.print(f"\nText file: [bold]{path.relative_to(ROOT)}[/bold]")
        except Exception as exc:
            console.print(f"\n[red]Could not write .txt: {exc}[/red]")
    if docx:
        try:
            from .report_docx import write_docx
            path = write_docx(question, testimonies, r)
            console.print(f"Word document: [bold]{path.relative_to(ROOT)}[/bold]")
        except ImportError:
            console.print("[yellow]No .docx: run  pip install python-docx[/yellow]")
        except Exception as exc:
            console.print(f"[red]Could not write .docx: {exc}[/red]")

    if r["error"]:
        console.print(Panel(f"[red]{r['error']}[/red]", title="No verdict", border_style="red"))
        return
    v = r["verdict"]
    status = "[yellow]UNRESOLVED[/yellow]  " if v.unresolved else ""
    body = (f"{status}[bold]{v.final_answer}[/bold]\n\n[bold]Why:[/bold] {v.why}\n\n"
            f"[bold]Relied on:[/bold] {', '.join(v.chosen) or 'none (judge overrode all)'}"
            f"    [bold]Confidence:[/bold] {v.confidence:.0%}")
    if v.disagreements:
        body += "\n\n[bold]Disagreements:[/bold]\n" + "\n".join(f"  • {d}" for d in v.disagreements)
    title = f"Verdict · {r['judge_model']}" + (" (escalated)" if r["escalated"] else "") + f" · seed {r['seed']}"
    console.print(Panel(body, title=title, border_style="cyan"))


if __name__ == "__main__":
    from .workers import safe_console_output
    safe_console_output()
    args = sys.argv[1:]
    manual = "--manual" in args
    save_txt = "--no-txt" not in args
    docx = "--docx" in args
    args = [a for a in args if a not in ("--manual", "--no-txt", "--docx")]
    q = " ".join(args) or "Is a tomato a fruit or a vegetable?"
    asyncio.run(_demo(q, manual, save_txt, docx))
