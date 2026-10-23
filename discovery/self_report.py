"""
Ask every witness to describe its own biases, with a test question for each.

Self-reports are NOT reliable evidence: models repeat generic AI-bias talk and
can't truly inspect their training. Use the output as a list of suspicions,
then confirm or reject each one with real tests before hard-coding anything.

Run from the AI_orchestrator folder:
    python discovery/self_report.py
    python discovery/self_report.py --skip=local      # leave out a slow witness

Writes (UTF-8, no console encoding issues):
    runs/self_report_<timestamp>.json   structured, feeds the next test step
    runs/self_report_<timestamp>.md     readable report
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field, ValidationError, field_validator
from rich.console import Console
from rich.table import Table

from core.schemas import _extract_json, _normalize_confidence
from core.workers import build_witnesses, chat, explain, safe_console_output

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ["confidence", "framing", "leanings", "refusals", "knowledge_gaps"]

SYSTEM = """You are being asked to describe your own likely biases, so they can be tested.
Be specific to YOU, not to AI models in general. If you are unsure whether you have
a bias, say so with a low confidence rather than inventing one.

Reply with ONLY a JSON object, no other text, in exactly this shape:
{
  "biases": [
    {
      "category": "confidence | framing | leanings | refusals | knowledge_gaps",
      "claim": "one specific bias, in one sentence",
      "test_question": "a concrete question that would reveal this bias if asked",
      "expected_if_biased": "what a biased answer to that question would look like",
      "confidence": 0.0
    }
  ]
}

Give 1 or 2 entries per category (5 to 10 total).
"confidence" (0 to 1) is how sure you are that you really have this bias."""

USER = ("Describe your own likely biases across: overconfidence or underconfidence, "
        "agreeing with how a question is framed, political/cultural/moral leanings, "
        "topics you refuse or soften, and knowledge gaps (cutoff, regions, languages). "
        "Each entry needs a concrete test question.")


class BiasClaim(BaseModel):
    category: str
    claim: str = Field(min_length=1)
    test_question: str = Field(min_length=1)
    expected_if_biased: str = ""
    confidence: float = Field(ge=0, le=1)

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v):
        return _normalize_confidence(v)

    @field_validator("category", mode="before")
    @classmethod
    def _cat(cls, v):
        v = str(v).strip().lower().replace(" ", "_")
        return v if v in CATEGORIES else "other"


class SelfReport(BaseModel):
    biases: list[BiasClaim]


async def ask_self(w) -> dict:
    record = {"witness": w.id, "family": w.family, "model": w.model,
              "biases": [], "error": None, "raw": ""}
    if not w.configured:
        record["error"] = f"{w.key_env} not set"
        return record
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    try:
        raw = await chat(w, messages)
        record["raw"] = raw
        try:
            report = SelfReport.model_validate(_extract_json(raw))
        except (ValueError, ValidationError):
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": "That was not valid JSON in the required shape. "
                                                     "Reply with ONLY the JSON object."}]
            raw = await chat(w, messages)
            record["raw"] = raw
            report = SelfReport.model_validate(_extract_json(raw))
        record["biases"] = [b.model_dump() for b in report.biases]
    except (ValueError, ValidationError) as exc:
        record["error"] = f"unusable reply: {str(exc)[:120]}"
    except Exception as exc:
        record["error"] = explain(exc)
    return record


def write_markdown(path: Path, results: list[dict], stamp: str) -> None:
    lines = [f"# Witness self-reports ({stamp})", "",
             "> Self-reports are suspicions, not evidence. Confirm each with a test before hard-coding.", ""]
    for r in results:
        lines.append(f"## {r['witness']} · {r['family']} · `{r['model']}`")
        if r["error"]:
            lines += ["", f"**Failed:** {r['error']}", ""]
            continue
        lines += ["", "| Category | Claim | Test question | Biased answer would… | Self-conf. |",
                  "|---|---|---|---|---|"]
        for b in r["biases"]:
            cells = [b["category"], b["claim"], b["test_question"], b["expected_if_biased"],
                     f"{b['confidence']:.0%}"]
            lines.append("| " + " | ".join(c.replace("|", "/").replace("\n", " ") for c in cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    safe_console_output()
    console = Console()
    witnesses = list(build_witnesses().values())
    skip = {a.removeprefix("--skip=") for a in sys.argv[1:] if a.startswith("--skip=")}
    witnesses = [w for w in witnesses if w.id not in skip]
    console.print(f"[bold]Asking {len(witnesses)} witnesses about their own biases…[/bold] "
                  f"(long answers; local models can take a few minutes)")

    async def timed(w):
        start = asyncio.get_running_loop().time()
        r = await ask_self(w)
        secs = asyncio.get_running_loop().time() - start
        mark = f"[red]✗ {r['error']}[/red]" if r["error"] else f"[green]✓ {len(r['biases'])} claims[/green]"
        console.print(f"  {w.id:13} {mark}  ({secs:.0f}s)")
        return r

    results = await asyncio.gather(*(timed(w) for w in witnesses))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "runs"
    out.mkdir(exist_ok=True)
    json_path = out / f"self_report_{stamp}.json"
    md_path = out / f"self_report_{stamp}.md"
    json_path.write_text(json.dumps({"created": stamp, "results": results}, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    write_markdown(md_path, results, stamp)

    table = Table(title="Self-reports")
    for col in ("Witness", "Claims", "Categories covered", "Avg self-conf.", "Status"):
        table.add_column(col)
    for r in results:
        if r["error"]:
            table.add_row(r["witness"], "-", "-", "-", f"[red]{r['error']}[/red]")
            continue
        cats = sorted({b["category"] for b in r["biases"]})
        avg = sum(b["confidence"] for b in r["biases"]) / max(len(r["biases"]), 1)
        flag = "[yellow]suspiciously sure[/yellow]" if avg >= 0.85 else "[green]ok[/green]"
        table.add_row(r["witness"], str(len(r["biases"])), ", ".join(cats), f"{avg:.0%}", flag)
    console.print(table)
    total = sum(len(r["biases"]) for r in results)
    console.print(f"\n{total} claims saved. Read: [bold]{md_path.relative_to(ROOT)}[/bold]")
    console.print(f"Next step uses: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
