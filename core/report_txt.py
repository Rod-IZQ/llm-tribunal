"""
Export one council run to a plain-text file (.txt).

    from core.report_txt import write_txt
    path = write_txt(question, testimonies, result)   # result = dict returned by judge()

Labels are in Spanish when the question is in Spanish, otherwise English.
The file reveals which witness wrote each exhibit (the judge never saw this).
Saved as UTF-8 with BOM so Notepad shows accents correctly.
"""

from __future__ import annotations

import re
import textwrap
from datetime import datetime
from pathlib import Path

from .schemas import Testimony

ROOT = Path(__file__).resolve().parent.parent
WIDTH = 90

LABELS = {
    "es": dict(
        title="DICTAMEN DEL CONSEJO", date="Fecha", judge="Juez", seed="Semilla",
        witnesses="Testigos", question="PREGUNTA", verdict="VEREDICTO", conf="Confianza del juez",
        unresolved="*** SIN RESOLVER: la evidencia no permite decidir ***", why="FUNDAMENTO DEL VEREDICTO",
        relied="Se apoyó en", none_relied="ninguno (el juez dio su propia respuesta)",
        escalated="escalado", disagreements="DESACUERDOS ENTRE TESTIGOS",
        table="RESUMEN DE TESTIMONIOS", exhibit="Prueba", witness="Testigo", time="Tiempo",
        wconf="Confianza", answer="Respuesta", detail="DETALLE DE CADA TESTIMONIO",
        reasoning="Razonamiento", caveats="Salvedades", none="(ninguna)", failed="TESTIGOS SIN RESPUESTA",
        no_verdict="No hubo veredicto", chosen="ELEGIDA POR EL JUEZ",
        disclaimer=("Documento generado automáticamente por modelos de IA. No constituye asesoría legal, "
                    "fiscal ni aduanera. Verifique la información y el fundamento legal con un profesional "
                    "antes de actuar."),
        stem="dictamen",
    ),
    "en": dict(
        title="COUNCIL RULING", date="Date", judge="Judge", seed="Seed",
        witnesses="Witnesses", question="QUESTION", verdict="VERDICT", conf="Judge confidence",
        unresolved="*** UNRESOLVED: the evidence does not settle it ***", why="REASONING BEHIND THE VERDICT",
        relied="Relied on", none_relied="none (the judge gave its own answer)",
        escalated="escalated", disagreements="DISAGREEMENTS BETWEEN WITNESSES",
        table="TESTIMONY SUMMARY", exhibit="Exhibit", witness="Witness", time="Time",
        wconf="Confidence", answer="Answer", detail="EACH TESTIMONY IN FULL",
        reasoning="Reasoning", caveats="Caveats", none="(none)", failed="WITNESSES WITH NO ANSWER",
        no_verdict="No verdict", chosen="CHOSEN BY THE JUDGE",
        disclaimer=("Generated automatically by AI models. Not legal, tax or customs advice. "
                    "Verify the information and any legal basis with a professional before acting."),
        stem="ruling",
    ),
}


def detect_lang(text: str) -> str:
    t = f" {text.lower()} "
    if re.search(r"[ñ¿¡áéíóú]", t):
        return "es"
    hits = sum(w in t for w in (" el ", " la ", " de ", " que ", " puedo ", " es ", " una ", " los ", " para "))
    return "es" if hits >= 2 else "en"


def _wrap(text: str, indent: str = "") -> list[str]:
    """Wrap model text, keeping its own line breaks as paragraphs."""
    out = []
    for chunk in str(text).split("\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        out += textwrap.wrap(chunk, WIDTH, initial_indent=indent, subsequent_indent=indent) or [indent]
    return out


def _bullets(items: list[str]) -> list[str]:
    out = []
    for item in items:
        lines = textwrap.wrap(str(item).replace("\n", " "), WIDTH - 4)
        out += [f"  - {lines[0]}"] + [f"    {l}" for l in lines[1:]]
    return out


def _section(title: str, char: str = "=") -> list[str]:
    return ["", title, char * len(title)]


def build_text(question: str, testimonies: list[Testimony], result: dict) -> tuple[str, str]:
    L = LABELS[detect_lang(question)]
    verdict = result.get("verdict")
    letter_map: dict[str, str] = result.get("letter_map", {})
    witness_to_letter = {v: k for k, v in letter_map.items()}
    chosen = set(verdict.chosen) if verdict else set()

    judge_line = result.get("judge_model", "")
    if result.get("escalated"):
        judge_line += f" ({L['escalated']})"

    out = ["=" * WIDTH, L["title"].center(WIDTH), "=" * WIDTH,
           f"{L['date']}: {datetime.now():%Y-%m-%d %H:%M}",
           f"{L['judge']}: {judge_line}",
           f"{L['seed']}: {result.get('seed', '-')}",
           f"{L['witnesses']}: {len(testimonies)}"]

    out += _section(L["question"])
    out += _wrap(question)

    out += _section(L["verdict"])
    if verdict is None:
        out.append(f"{L['no_verdict']}: {result.get('error', '')}")
    else:
        if verdict.unresolved:
            out += [L["unresolved"], ""]
        out += _wrap(verdict.final_answer, indent="  ")
        relied = ", ".join(f"{c} ({letter_map.get(c, '?')})" for c in verdict.chosen) or L["none_relied"]
        out += ["", f"{L['conf']}: {verdict.confidence:.0%}", f"{L['relied']}: {relied}"]

        out += _section(L["why"], "-")
        out += _wrap(verdict.why)
        if verdict.disagreements:
            out += _section(L["disagreements"], "-")
            out += _bullets(verdict.disagreements)

    usable = sorted([t for t in testimonies if t.witness in witness_to_letter],
                    key=lambda t: witness_to_letter[t.witness])

    out += _section(L["table"])
    head = f"{L['exhibit']:<8} {L['witness']:<14} {L['time']:>7} {L['wconf']:>10}  {L['answer']}"
    out += [head, "-" * WIDTH]
    for t in usable:
        letter = witness_to_letter[t.witness] + (" *" if witness_to_letter[t.witness] in chosen else "")
        took = f"{t.latency_ms / 1000:.1f} s" if t.latency_ms else "-"
        prefix = f"{letter:<8} {t.witness:<14} {took:>7} {t.answer.confidence:>10.0%}  "
        short = t.answer.answer.replace("\n", " ")
        lines = textwrap.wrap(short, WIDTH - len(prefix)) or [""]
        if len(lines) > 2:
            lines = lines[:2]
            lines[1] = lines[1][: WIDTH - len(prefix) - 1] + "…"
        out.append(prefix + lines[0])
        out += [" " * len(prefix) + l for l in lines[1:]]
    if chosen:
        out.append(f"(* = {L['chosen'].lower()})")

    out += _section(L["detail"])
    for t in usable:
        letter = witness_to_letter[t.witness]
        title = f"{L['exhibit']} {letter} · {t.witness} ({t.model})"
        if letter in chosen:
            title += f"  [{L['chosen']}]"
        out += ["", title, "-" * min(len(title), WIDTH),
                f"{L['wconf']}: {t.answer.confidence:.0%}", "", f"{L['answer']}:"]
        out += _wrap(t.answer.answer, indent="  ")
        if t.answer.reasoning:
            out += ["", f"{L['reasoning']}:"]
            out += _wrap(t.answer.reasoning, indent="  ")
        out += ["", f"{L['caveats']}:"]
        out += _bullets(t.answer.caveats) if t.answer.caveats else [f"  {L['none']}"]

    failed = [t for t in testimonies if t.witness not in witness_to_letter]
    if failed:
        out += _section(L["failed"], "-")
        out += _bullets([f"{t.witness} ({t.model}): {t.error or 'late'}" for t in failed])

    out += ["", "=" * WIDTH]
    out += _wrap(L["disclaimer"])
    return "\n".join(out) + "\n", L["stem"]


def write_txt(question: str, testimonies: list[Testimony], result: dict,
              out_dir: Path | None = None) -> Path:
    text, stem = build_text(question, testimonies, result)
    out_dir = out_dir or (ROOT / "runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}.txt"
    path.write_text(text, encoding="utf-8-sig")
    return path
