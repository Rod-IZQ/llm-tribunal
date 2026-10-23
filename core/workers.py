"""
Questioning the witnesses.

Each witness is defined from .env (same names probe.py uses). ask() sends one
question to one witness and always returns a Testimony: with an answer if it
worked, or with a plain-language error if it didn't. It never raises, so one
bad witness can't crash a run.

Try a single witness from the AI_orchestrator folder:
    python -m core.workers groq "Why is the sky blue?"
    python -m core.workers list
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
import litellm
from dotenv import load_dotenv
from tenacity import (
    AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential,
)

from .prompts import REPAIR_USER, WORKER_SYSTEM, worker_user
from .schemas import ParseError, Testimony, parse_worker_answer

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)
litellm.suppress_debug_info = True

MAX_TOKENS = 1500
RETRIES = 2            # extra attempts on 429 / timeout / provider busy


# ---------------------------------------------------------------- witness registry

@dataclass(frozen=True)
class Witness:
    id: str                       # short name used in logs: groq, local...
    family: str                   # model family, for the diversity rule
    model: str                    # litellm model string (or plain name for ollama_cloud)
    key_env: Optional[str]        # .env variable holding the key (None = no key needed)
    api_base: Optional[str] = None
    timeout: float = 30.0

    @property
    def key(self) -> str:
        return os.getenv(self.key_env, "").strip() if self.key_env else ""

    @property
    def configured(self) -> bool:
        return self.key_env is None or bool(self.key)


FAMILY_HINTS = {
    "llama": "meta", "gemma": "google", "gemini": "google", "qwen": "qwen",
    "deepseek": "deepseek", "nemotron": "nvidia", "mistral": "mistral",
    "ministral": "mistral", "gpt-oss": "openai", "kimi": "moonshot", "glm": "zhipu",
}


def guess_family(model: str) -> str:
    low = model.lower()
    for hint, family in FAMILY_HINTS.items():
        if hint in low:
            return family
    return low.split("/")[0] if "/" in low else "unknown"


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def build_witnesses() -> dict[str, Witness]:
    """All witnesses, read fresh from .env."""
    gemini = _env("GEMINI_MODEL", "gemini-3.5-flash-lite").removeprefix("gemini/")
    mistral = _env("MISTRAL_MODEL", "ministral-8b-latest").removeprefix("mistral/")
    openrouter = _env("OPENROUTER_MODEL", "recursive/flint-v1.0:free").removeprefix("openrouter/")
    local = _env("OLLAMA_MODEL", "llama3.1:8b")
    cloud = _env("OLLAMA_CLOUD_MODEL", "nemotron-3-nano:30b").removesuffix("-cloud")

    witnesses = [
        Witness("gemini", "google", f"gemini/{gemini}", "GEMINI_API_KEY"),
        Witness("groq", "openai", "groq/openai/gpt-oss-120b", "GROQ_API_KEY"),
        Witness("mistral", "mistral", f"mistral/{mistral}", "MISTRAL_API_KEY"),
        Witness("openrouter", guess_family(openrouter), f"openrouter/{openrouter}", "OPENROUTER_API_KEY"),
        Witness("local", guess_family(local), f"ollama_chat/{local}", None,
                api_base=_env("OLLAMA_API_BASE", "http://localhost:11434"), timeout=120.0),
        Witness("ollama_cloud", guess_family(cloud), cloud, "OLLAMA_API_KEY",
                api_base="https://ollama.com", timeout=120.0),
    ]
    return {w.id: w for w in witnesses}


# ---------------------------------------------------------------- errors

class ProviderBusy(Exception):
    """Retryable: rate limit, timeout, overloaded."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, ProviderBusy):
        return True
    name, text = type(exc).__name__.lower(), str(exc).lower()
    return any(s in name for s in ("ratelimit", "timeout", "serviceunavailable", "internalserver")) \
        or any(s in text for s in (" 429", " 503", "overloaded"))


def explain(exc: BaseException) -> str:
    name, text = type(exc).__name__.lower(), str(exc).lower()
    if "auth" in name or "401" in text:
        return "key rejected"
    if "ratelimit" in name or "429" in text:
        return "rate-limited"
    if "notfound" in name or "404" in text:
        return "model not found"
    if "timeout" in name:
        return "timed out"
    if "serviceunavailable" in name or "503" in text or "overloaded" in text:
        return "provider busy"
    if "connect" in name:
        return "could not connect"
    return f"{type(exc).__name__}: {str(exc)[:120]}"


# ---------------------------------------------------------------- raw calls

async def _call_litellm(w: Witness, messages: list[dict]) -> str:
    kwargs = dict(model=w.model, messages=messages, max_tokens=MAX_TOKENS, timeout=w.timeout)
    if w.key:
        kwargs["api_key"] = w.key
    if w.api_base:
        kwargs["api_base"] = w.api_base
    resp = await litellm.acompletion(**kwargs)
    return resp.choices[0].message.content or ""


async def _call_ollama_cloud(w: Witness, messages: list[dict]) -> str:
    async with httpx.AsyncClient(timeout=w.timeout) as client:
        r = await client.post(
            f"{w.api_base}/api/chat",
            headers={"Authorization": f"Bearer {w.key}"},
            json={"model": w.model, "messages": messages, "stream": False},
        )
    if r.status_code in (429, 503):
        raise ProviderBusy(f"HTTP {r.status_code}")
    if r.status_code == 401:
        raise PermissionError("401 key rejected")
    if r.status_code == 404:
        raise LookupError("404 model not found")
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "") or ""


async def _call(w: Witness, messages: list[dict]) -> str:
    """One call with retry/backoff on busy errors."""
    call = _call_ollama_cloud if w.id == "ollama_cloud" else _call_litellm
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(RETRIES + 1),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    ):
        with attempt:
            return await call(w, messages)
    return ""  # unreachable


# ---------------------------------------------------------------- public API

def safe_console_output() -> None:
    """Windows consoles can't print some characters (e.g. U+2011). Never crash on them."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


async def chat(w: Witness, messages: list[dict]) -> str:
    """Raw call with retries, for tools that need their own prompt (e.g. self_report)."""
    return await _call(w, messages)


async def ask(w: Witness, question: str) -> Testimony:
    """Question one witness. Always returns a Testimony; never raises."""
    base = dict(witness=w.id, family=w.family, model=w.model)
    if not w.configured:
        return Testimony(**base, error=f"{w.key_env} not set in .env")

    messages = [
        {"role": "system", "content": WORKER_SYSTEM},
        {"role": "user", "content": worker_user(question)},
    ]
    start = time.perf_counter()
    try:
        raw = await _call(w, messages)
        try:
            answer = parse_worker_answer(raw)
        except ParseError:
            # one repair attempt: show it its own reply and ask for valid JSON
            messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": REPAIR_USER}]
            raw = await _call(w, messages)
            answer = parse_worker_answer(raw)
    except ParseError as exc:
        return Testimony(**base, raw_text=raw, latency_ms=_ms(start), error=f"unusable reply ({exc})")
    except Exception as exc:
        return Testimony(**base, latency_ms=_ms(start), error=explain(exc))

    return Testimony(**base, answer=answer, raw_text=raw, latency_ms=_ms(start))


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


# ---------------------------------------------------------------- try it from the terminal

def _print_testimony(t: Testimony) -> None:
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    took = f"{t.latency_ms / 1000:.1f} s" if t.latency_ms else "-"
    if t.answer is None:
        console.print(Panel(f"[red]{t.error}[/red]\n\n[dim]{t.raw_text[:500]}[/dim]",
                            title=f"{t.witness} ({t.model}) failed after {took}", border_style="red"))
        return
    a = t.answer
    caveats = "\n".join(f"  • {c}" for c in a.caveats) or "  (none)"
    body = (f"[bold]Answer:[/bold] {a.answer}\n\n[bold]Reasoning:[/bold] {a.reasoning}\n\n"
            f"[bold]Confidence:[/bold] {a.confidence:.0%}\n\n[bold]Caveats:[/bold]\n{caveats}")
    console.print(Panel(body, title=f"{t.witness} · {t.family} · {t.model} · {took}", border_style="green"))


def _main() -> None:
    safe_console_output()
    witnesses = build_witnesses()
    args = sys.argv[1:]
    if not args or args[0] == "list":
        for w in witnesses.values():
            state = "ready" if w.configured else f"missing {w.key_env}"
            print(f"{w.id:13} {w.family:10} {w.model:45} {state}")
        if not args:
            print('\nUsage: python -m core.workers <witness> "your question"')
        return
    name, question = args[0], " ".join(args[1:]) or "Why is the sky blue?"
    if name not in witnesses:
        sys.exit(f"Unknown witness '{name}'. Choose from: {', '.join(witnesses)}")
    _print_testimony(asyncio.run(ask(witnesses[name], question)))


if __name__ == "__main__":
    _main()
