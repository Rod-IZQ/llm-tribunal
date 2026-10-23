"""
Connection check for every council member.

Run from the AI_orchestrator folder (with .venv active):
    python discovery/probe.py
    python discovery/probe.py --list-free   # also list Gemini, OpenRouter free + Ollama cloud models

Optional .env lines:
    JUDGE_MODEL=claude-haiku-5-5
    OPENROUTER_MODEL=<a model id ending in :free>
    OLLAMA_MODEL=llama3.1:8b
    OLLAMA_API_KEY=<from ollama.com/settings/keys>
    OLLAMA_CLOUD_MODEL=<a cloud model name>

For each service it checks that the key exists in .env, sends one tiny
test message, and reports whether the key works, how long it took, and
what went wrong if it didn't. Free workers cost nothing; the Anthropic
check uses Haiku with a 10-token reply (a fraction of a cent).
"""

import asyncio
import os
import sys
import time
from pathlib import Path

import httpx
import litellm
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)  # .env always wins over old Windows/terminal variables

litellm.suppress_debug_info = True
console = Console()

TEST_PROMPT = "Reply with just the word OK."
TIMEOUT = 30

# name, env var, litellm model string
LITELLM_WORKERS = [
    ("gemini", "GEMINI_API_KEY", "gemini/" + os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip().removeprefix("gemini/")),
    ("groq", "GROQ_API_KEY", "groq/openai/gpt-oss-120b"),
    ("mistral", "MISTRAL_API_KEY", "mistral/" + os.getenv("MISTRAL_MODEL", "mistral-small-latest").strip().removeprefix("mistral/")),
]

OPENROUTER_PREFERRED = ("deepseek", "qwen", "moonshot", "kimi")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-haiku-5-5").strip()


def mask(key: str) -> str:
    return f"…{key[-4:]}" if len(key) > 8 else "set"


def explain(exc: Exception) -> str:
    """Turn a provider error into a plain-language reason."""
    name = type(exc).__name__
    text = str(exc).lower()
    if "auth" in name.lower() or "401" in text or "invalid api key" in text or "api key not valid" in text:
        return "Key rejected: check it was copied fully"
    if "ratelimit" in name.lower() or "429" in text:
        return "Key works, but rate-limited right now"
    if "notfound" in name.lower() or "404" in text or "model" in text and "not" in text and "found" in text:
        return "Key works, but model name not found"
    if "permission" in name.lower() or "403" in text:
        return "Key lacks permission (account not verified?)"
    if "serviceunavailable" in name.lower() or "503" in text or "overloaded" in text or "internalserver" in name.lower():
        return "Provider busy/down right now (not your key) - try again in a few minutes"
    if "timeout" in name.lower():
        return f"No reply within {TIMEOUT}s"
    if "connect" in name.lower():
        return "Could not reach the service (network?)"
    return f"{name}: {str(exc)[:80]}"


async def check_litellm(name: str, env: str, model: str) -> dict:
    key = os.getenv(env, "").strip()
    if not key:
        return {"name": name, "status": "missing", "detail": f"{env} is empty in .env"}
    start = time.perf_counter()
    try:
        resp = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=200,
            timeout=TIMEOUT,
            api_key=key,
        )
        ms = (time.perf_counter() - start) * 1000
        reply = (resp.choices[0].message.content or "").strip()[:30]
        return {"name": name, "status": "ok", "model": model, "ms": ms,
                "detail": f"{mask(key)} replied: {reply or '(empty)'}"}
    except Exception as exc:
        return {"name": name, "status": "fail", "model": model, "detail": explain(exc)}


async def list_gemini_models() -> None:
    """With --list-free: print the Gemini chat models this key can use."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if "--list-free" not in sys.argv or not key:
        return
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": key, "pageSize": 200},
            )
        models = [
            m["name"].removeprefix("models/")
            for m in r.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
            and "flash" in m["name"]
        ]
    except Exception as exc:
        console.print(f"[red]Could not list Gemini models: {explain(exc)}[/red]")
        return
    console.print(f"\n[bold]Gemini Flash models for your key ({len(models)}):[/bold]")
    for m in sorted(models):
        console.print(f"  {m}")
    console.print("Pin one in .env with:  GEMINI_MODEL=<model name>\n")


async def check_openrouter() -> dict:
    name, env = "openrouter", "OPENROUTER_API_KEY"
    key = os.getenv(env, "").strip()
    if not key:
        return {"name": name, "status": "missing", "detail": f"{env} is empty in .env"}
    headers = {"Authorization": f"Bearer {key}"}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            auth = await client.get("https://openrouter.ai/api/v1/key", headers=headers)
            if auth.status_code == 401:
                return {"name": name, "status": "fail", "detail": "Key rejected: check it was copied fully"}
            models = await client.get("https://openrouter.ai/api/v1/models")
            free = [m["id"] for m in models.json().get("data", []) if m["id"].endswith(":free")]
    except Exception as exc:
        return {"name": name, "status": "fail", "detail": explain(exc)}

    if "--list-free" in sys.argv:
        console.print(f"\n[bold]OpenRouter free models right now ({len(free)}):[/bold]")
        for m in sorted(free):
            console.print(f"  {m}")
        console.print("Pin one in .env with:  OPENROUTER_MODEL=<model id>\n")

    pinned = os.getenv("OPENROUTER_MODEL", "").strip().removeprefix("openrouter/")
    if pinned:
        pick, note = pinned, "pinned in .env"
    else:
        preferred = [m for m in free if any(p in m.lower() for p in OPENROUTER_PREFERRED)]
        pick = (preferred or free)[0]
        note = "auto-picked" if preferred else "no DeepSeek/Qwen/Kimi free today; run with --list-free and pin one"

    result = await check_litellm(name, env, f"openrouter/{pick}")
    result["detail"] += f"  ({note}; {len(free)} free models)"
    return result


async def check_ollama() -> dict:
    base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{base}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return {"name": "local", "status": "missing", "detail": "Ollama not running (optional worker)"}
    if not models:
        return {"name": "local", "status": "fail", "detail": "Ollama running, but no model pulled (try: ollama pull llama3.1:8b)"}
    wanted = os.getenv("OLLAMA_MODEL", "").strip()
    if wanted and not any(m == wanted or m.split(":")[0] == wanted for m in models):
        return {"name": "local", "status": "fail", "model": wanted,
                "detail": f"OLLAMA_MODEL not pulled yet (try: ollama pull {wanted})"}
    result = await check_litellm_local(base, wanted or models[0])
    result["detail"] += f"  (installed: {', '.join(models[:3])})"
    return result


async def check_litellm_local(base: str, model: str) -> dict:
    start = time.perf_counter()
    try:
        resp = await litellm.acompletion(
            model=f"ollama_chat/{model}",
            messages=[{"role": "user", "content": TEST_PROMPT}],
            api_base=base,
            timeout=120,  # first load into memory can be slow
        )
        ms = (time.perf_counter() - start) * 1000
        reply = (resp.choices[0].message.content or "").strip()[:30]
        return {"name": "local", "status": "ok", "model": model, "ms": ms, "detail": f"replied: {reply or '(empty)'}"}
    except Exception as exc:
        return {"name": "local", "status": "fail", "model": model, "detail": explain(exc)}


async def check_ollama_cloud() -> dict:
    """Ollama's hosted models, called directly with an API key (no local app needed)."""
    name, env = "ollama_cloud", "OLLAMA_API_KEY"
    key = os.getenv(env, "").strip()
    if not key:
        return {"name": name, "status": "missing", "detail": f"{env} is empty in .env (optional worker)"}
    base = "https://ollama.com"
    headers = {"Authorization": f"Bearer {key}"}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            tags = await client.get(f"{base}/api/tags", headers=headers)
            if tags.status_code in (401, 403):
                return {"name": name, "status": "fail", "detail": "Key rejected: check it was copied fully"}
            models = [m["name"] for m in tags.json().get("models", [])]

            if "--list-free" in sys.argv and models:
                console.print(f"\n[bold]Ollama cloud models ({len(models)}):[/bold]")
                for m in sorted(models):
                    console.print(f"  {m}")
                console.print("Pin one in .env with:  OLLAMA_CLOUD_MODEL=<model name>\n")

            wanted = os.getenv("OLLAMA_CLOUD_MODEL", "").strip().removesuffix("-cloud")
            if not wanted:
                if not models:
                    return {"name": name, "status": "fail", "detail": "Key works, but no models listed; set OLLAMA_CLOUD_MODEL"}
                wanted = models[0]
                note = "auto-picked; run with --list-free and pin one"
            else:
                note = "pinned in .env"

            start = time.perf_counter()
            r = await client.post(
                f"{base}/api/chat",
                headers=headers,
                json={"model": wanted, "messages": [{"role": "user", "content": TEST_PROMPT}], "stream": False},
                timeout=60,
            )
            ms = (time.perf_counter() - start) * 1000
    except Exception as exc:
        return {"name": name, "status": "fail", "detail": explain(exc)}

    if r.status_code in (401, 403):
        return {"name": name, "status": "fail", "model": wanted,
                "detail": "Key rejected: re-copy it from ollama.com/settings/keys (no spaces or quotes)"}
    if r.status_code == 404:
        return {"name": name, "status": "fail", "model": wanted, "detail": "Key works, but model name not found"}
    if r.status_code == 429:
        return {"name": name, "status": "fail", "model": wanted, "detail": "Key works, but usage limit reached"}
    if r.status_code != 200:
        return {"name": name, "status": "fail", "model": wanted, "detail": f"HTTP {r.status_code}: {r.text[:80]}"}
    reply = (r.json().get("message", {}).get("content") or "").strip()[:30]
    return {"name": name, "status": "ok", "model": wanted, "ms": ms,
            "detail": f"{mask(key)} replied: {reply or '(empty)'}  ({note})"}


async def check_anthropic() -> dict:
    env = "ANTHROPIC_API_KEY"
    key = os.getenv(env, "").strip()
    if not key:
        return {"name": "judge (claude)", "status": "missing", "detail": f"{env} is empty in .env"}
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=key, timeout=TIMEOUT)
    start = time.perf_counter()
    try:
        msg = await client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": TEST_PROMPT}],
        )
        ms = (time.perf_counter() - start) * 1000
        reply = "".join(b.text for b in msg.content if b.type == "text").strip()[:30]
        return {"name": "judge (claude)", "status": "ok", "model": JUDGE_MODEL, "ms": ms,
                "detail": f"{mask(key)} replied: {reply}"}
    except anthropic.AuthenticationError:
        detail = "Key rejected: check it was copied fully"
    except anthropic.PermissionDeniedError:
        detail = "Key lacks permission"
    except anthropic.BadRequestError as exc:
        detail = "No credits on the account" if "credit" in str(exc).lower() else f"Bad request: {str(exc)[:80]}"
    except Exception as exc:
        detail = explain(exc)
    return {"name": "judge (claude)", "status": "fail", "model": JUDGE_MODEL, "detail": detail}


async def main() -> None:
    if not (ROOT / ".env").exists():
        console.print(f"[red]No .env file found in {ROOT}[/red]")
        console.print("Create it with:  copy .env.example .env")
        return

    await list_gemini_models()
    console.print("[bold]Checking connections…[/bold] (each service gets one tiny test message)\n")
    checks = [check_litellm(*w) for w in LITELLM_WORKERS]
    checks += [check_openrouter(), check_ollama(), check_ollama_cloud(), check_anthropic()]
    results = await asyncio.gather(*checks)

    icons = {"ok": "[green]✓ connected[/green]", "fail": "[red]✗ failed[/red]", "missing": "[dim]– not set[/dim]"}
    table = Table(show_lines=False)
    table.add_column("Agent", style="bold")
    table.add_column("Status")
    table.add_column("Time", justify="right")
    table.add_column("Model")
    table.add_column("Details")
    for r in results:
        ms = f"{r['ms']:.0f} ms" if "ms" in r else ""
        table.add_row(r["name"], icons[r["status"]], ms, r.get("model", ""), r["detail"])
    console.print(table)

    workers_ok = sum(1 for r in results if r["status"] == "ok" and not r["name"].startswith("judge"))
    judge_ok = any(r["status"] == "ok" and r["name"].startswith("judge") for r in results)
    console.print()
    if workers_ok >= 2 and judge_ok:
        console.print(f"[green bold]Ready:[/green bold] {workers_ok} workers + judge connected.")
    else:
        need = []
        if workers_ok < 2:
            need.append(f"at least 2 working workers (have {workers_ok})")
        if not judge_ok:
            need.append("a working Claude judge key")
        console.print(f"[yellow]Not ready yet:[/yellow] need {' and '.join(need)}.")


if __name__ == "__main__":
    asyncio.run(main())