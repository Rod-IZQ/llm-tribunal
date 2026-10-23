[i] Using default configuration
[✓] Codebase Traversal Done!                                                                                            [i] Token count: 2K, Model info: ChatGPT models, text-embedding-ada-002
Project Path: AI_orchestrator

Source Tree:

```txt
AI_orchestrator
├── PLAN.md
├── core
├── discovery
│   └── probe.py
└── orchestrator
    ├── Orchestrator.py
    ├── jury
    ├── problem_batch
    ├── testimony
    └── verdict

```

`PLAN.md`:

```md
# Multi-Agent Council: Free Workers, Claude Has the Final Say

**Status:** Planning (no code yet)
**Version:** 0.1
**Goal:** Send one question to several free LLM "worker" agents in parallel, collect their structured answers, and have Claude act as the judge that reviews them all and gives the final answer.

---

## 1. How a run works

```
              ┌──────────────┐
  question ──▶│ Orchestrator │
              └──────┬───────┘
        fan out in parallel (asyncio)
   ┌────────┬────────┼────────┬────────┐
   ▼        ▼        ▼        ▼        ▼
 Gemini   Groq   OpenRouter Mistral  Ollama     ← free workers
  Flash  gpt-oss  (:free)    Small   (local)
   └────────┴────────┼────────┴────────┘
                     ▼
         normalize → WorkerAnswer (JSON)
                     ▼
         anonymize (Answer A, B, C…)
                     ▼
              ┌──────────────┐
              │ Claude judge │  ← the only paid call
              └──────┬───────┘
                     ▼
     final answer + reasoning + disagreements
                     ▼
        saved to runs/<timestamp>.json
```

1. **Fan out:** the same question goes to every enabled worker at once.
2. **Normalize:** each worker must return the same JSON shape (see section 5).
3. **Anonymize:** worker names are hidden from the judge so Claude judges the answers, not the brands.
4. **Judge:** Claude reads all answers, picks or merges the best, and lists where workers disagreed.
5. **Log:** the full run (question, every raw answer, verdict, timings, errors) is saved for review.

A worker that times out or hits a rate limit is **skipped, not fatal**. The run continues as long as at least 2 workers answered.

---

## 2. Agents in this run

### Workers (free)

| # | Agent name | Provider | Model (starting choice) | Why it's in the council | Free limit (approx., changes often) |
|---|---|---|---|---|---|
| 1 | `gemini` | Google AI Studio | Gemini Flash (current free Flash model) | Strong general reasoning, large context | Per-project; check AI Studio console |
| 2 | `groq` | Groq | `openai/gpt-oss-120b` | Very fast; different model family | ~30 req/min, ~1,000 req/day |
| 3 | `openrouter` | OpenRouter | One `:free` model (DeepSeek, Qwen or Kimi), picked at startup | Adds a Chinese open-weight family for diversity | 20 req/min; 50 req/day (1,000 after a one-time $10 top-up) |
| 4 | `mistral` | Mistral (Free mode) | `mistral-small-latest` | European model family, different training | Free mode, rate-limited |
| 5 | `local` | Ollama (your computer) | A Qwen or Gemma model that fits your RAM | No limits, private, works offline | Unlimited (hardware-bound) |

**Optional extras** (off by default, easy to switch on): Cerebras, Cohere trial key.

**Diversity rule:** no two workers from the same model family. Five different families give more useful disagreement than ten variants of one.

### Judge (paid, cheap)

| Role | Provider | Model | Notes |
|---|---|---|---|
| Final say | Anthropic API | `claude-sonnet-5-5` (default) or `claude-haiku-5-5` (cheaper) | Anthropic has no free API tier, so this is the one paid call per run. It only reads the workers' short answers, so each run is low cost. |

**No-API fallback:** a `--manual-judge` mode that skips the API call and writes a ready-to-paste prompt file you can drop into claude.ai yourself.

---

## 3. Libraries

Versions are the latest on PyPI as of today. Pin them in `requirements.txt`.

| Library | Version | What it's for |
|---|---|---|
| `litellm` | 1.106.2 | **One interface to every worker.** Same `acompletion()` call for Gemini, Groq, OpenRouter, Mistral, Cohere and Ollama; also has built-in retries and fallbacks. |
| `anthropic` | 2.2.0 | Official SDK for the Claude judge call. Kept separate from litellm so the judge step is explicit and easy to swap. |
| `pydantic` | 2.14.1 | Defines and validates the `WorkerAnswer` and `Verdict` shapes, so a malformed worker reply is caught instead of reaching the judge. |
| `python-dotenv` | 1.2.4 | Loads API keys from a `.env` file so they never live in code. |
| `tenacity` | 9.2.1 | Retry with exponential backoff on HTTP 429 / timeouts, per worker. |
| `rich` | 15.0.0 | Readable terminal output: a live table of worker status and the final verdict. |
| `asyncio` | stdlib | Runs all workers in parallel. |

**Not needed for v1:** LangGraph, CrewAI, or another orchestration framework. This is a single fan-out → judge pattern, which is about 200 lines of plain Python. A framework adds weight without adding anything we use yet. We can move to LangGraph later if we add multi-round debate or tool use.

**Alternative considered:** the `openai` SDK (3.37.0) pointed at each provider's OpenAI-compatible endpoint. It works, but means handling each provider's quirks ourselves; litellm already does that.

---

## 4. What you need before building

### Accounts and keys (all free except Anthropic)

| Key | Where to get it | Needs a card? |
|---|---|---|
| `GEMINI_API_KEY` | aistudio.google.com | No |
| `GROQ_API_KEY` | console.groq.com | No |
| `OPENROUTER_API_KEY` | openrouter.ai | No (optional $10 top-up raises daily limit to 1,000) |
| `MISTRAL_API_KEY` | console.mistral.ai (Free mode) | No |
| `ANTHROPIC_API_KEY` | console.anthropic.com | Yes, prepaid credits |

### Local setup

- Python 3.11 or newer
- Ollama installed, plus one model pulled (e.g. `ollama pull qwen3:8b`), if you want the local worker
- A `.env` file with the keys above (a `.env.example` will be provided)

### Things to know about free tiers

- **Data use:** Gemini's free tier may use your prompts to improve its models. Don't send private or confidential data to free workers.
- **Commercial use:** Gemini's free tier excludes commercial use. Fine for personal projects and prototypes.
- **Limits change without notice.** The code checks each provider at startup and disables any that fail, instead of crashing.
- **No scraping chat websites.** Workers use official APIs only.

---

## 5. Data shapes

**Worker output** (every worker must return this JSON):

```json
{
  "answer": "the worker's direct answer",
  "reasoning": "short explanation of how it got there",
  "confidence": 0.0,
  "caveats": ["anything it is unsure about"]
}
```

**Judge output:**

```json
{
  "final_answer": "Claude's decided answer",
  "chosen": ["B", "D"],
  "why": "why these answers won or how they were merged",
  "disagreements": ["where workers conflicted and how it was resolved"],
  "confidence": 0.0
}
```

The judge is told explicitly **not** to average conflicting answers into a vague middle ground; it must pick a side or say the question is unresolved.

---

## 6. Planned project layout

```
council/
├── PLAN.md               ← this file
├── requirements.txt
├── .env.example
├── config.yaml           ← which workers are on, model names, timeouts
├── council/
│   ├── workers.py        ← litellm calls, retries, JSON parsing
│   ├── judge.py          ← Claude call + manual-judge fallback
│   ├── schemas.py        ← pydantic models (WorkerAnswer, Verdict)
│   ├── prompts.py        ← worker and judge system prompts
│   └── run.py            ← CLI entry point: python -m council.run "question"
└── runs/                 ← one JSON log per run
```

---

## 7. Build order

1. `schemas.py` and `prompts.py`
2. `workers.py` with one worker (Groq) working end to end
3. Add the other workers and the startup health check
4. `judge.py` with the Claude call and the `--manual-judge` fallback
5. `run.py` CLI with rich output and JSON logging
6. Test with three question types: a factual one, a reasoning puzzle, and an opinion question where workers should disagree

---

## 8. Open decisions

- **Judge model:** Sonnet (better judgment) or Haiku (cheaper)? Default is Sonnet.
- **Local worker:** do you have Ollama installed, and how much RAM does your computer have?
- **One round or debate?** v1 is one round. A later version could show workers each other's answers for a second round before Claude judges.
- **Use case:** what kinds of questions will you mostly ask? That decides whether we tune the prompts for research, coding or general reasoning.

```