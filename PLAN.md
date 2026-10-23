# Project Plan: LLM Tribunal

**Status:** Working v1. Tribunal, judge, reports and bias tooling run end to end.
**Last updated:** 2026-10-23

**Goal:** Send one question to several free LLM witnesses in parallel, collect structured answers, and have Claude judge them anonymously and deliver a reasoned verdict.

---

## 1. Status

### Built

| Area | Files | Notes |
|---|---|---|
| Data shapes and parsing | `core/schemas.py` | Tolerates code fences, extra text, `85%` confidence, raw line breaks, answers sent as objects |
| Prompts | `core/prompts.py` | Independent witnesses; judge told to weigh reasoning, not votes, and never average |
| Witnesses | `core/workers.py` | 6 witnesses from `.env`, retries with backoff, one JSON repair attempt, never crashes a run |
| Judge | `core/judge.py` | Anonymize, shuffle with logged seed, validate letters, optional escalation, `--manual` fallback |
| Reports | `core/report_txt.py` | `.txt` per run in `runs/`, English or Spanish labels |
| Connection check | `discovery/probe.py` | Per-service status, `--list-free` model discovery |
| Bias tooling | `discovery/self_report.py`, `bias_test.py`, `bias_suite.yaml` | Self-reports, graded tests, proposed profiles |
| Tests | `tests/test_core.py` | 20 offline tests |

### Designed, not built yet
See `TESTIMONY_ORDER.md` for the full design.

- Soft deadline (close testimony at quorum + 20 s)
- Priority tiers for batch runs
- `--double-check` (judge twice in reversed order)
- Health-check caching between runs
- Reading `config.yaml` (values are currently set in code and `.env`)
- Applying measured `confidence_scale` before judging
- `--file` option for code or document review
- JSON run log per question (`RunRecord` schema exists, not yet written)

---

## 2. The council

| Witness | Family | Model | Why |
|---|---|---|---|
| `gemini` | Google | `gemini-3.5-flash-lite` | Stable free Flash model (the `-latest` alias was often overloaded) |
| `groq` | OpenAI | `openai/gpt-oss-120b` | Fastest and most reliable JSON |
| `mistral` | Mistral | `ministral-8b-latest` | `mistral-small` is blocked on the free tier |
| `openrouter` | Recursive | `recursive/flint-v1.0:free` | No DeepSeek/Qwen on the free list; Flint sometimes returns empty replies on long prompts |
| `local` | Meta | `llama3.1:8b` | Fits 16 GB RAM; unlimited and private |
| `ollama_cloud` | NVIDIA | `nemotron-3-nano:30b` | New family; on Ollama's free credit list |
| judge | Anthropic | `claude-haiku-5-5` | Cheap for building; Sonnet planned for normal use, Opus for escalation |

**Diversity rule:** no two witnesses from the same model family.

---

## 3. Decisions taken

| Question | Decision |
|---|---|
| Judge model | Haiku while building, Sonnet in use, Opus only for escalations |
| Local worker | Ollama + `llama3.1:8b` (16 GB RAM) |
| One round or debate | One round in v1; debate design in `TESTIMONY_ORDER.md` §5 |
| Who testifies first | Nobody: parallel questioning, shuffled presentation |
| Report format | `.txt` by default |
| Main use case | Mexican customs/tax questions (Spanish) and general reasoning |

---

## 4. Lessons from real runs

- **Stated confidence is unreliable.** In the pedimento A1 run, the two most confident witnesses (98%, 95%) were wrong. Calibration needs measuring.
- **Self-reported biases are boilerplate.** One witness claimed a 2021 cutoff. Another's own test question contained invented "proverbs," which became a hallucination trap.
- **Small models break the JSON format** (raw line breaks, nested objects), so the parser is deliberately tolerant.
- **Free tiers drift.** Models get retired (`gemini-2.5-flash-lite`), aliases get overloaded, limits tighten. `probe.py --list-free` is the fix.
- **Windows consoles** can't print some Unicode characters; output is forced to UTF-8.

---

## 5. Next steps

1. Run `bias_test.py`, review the proposed profiles, save confirmed values to `config/witness_profiles.yaml`
2. Apply `confidence_scale` before the judge sees confidence
3. Domain prompts for legal/customs questions: cite specific articles, flag uncertainty about whether an article exists
4. Add domain questions with known answers to the bias suite
5. Implement the soft deadline and the JSON run log
6. `--file` option for reviewing source code (e.g. formula checks)
