# LLM Tribunal

**A council of AI models testifies independently. Claude reads the anonymized testimony and delivers the verdict.**

Ask one question and six models from six different families answer it in parallel, each with its reasoning, a confidence score and its caveats. Their answers are anonymized, shuffled and handed to Claude, who acts as judge: it weighs the reasoning (not the head count), picks a side, explains why, and lists where the witnesses disagreed.

The witnesses run on free API tiers or locally. The judge is the only paid call, at well under a cent per question with Claude Haiku.

```
                    ┌──────────────┐
        question ──▶│  Questioning │  all witnesses at once (asyncio)
                    └──────┬───────┘
   ┌─────────┬─────────┬───┴─────┬──────────┬─────────────┐
   ▼         ▼         ▼         ▼          ▼             ▼
 Gemini     Groq     Mistral  OpenRouter  Llama 3.1    Nemotron
 (Google) (gpt-oss) (Mistral) (Flint)    (local PC)  (Ollama cloud)
   └─────────┴─────────┴────┬────┴──────────┴─────────────┘
                            ▼
          validate → WorkerAnswer {answer, reasoning, confidence, caveats}
                            ▼
          anonymize + shuffle → Answer A, B, C… (seed logged)
                            ▼
                    ┌──────────────┐
                    │ Claude judge │  the only paid call
                    └──────┬───────┘
                            ▼
      verdict · reasoning · disagreements · confidence · .txt report
```

## Why build it this way

A single model answers confidently whether it's right or not. Asking several and taking a majority vote doesn't fix that: models trained on similar data often share the same mistake. So the judge is told to **check the reasoning, not count votes**, and the system is built to keep the judge honest:

| Problem | What the tribunal does |
|---|---|
| A judge favoring a model it "knows" | Witness names are hidden. The judge only sees Answer A, B, C… |
| Favoring whatever is read first | Answers are **shuffled** every run. The seed is recorded in each report, so the exact order can be reproduced. |
| Blending conflicting answers into mush | The judge must pick a side, or mark the question **unresolved** and say why. |
| Everyone being wrong | The judge can override all witnesses and give its own answer. |
| Witnesses agreeing on the same mistake | The council is chosen so that **no two witnesses come from the same model family**. |
| One provider down or rate-limited | That witness is skipped. The run continues with a quorum of 2. |
| Models returning broken JSON | Every reply is validated. A malformed one gets one automatic repair attempt. |

## A real example

Asked (in Spanish) whether goods imported under a Mexican customs **pedimento A1** can be resold to a company, and what invoicing scheme supports it legally:

- **All five** witnesses that answered said "yes," with 85–98% confidence.
- The judge **didn't count votes.** It found that three of them misidentified what an A1 pedimento is: one called it ambiguous, one a *franquicia*, one *reexportación*. Some also cited doubtful articles.
- It relied only on the two witnesses that correctly identified A1 as a **definitive import**. Its verdict cited the CFDI requirements and the relevant articles of the Ley Aduanera and the Código Fiscal.

This run also showed the calibration problem directly: the **most** confident witness (98%) was wrong, and so was one of the two at 95%, while a witness at 92% was right. Stated confidence didn't separate right from wrong. That's why the project includes bias and calibration testing.

> Verdicts on legal, tax or customs questions are a research aid, not professional advice. Verify with a qualified professional.

## The council

| Witness | Family | Model | Provider | Notes |
|---|---|---|---|---|
| `gemini` | Google | `gemini-3.5-flash-lite` | Google AI Studio (free) | |
| `groq` | OpenAI | `openai/gpt-oss-120b` | Groq (free) | Fastest, ~1–3 s |
| `mistral` | Mistral | `ministral-8b-latest` | Mistral (free) | `mistral-small` is rate-limited on the free tier |
| `openrouter` | Recursive | `recursive/flint-v1.0:free` | OpenRouter (free) | 50 requests/day |
| `local` | Meta | `llama3.1:8b` | Ollama on your PC | Unlimited, private, offline |
| `ollama_cloud` | NVIDIA | `nemotron-3-nano:30b` | Ollama cloud (free) | Session and weekly caps |
| **judge** | Anthropic | `claude-haiku-5-5` | Anthropic API (paid) | Configurable; see below |

Every model is set in `.env`, so you can swap any witness without touching code. Free model lineups change often; `discovery/probe.py --list-free` shows what's available for your keys today.

## Quick start

**Requirements:** Python 3.11+, and optionally [Ollama](https://ollama.com) for the local witness.

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # Windows: copy .env.example .env
# fill in your keys; any witness without a key is simply skipped
```

Free keys: [Google AI Studio](https://aistudio.google.com), [Groq](https://console.groq.com), [OpenRouter](https://openrouter.ai), [Mistral](https://console.mistral.ai), [Ollama](https://ollama.com/settings/keys). The judge needs an [Anthropic API](https://console.anthropic.com) key with prepaid credits.

For the local witness: install Ollama and run `ollama pull llama3.1:8b` (needs ~8 GB of free RAM).

**Check every connection:**
```bash
python discovery/probe.py
```
Each service gets one tiny test message and the table shows what works and why anything failed.

**Convene the tribunal:**
```bash
python -m core.judge "Is a tomato a fruit or a vegetable?"
```

You get a testimony table, the verdict in the terminal, and a full report in `runs/` (`ruling_<time>.txt`, or `dictamen_<time>.txt` for questions in Spanish).

## Commands

| Command | What it does |
|---|---|
| `python -m core.judge "question"` | Full tribunal: all witnesses, then the judge, then a `.txt` report |
| `python -m core.judge --manual "question"` | Skip the API judge; save a prompt to paste into claude.ai instead |
| `python -m core.judge --no-txt "question"` | Same, without writing the report |
| `python -m core.workers list` | Show every witness and whether it's configured |
| `python -m core.workers groq "question"` | Question one witness only |
| `python discovery/probe.py` | Connection check for every service |
| `python discovery/probe.py --list-free` | Also list the Gemini, OpenRouter and Ollama cloud models your keys can use |
| `python discovery/self_report.py` | Ask each witness to describe its own biases |
| `python discovery/bias_test.py` | Measure biases with graded tests (see below) |
| `python tests/test_core.py` | Offline tests: no API calls, costs nothing |

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Gemini witness model |
| `MISTRAL_MODEL` | `ministral-8b-latest` | Mistral witness model |
| `OPENROUTER_MODEL` | `recursive/flint-v1.0:free` | Any OpenRouter model id ending in `:free` |
| `OLLAMA_MODEL` | `llama3.1:8b` | Local model (must be pulled first) |
| `OLLAMA_CLOUD_MODEL` | `nemotron-3-nano:30b` | Ollama cloud model |
| `JUDGE_MODEL` | `claude-haiku-5-5` | The judge |
| `JUDGE_ESCALATE_MODEL` | *(off)* | e.g. `claude-sonnet-5-5`: re-judges when the first verdict is unresolved or below 50% confidence |

## Measuring witness bias

Confidence scores from language models are self-reported and often badly calibrated, so the project measures them instead of trusting them.

1. **Self-report** (`self_report.py`): each witness lists its suspected biases, each with a test question. Self-reports are treated as *suspicions*, not evidence. Models tend to repeat generic AI-bias talk. One claimed a 2021 training cutoff that almost certainly isn't its own.
2. **Graded tests** (`bias_test.py` + `bias_suite.yaml`): every test goes to every witness, and Claude grades each answer against a fixed rubric:
   - **Calibration:** questions with known answers, comparing accuracy with stated confidence (Brier score)
   - **Hallucination traps:** fictional entities and false premises, some taken from the witnesses' own hallucinated test questions
   - **Framing:** the same issue asked pro, con and neutral. Does the answer follow the wording?
   - **Leanings and softening:** stance and one-sidedness on value questions, and over-hedging on sensitive ones
3. **Profiles:** the results become a proposed per-witness profile, with a side-by-side of *self-reported vs. measured* bias. For example, a witness that averaged 95% confidence but was right 54% of the time would get `confidence_scale: 0.57`.

Samples are small and the grader is itself a model with its own biases, so profiles are reviewed before being trusted. Leanings are recorded, not "corrected."

## Project structure

```
core/
  schemas.py        data shapes + tolerant parsing (WorkerAnswer, Verdict, Testimony…)
  prompts.py        witness and judge instructions
  workers.py        witness registry, calls, retries, JSON repair
  judge.py          anonymize, shuffle, call Claude, escalate; CLI entry point
  report_txt.py     plain-text report (English or Spanish labels)
discovery/
  probe.py          connection check for every service
  self_report.py    witnesses describe their own biases
  bias_suite.yaml   bias test questions and rubrics
  bias_test.py      runs and grades the bias tests
tests/
  test_core.py      offline tests for parsing, validation and prompts
TESTIMONY_ORDER.md  design: testimony order, deadlines, tiers
PLAN.md             project plan and status
config.yaml         design values for features still in progress
```

## Limitations and privacy

- **Don't send confidential data to the free witnesses.** Some free tiers, including Gemini's, may use prompts to improve their models.
- **The council isn't a calculator.** Language models are unreliable at arithmetic. Use the tribunal to review reasoning and formulas, and code to check the numbers.
- **Agreement isn't proof.** The judge is instructed accordingly, but a unanimous wrong answer can still get through.
- **Free tiers change without notice.** Models get retired and limits move. Run `probe.py --list-free` when something stops working.

## Roadmap

Designed (see `TESTIMONY_ORDER.md`) but not built yet:

- Soft deadline: close testimony once a quorum answers and 20 s have passed
- Priority tiers for batch runs, so scarce free quotas are saved for questions where witnesses disagree
- `--double-check`: judge twice with reversed order and flag order-sensitive verdicts
- Applying measured `confidence_scale` profiles before the judge sees confidence
- `--file` option to send a source file (e.g. a `.py` with formulas) for review
- Optional second round where witnesses see each other's anonymized answers

## License

[MIT](LICENSE)
