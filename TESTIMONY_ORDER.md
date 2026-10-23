# Testimony Order: Who Speaks First

**Status:** Design document. Partly implemented (see table below).
**Applies to:** `core/workers.py` (questioning), `core/judge.py` (exhibits and verdict)
**Based on:** connection check of 2026-10-23 (6 workers + judge connected)

### Implementation status

| Feature | Status |
|---|---|
| Parallel questioning of all witnesses | ✅ Built |
| Shuffled exhibit order with logged seed | ✅ Built |
| Quorum of 2 before calling the judge | ✅ Built |
| Retries with backoff on 429 / timeout | ✅ Built |
| Judge escalation on low confidence or unresolved | ✅ Built (`JUDGE_ESCALATE_MODEL`) |
| `--manual` fallback when the API judge isn't used | ✅ Built |
| Per-witness timeouts | ✅ Built (set in code, not `config.yaml`) |
| Soft deadline (quorum + 20 s) | ⏳ Planned |
| Health-check caching | ⏳ Planned |
| Priority tiers for batch runs | ⏳ Planned |
| `--double-check` (reversed-order re-judging) | ⏳ Planned |
| Automatic fallback to `--manual` when the API judge fails | ⏳ Planned |
| Reading these values from `config.yaml` | ⏳ Planned |

---

## 1. Short answer

In v1, **no worker testifies "first" in time.** All witnesses are questioned at once, in parallel. What *does* have an order is:

1. **Swearing in.** The health check runs before every session; only witnesses that pass are called.
2. **Presentation order.** The order the judge reads the answers in. This is **shuffled randomly on every run.**
3. **Priority.** When we can't call everyone (quota, batch jobs), a fixed tier list decides who is called and who is dropped first.

The rest of this document explains each one.

---

## 2. The current council

| Witness | Family | Model | Check time | Free limit | Tier |
|---|---|---|---|---|---|
| `local` | Meta (Llama) | `llama3.1:8b` | 0.1 s (warm), ~10 s cold | Unlimited (your PC) | 1 |
| `groq` | OpenAI (gpt-oss) | `openai/gpt-oss-120b` | 5.3 s | ~30/min, ~1,000/day | 1 |
| `gemini` | Google | `gemini-3.5-flash-lite` | 7.7 s | Per-project, check AI Studio | 1 |
| `ollama_cloud` | NVIDIA (Nemotron) | `nemotron-3-nano:30b` | 2.9 s | 5-hour session + weekly cap, 1 model at a time | 2 |
| `mistral` | Mistral | `ministral-8b-latest` | 5.2 s | Tight; `mistral-small` already blocked | 2 |
| `openrouter` | Recursive (Flint) | `recursive/flint-v1.0:free` | 1.6 s | 20/min, 50/day | 3 |
| **judge** | Anthropic | `claude-haiku-5-5` | 1.1 s | Paid, prepaid credits | — |

Limits are approximate and change without notice. The health check is the source of truth.

---

## 3. Phase by phase

### 3.1 Swearing in (before questioning)

`discovery/probe.py` logic runs at session start:

- A witness that fails (bad key, model not found, rate-limited) is **excused for this session**, not retried mid-run.
- If fewer than **2 witnesses** pass, the session stops before spending a judge call.
- Results are cached for 10 minutes so a batch of questions doesn't re-check every time.

### 3.2 Questioning (call order)

**All sworn-in witnesses are called at the same moment** (`asyncio.gather`).

Why not sequential:
- **Speed.** A run takes as long as the slowest witness (~8 s), not the sum (~23 s).
- **Independence.** No witness can see another's answer, so there's no anchoring. Disagreement stays genuine.

**Deadlines:**

| Rule | Value |
|---|---|
| Per-witness timeout | 30 s (cloud), 120 s (`local`, first load only) |
| Quorum | 2 answers minimum |
| Soft deadline | Once **quorum is met and 20 s have passed**, close testimony. Late answers are logged in the run file but not shown to the judge. |

Retries follow `tenacity`: on HTTP 429 or timeout, up to 2 retries with exponential backoff, but **never past the soft deadline**.

### 3.3 Presentation to the judge (exhibit order)

This is the order that matters most, because judges, human or model, tend to favor whatever they read first and whatever is longest.

1. Collect the answers that arrived before the deadline.
2. **Shuffle them randomly.** Arrival order is *not* used, because the fastest witness (`local`) would almost always be first.
3. Assign letters **after** shuffling: Answer A, B, C…
4. Log the **seed** and the letter → witness mapping in `runs/<timestamp>.json`, so any verdict can be traced and replayed.

**High-stakes mode (optional flag `--double-check`):** run the judge twice, the second time with the order reversed. If the verdict changes, mark the result **"order-sensitive, low confidence"** in the output.

### 3.4 Verdict

The judge always speaks last and alone.

| Situation | Judge |
|---|---|
| Building / debugging / bias probes | `claude-haiku-5-5` |
| Normal use | `claude-sonnet-5-5` |
| Witnesses strongly split, or judge confidence < 0.5 | Escalate once to `claude-opus-5-5` |

Set via `JUDGE_MODEL` and `JUDGE_ESCALATE_MODEL` in `.env`; escalation logic lives in `core/judge.py`.

---

## 4. Priority tiers (who gets dropped first)

Used when we **can't** call everyone, mainly for `problem_batch` runs with many questions.

| Tier | Witnesses | Rule |
|---|---|---|
| **1. Always called** | `local`, `groq`, `gemini` | Generous or unlimited quotas. These carry the batch. |
| **2. Called while quota lasts** | `ollama_cloud`, `mistral` | Dropped automatically on the first 429 of the session. |
| **3. Called sparingly** | `openrouter` | 50/day. In batches, only on every Nth question or when tiers 1–2 disagree. |

**Batch budget example (100 questions):** tier 1 answers all 100; tier 2 answers until it hits a limit; `openrouter` is saved for the ~20–30 questions where the other witnesses split. That keeps it inside 50/day.

The judge is **never** dropped. If the Anthropic call fails, the run falls back to `--manual-judge` and writes a ready-to-paste prompt file.

---

## 5. Later: debate rounds (v2)

If we add a second round where witnesses see each other's anonymized answers:

- Round 2 is **also parallel**. Every witness sees the same shuffled set at the same time, so nobody gets the "last word."
- Witnesses never see their own answer labeled as theirs, which avoids defending it out of pride.
- Only witnesses that **disagreed with the majority** in round 1 are re-questioned. That saves quota and targets the real dispute.

---

## 6. Config sketch

How this will look in `config.yaml` (the file exists but isn't read by the code yet):

```yaml
testimony:
  mode: parallel
  quorum: 2
  soft_deadline_s: 20
  timeout_s: {default: 30, local: 120}
  retries: 2

exhibits:
  shuffle: true
  log_seed: true
  double_check: false   # --double-check flag turns it on

tiers:
  1: [local, groq, gemini]
  2: [ollama_cloud, mistral]
  3: [openrouter]

judge:
  default: claude-haiku-5-5
  escalate_to: claude-opus-5-5
  escalate_when: {split: true, confidence_below: 0.5}
```

---

## 7. Open questions

- **Soft deadline value.** 20 s fits today's timings (slowest = Gemini at 7.7 s). Revisit after real questions, since long answers take longer than "OK."
- **`openrouter` sampling rule** for batches: every Nth question, or only on disagreement? Start with "only on disagreement."
- **Re-test `mistral-small-latest`** occasionally. If the limit lifts, switch back from Ministral 8B.
