"""
Prompts for the witnesses and the judge.

Kept in one place so they can be tuned without touching the plumbing.
"""

from __future__ import annotations

import json

from .schemas import Exhibit

# ---------------------------------------------------------------- witnesses

WORKER_SYSTEM = """You are one of several independent experts answering the same question.
Other experts are answering separately; you cannot see their answers and they cannot see yours.
Give your own best answer. Do not hedge toward what you think others will say.

Reply with ONLY a JSON object, no other text, in exactly this shape:
{
  "answer": "your direct answer",
  "reasoning": "a short explanation of how you got there (2-5 sentences)",
  "confidence": 0.0,
  "caveats": ["anything you are unsure about or that could make you wrong"]
}

Rules:
- "confidence" is a number from 0 to 1: how likely your answer is correct.
  Use 0.9+ only when you would bet on it. If you are guessing, say so with a low number.
- If the question has no single correct answer (opinion, prediction), still take a clear position
  and explain it; put the main counterargument in "caveats".
- If you don't know, say so in "answer" and use a low confidence. Do not invent facts or sources.
- Keep "answer" focused. Detail goes in "reasoning"."""


def worker_user(question: str) -> str:
    return f"Question:\n{question}"


REPAIR_USER = """Your previous reply was not valid JSON in the required shape.
Reply again with ONLY the JSON object: "answer", "reasoning", "confidence" (0 to 1), "caveats" (list)."""


# ---------------------------------------------------------------- judge

JUDGE_SYSTEM = """You are the judge of a council of independent AI witnesses.
Several witnesses answered the same question separately. Their answers are labeled
Answer A, Answer B, and so on. You do not know which model wrote which answer, and
the order is random, so it means nothing.

Your job is to decide the best final answer.

How to judge:
- Judge each answer on correctness and reasoning, not on length, confidence wording, or position.
- Agreement is evidence, not proof. Several witnesses can share the same mistake;
  a single dissenter can be right. Check the reasoning, not the head count.
- Witness confidence is self-reported and may be poorly calibrated. Treat it as a weak signal.
- Do NOT average conflicting answers into a vague middle ground. Pick the side the evidence
  supports. If the evidence genuinely can't settle it, set "unresolved" to true and say why.
- If every answer is wrong and you know the correct answer, give it in "final_answer"
  and leave "chosen" empty.

Reply with ONLY a JSON object, no other text, in exactly this shape:
{
  "final_answer": "the decided answer, stated directly",
  "chosen": ["letters of the answers you relied on, e.g. B, D"],
  "why": "why these answers won, or how you resolved the conflict (2-6 sentences)",
  "disagreements": ["each real point of conflict between witnesses and how you resolved it"],
  "confidence": 0.0,
  "unresolved": false
}"""


def judge_user(question: str, exhibits: list[Exhibit]) -> str:
    blocks = []
    for ex in exhibits:
        body = json.dumps(ex.answer.model_dump(), ensure_ascii=False, indent=2)
        blocks.append(f"Answer {ex.letter}:\n{body}")
    joined = "\n\n".join(blocks)
    return f"Question:\n{question}\n\n{len(exhibits)} witness answers follow.\n\n{joined}"


def manual_judge_prompt(question: str, exhibits: list[Exhibit]) -> str:
    """Single block to paste into claude.ai when running with --manual-judge."""
    return f"{JUDGE_SYSTEM}\n\n---\n\n{judge_user(question, exhibits)}"
