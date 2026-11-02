"""Version 2 of the stand-in support agent: the "prompt change" a team ships on a Friday.

What changed, the way a real change looks:
- a shorter, friendlier prompt that opens every reply with an apology;
- a larger model, because the replies "felt better" in a few manual tries;
- the upset-customer list was trimmed while rewriting the prompt;
- a new how-to phrase was added, which fixes one ticket version 1 got wrong.

Deterministic and simulated like version 1: no model is called.
"""
from __future__ import annotations

from agent_v1 import CATEGORIES, REPLIES, tokens, triage

MODEL = "example-large"
PROMPT_TOKENS = 140

CATEGORIES_V2 = CATEGORIES[:-1] + [
    ("how_to", ["how do i", "how can i", "where is", "where do i", "is there a way"]),
]
UPSET_V2 = ["unacceptable", "furious", "lawyer", "!!!"]  # "cancel" and "third time" fell out


def run(ticket: dict) -> dict:
    category, escalate = triage(ticket, CATEGORIES_V2, UPSET_V2)
    reply = "So sorry about this! " \
        + ("I have passed this to a senior teammate, who will take it from here. " if escalate else "") \
        + REPLIES[category]
    out_tokens = tokens(reply) + 12
    return {
        "output": {"category": category, "escalate": escalate, "reply": reply},
        "model": MODEL,
        "input_tokens": PROMPT_TOKENS + tokens(ticket.get("subject", "") + " " + ticket.get("body", "")),
        "output_tokens": out_tokens,
        "latency_ms": 400 + 9 * out_tokens,  # simulated
    }
