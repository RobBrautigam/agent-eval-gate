"""A stand-in support agent, version 1: it sorts a ticket, decides whether a person should take
it, and drafts a reply.

It is deterministic keyword rules, not a language model, so the quick start runs offline and
costs nothing. Its token counts and latency are SIMULATED from the text length, in the same
shape a real model call reports them. To evaluate your own agent, point --agent at a function
that calls your model and returns the same fields (see the README).
"""
from __future__ import annotations

import math

MODEL = "example-small"
PROMPT_TOKENS = 180

CATEGORIES = [
    ("billing", ["charge", "refund", "invoice", "card", "billing"]),
    ("bug", ["error", "crash", "broken", "doesn't work", "bug", " 500"]),
    ("account", ["password", "log in", "login", "locked", "2fa"]),
    ("how_to", ["how do i", "how can i", "where is", "where do i"]),
]
UPSET = ["unacceptable", "furious", "lawyer", "cancel", "third time", "!!!"]

REPLIES = {
    "billing": "Thanks for flagging this. I have opened a billing review for your account, "
               "and you will get an update within one business day.",
    "bug": "Thanks for the report. I have logged this with our engineers and attached your details.",
    "account": "You can reset access from the sign-in page with Forgot password. If you are still "
               "locked out, reply here and we will verify your identity.",
    "how_to": "Good question. The guide is here: https://example.com/help",
    "other": "Thanks for reaching out. A teammate will reply shortly.",
}


def tokens(text: str) -> int:
    return math.ceil(len(text.split()) * 1.3)


def triage(ticket: dict, categories=CATEGORIES, upset=UPSET) -> tuple[str, bool]:
    text = f"{ticket.get('subject', '')} {ticket.get('body', '')}".lower()
    category = next((name for name, words in categories if any(w in text for w in words)), "other")
    escalate = any(word in text for word in upset)
    return category, escalate


def run(ticket: dict) -> dict:
    category, escalate = triage(ticket)
    reply = ("I have passed this to a senior teammate, who will take it from here. " if escalate else "") \
        + REPLIES[category]
    out_tokens = tokens(reply) + 12
    return {
        "output": {"category": category, "escalate": escalate, "reply": reply},
        "model": MODEL,
        "input_tokens": PROMPT_TOKENS + tokens(ticket.get("subject", "") + " " + ticket.get("body", "")),
        "output_tokens": out_tokens,
        "latency_ms": 250 + 6 * out_tokens,  # simulated
    }
