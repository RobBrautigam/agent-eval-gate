"""Cases, checks, recordings and scores.

A case is one line of a JSONL file: an id, an input, and the checks a right answer passes.
A recording (a "cassette") is what the agent answered for each case, with its token usage and
latency. Scoring reads a recording, never the agent, so changing a check costs nothing.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import math
import re
import sys
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

FORMAT_VERSION = 1


class EvalError(Exception):
    """A case file, recording, price table or agent that cannot be used, with the reason."""


# ---------------------------------------------------------------- cases

@dataclass(frozen=True)
class Case:
    id: str
    input: Any
    checks: tuple[dict, ...]
    tags: tuple[str, ...] = ()

    @property
    def input_hash(self) -> str:
        blob = json.dumps(self.input, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def load_cases(path: str | Path) -> list[Case]:
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise EvalError(f"cannot read the case file {path}: {exc}") from None
    cases: list[Case] = []
    seen: set[str] = set()
    for number, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            raw = json.loads(line)
        except ValueError as exc:
            raise EvalError(f"{path.name} line {number} is not JSON: {exc}") from None
        if not isinstance(raw, dict):
            raise EvalError(f"{path.name} line {number} is not a JSON object")
        cid = str(raw.get("id") or "")
        if not cid:
            raise EvalError(f"{path.name} line {number} has no id")
        if cid in seen:
            raise EvalError(f"{path.name} line {number}: the id {cid!r} is used twice")
        seen.add(cid)
        checks = raw.get("checks") or []
        if not checks:
            raise EvalError(f"case {cid!r} has no checks, so it cannot fail")
        for check in checks:
            _validate(cid, check)
        cases.append(Case(cid, raw.get("input"), tuple(checks), tuple(raw.get("tags") or ())))
    if not cases:
        raise EvalError(f"{path} holds no cases")
    return cases


# ---------------------------------------------------------------- checks

def _field(output: Any, path: str | None) -> Any:
    """`path` is a dotted path into a JSON answer: "category", "actions.0.name"."""
    if not path:
        return output
    value = output
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise KeyError(path) from None
    for part in path.split("."):
        if isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise KeyError(path)
    return value


def _leaves(value: Any) -> list[str]:
    """Every piece of text in a value: strings, keys, and numbers written out."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for key, item in value.items() for text in (str(key), *_leaves(item))]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _leaves(item)]
    return [] if value is None else [json.dumps(value)]


def _text(value: Any) -> str:
    """The text a check searches: the string itself, or every string in a JSON value, one per
    line (never the JSON encoding, where a quote reads as backslash-quote)."""
    return value if isinstance(value, str) else "\n".join(_leaves(value))


def _fold(text: str, check: dict) -> str:
    return text.lower() if check.get("ignore_case", True) else text


def _same(a: Any, b: Any) -> bool:
    """Equal and of the same JSON type: false is not 0, true is not 1, 1 is not 1.0 for a bool."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _equals(value: Any, check: dict) -> str | None:
    want = check.get("value")
    return None if _same(value, want) else f"expected {want!r}, got {value!r}"


def _one_of(value: Any, check: dict) -> str | None:
    allowed = check.get("value") or []
    return None if any(_same(value, a) for a in allowed) else f"expected one of {allowed!r}, got {value!r}"


def _contains(value: Any, check: dict) -> str | None:
    wants = check.get("value")
    wants = wants if isinstance(wants, list) else [wants]
    text = _fold(_text(value), check)
    missing = [w for w in wants if _fold(str(w), check) not in text]
    return None if not missing else f"missing {missing!r}"


def _not_contains(value: Any, check: dict) -> str | None:
    banned = check.get("value")
    banned = banned if isinstance(banned, list) else [banned]
    text = _fold(_text(value), check)
    present = [b for b in banned if _fold(str(b), check) in text]
    return None if not present else f"must not contain {present!r}"


def _regex(value: Any, check: dict) -> str | None:
    flags = re.IGNORECASE if check.get("ignore_case", True) else 0
    return None if re.search(check.get("value", ""), _text(value), flags) else f"no match for /{check.get('value')}/"


def _max_length(value: Any, check: dict) -> str | None:
    limit = int(check.get("value", 0))
    size = len(_text(value))
    return None if size <= limit else f"{size} characters, limit {limit}"


CHECKS: dict[str, Callable[[Any, dict], str | None]] = {
    "equals": _equals,
    "one_of": _one_of,
    "contains": _contains,
    "not_contains": _not_contains,
    "regex": _regex,
    "max_length": _max_length,
}


def _validate(cid: str, check: Any) -> None:
    """Refuse a check that could never run, when the file loads rather than halfway through a score."""
    if not isinstance(check, dict) or check.get("type") not in CHECKS:
        kind = check.get("type") if isinstance(check, dict) else check
        raise EvalError(f"case {cid!r}: unknown check type {kind!r} (known: {', '.join(sorted(CHECKS))})")
    if "value" not in check:
        raise EvalError(f"case {cid!r}: the {check['type']} check has no value")
    if "path" in check and (not isinstance(check["path"], str) or not check["path"]):
        raise EvalError(f"case {cid!r}: a check's path must be text such as \"category\" or \"actions.0.name\"")
    if check["type"] == "regex":
        if not isinstance(check["value"], str):
            raise EvalError(f"case {cid!r}: a regex check needs text as its value")
        try:
            re.compile(check["value"])
        except re.error as exc:
            raise EvalError(f"case {cid!r}: the regex {check.get('value')!r} does not compile: {exc}") from None
    if check["type"] == "max_length" and not isinstance(check.get("value"), int):
        raise EvalError(f"case {cid!r}: max_length needs a whole number as its value")
    if check["type"] == "one_of" and not isinstance(check.get("value"), list):
        raise EvalError(f"case {cid!r}: one_of needs a list as its value")


def run_checks(case: Case, output: Any) -> list[str]:
    """The reasons this output fails the case; an empty list is a pass."""
    reasons: list[str] = []
    for check in case.checks:
        label = f"{check['type']}" + (f" {check['path']}" if check.get("path") else "")
        try:
            value = _field(output, check.get("path"))
        except KeyError:
            reasons.append(f"{label}: no field {check.get('path')!r} in the answer")
            continue
        reason = CHECKS[check["type"]](value, check)
        if reason:
            reasons.append(f"{label}: {reason}")
    return reasons


# ---------------------------------------------------------------- agents and recordings

@dataclass
class Reply:
    """What an agent returns: the answer, and what it cost to get it.

    An agent may also return a bare string or a dict with the same keys."""
    output: Any
    model: str | None = None
    input_tokens: int | None = None   # None: not reported, so the cost is unknown, never $0
    output_tokens: int | None = None
    latency_ms: float | None = None   # the agent's own measure; otherwise the wall clock is used


def load_agent(spec: str) -> Callable[[Any], Any]:
    """`package.module:function` or `path/to/file.py:function`."""
    target, _, name = spec.rpartition(":")
    if not target or not name:
        raise EvalError(f"agent {spec!r} must look like module:function or file.py:function")
    try:
        if target.endswith(".py"):
            path = Path(target).resolve()
            if not path.exists():
                raise EvalError(f"no agent file at {path}")
            if str(path.parent) not in sys.path:
                sys.path.insert(0, str(path.parent))  # so the agent file can import its neighbors
            module_name = f"_agent_eval_gate_{path.stem}_{abs(hash(str(path)))}"
            loader_spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(loader_spec)
            sys.modules[module_name] = module
            loader_spec.loader.exec_module(module)
        else:
            if os.getcwd() not in sys.path:
                sys.path.insert(0, os.getcwd())  # the console script does not put the current folder there
            module = importlib.import_module(target)
    except EvalError:
        raise
    except (ImportError, SyntaxError) as exc:
        raise EvalError(f"cannot load the agent {target}: {type(exc).__name__}: {exc}") from None
    fn = getattr(module, name, None)
    if not callable(fn):
        raise EvalError(f"{target} has no function {name!r}")
    return fn


def _as_reply(answer: Any) -> Reply:
    if isinstance(answer, Reply):
        return answer
    if isinstance(answer, dict) and "output" in answer:
        return Reply(answer["output"], answer.get("model"), _count(answer.get("input_tokens")),
                     _count(answer.get("output_tokens")), answer.get("latency_ms"))
    return Reply(answer)


def _count(value: Any) -> int | None:
    return None if value is None else int(value)


def record(cases: list[Case], agent: Callable[[Any], Any], agent_name: str,
           on_case: Callable[[dict], None] | None = None) -> dict:
    """Call the agent once per case and keep everything scoring needs. `on_case` sees the
    recording after each case (the CLI saves it there, so an interrupted run keeps what it paid for)."""
    recording = {
        "format": FORMAT_VERSION,
        "agent": agent_name,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": {},
    }
    for case in cases:
        started = time.perf_counter()
        try:
            reply = _as_reply(agent(case.input))
            wall_ms = (time.perf_counter() - started) * 1000
            entry = {
                "input_hash": case.input_hash,
                # default=str: an answer that is not JSON (an SDK object) is kept as its text
                "output": json.loads(json.dumps(reply.output, ensure_ascii=False, default=str)),
                "model": None if reply.model is None else str(reply.model),
                "input_tokens": _count(reply.input_tokens),
                "output_tokens": _count(reply.output_tokens),
                "latency_ms": round(float(reply.latency_ms if reply.latency_ms is not None else wall_ms), 3),
                "latency_source": "agent" if reply.latency_ms is not None else "wall clock",
                "error": None,
            }
        except Exception as exc:  # an agent that raises, or answers in a shape that cannot be kept,
            wall_ms = (time.perf_counter() - started) * 1000  # is a failing case, not a crashed run
            entry = {"input_hash": case.input_hash, "output": None, "model": None, "input_tokens": None,
                     "output_tokens": None, "latency_ms": round(wall_ms, 3), "latency_source": "wall clock",
                     "error": f"{type(exc).__name__}: {exc}"}
        recording["cases"][case.id] = entry
        if on_case:
            on_case(recording)
    return recording


def save_recording(recording: dict, path: str | Path) -> None:
    """Write through a temporary file, so a run cut off mid-write never leaves half a recording."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(recording, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_recording(path: str | Path) -> dict:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise EvalError(f"cannot read the recording {path}: {exc}") from None
    if data.get("format") != FORMAT_VERSION or not isinstance(data.get("cases"), dict):
        raise EvalError(f"{path} is not a recording this version can read")
    return data


# ---------------------------------------------------------------- prices and scores

def load_prices(path: str | Path | None) -> dict[str, tuple[float, float]]:
    """Dollars per million input and output tokens, per model, from a TOML table you maintain."""
    if path is None:
        return {}
    try:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise EvalError(f"cannot read the price table {path}: {exc}") from None
    prices: dict[str, tuple[float, float]] = {}
    for model, row in (data.get("models") or {}).items():
        try:
            prices[model] = (float(row["input_per_million"]), float(row["output_per_million"]))
        except (KeyError, TypeError, ValueError):
            raise EvalError(f"price row {model!r} needs input_per_million and output_per_million") from None
    return prices


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile: p95 of 20 values is the 19th smallest."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


@dataclass
class CaseScore:
    id: str
    passed: bool
    reasons: list[str]
    cost_usd: float | None
    latency_ms: float | None   # None: no current measure (missing, or recorded for an old input)
    output: Any = None


@dataclass
class Score:
    agent: str
    cases: list[CaseScore] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def accuracy(self) -> float:
        return self.passed / len(self.cases) if self.cases else 0.0

    @property
    def cost_total(self) -> float | None:
        costs = [c.cost_usd for c in self.cases]
        return None if any(c is None for c in costs) else sum(costs)

    @property
    def cost_per_case(self) -> float | None:
        total = self.cost_total
        return None if total is None or not self.cases else total / len(self.cases)

    def latency(self, pct: float) -> float:
        return percentile([c.latency_ms for c in self.cases if c.latency_ms is not None], pct)

    def summary(self) -> dict:
        return {
            "agent": self.agent,
            "cases": len(self.cases),
            "passed": self.passed,
            "accuracy": round(self.accuracy, 4),
            "cost_total_usd": None if self.cost_total is None else round(self.cost_total, 6),
            "cost_per_case_usd": None if self.cost_per_case is None else round(self.cost_per_case, 6),
            "latency_p50_ms": round(self.latency(50), 1),
            "latency_p95_ms": round(self.latency(95), 1),
            "failures": [{"id": c.id, "reasons": c.reasons} for c in self.cases if not c.passed],
        }


def score(cases: list[Case], recording: dict, prices: dict[str, tuple[float, float]]) -> Score:
    result = Score(recording.get("agent", "agent"))
    entries = recording["cases"]
    for case in cases:
        entry = entries.get(case.id)
        if entry is None:
            result.cases.append(CaseScore(case.id, False, ["not in the recording: record again"], None, None))
            continue
        if entry.get("input_hash") != case.input_hash:
            result.cases.append(CaseScore(case.id, False, ["the case's input changed since it was recorded: "
                                                           "record again"], None, None))
            continue
        reasons = [f"the agent raised {entry['error']}"] if entry.get("error") else run_checks(case, entry.get("output"))
        cost = _cost(entry, prices)
        result.cases.append(CaseScore(case.id, not reasons, reasons, cost, float(entry.get("latency_ms") or 0.0),
                                      entry.get("output")))
    return result


def _reported(entry: dict) -> bool:
    return entry.get("input_tokens") is not None or entry.get("output_tokens") is not None


def _cost(entry: dict, prices: dict[str, tuple[float, float]]) -> float | None:
    if entry.get("error"):
        return 0.0  # the call raised: there is no answer to price
    if not _reported(entry):
        return None  # no token counts: unknown, never free
    tokens_in, tokens_out = int(entry.get("input_tokens") or 0), int(entry.get("output_tokens") or 0)
    if tokens_in + tokens_out == 0:
        return 0.0  # the agent said zero, in so many words
    model = entry.get("model")
    if model not in prices:
        return None  # unpriced: reported as unknown, never as free
    per_in, per_out = prices[model]
    return (tokens_in * per_in + tokens_out * per_out) / 1_000_000


def unpriced_models(recording: dict, prices: dict[str, tuple[float, float]]) -> list[str]:
    """What makes a cost unknown: models with no price row, and answers with no token counts."""
    entries = [e for e in recording["cases"].values() if not e.get("error")]
    models = {e.get("model") for e in entries
              if _reported(e) and int(e.get("input_tokens") or 0) + int(e.get("output_tokens") or 0) > 0}
    found = sorted(str(m) for m in models if m not in prices)
    silent = sum(1 for e in entries if not _reported(e))
    return found + ([f"{silent} answer(s) with no token counts"] if silent else [])
