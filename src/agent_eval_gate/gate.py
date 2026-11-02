"""The gate (did the candidate get worse than the baseline?) and the proof (can each case fail?)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .core import Case, Score, run_checks


@dataclass
class Limits:
    max_accuracy_drop: float = 0.0       # percentage points
    max_cost_increase: float = 10.0      # percent, per case
    max_p95_increase: float = 20.0       # percent


@dataclass
class GateResult:
    passed: bool
    lines: list[str]
    newly_failing: list[tuple[str, list[str]]] = field(default_factory=list)
    newly_passing: list[str] = field(default_factory=list)


def _pct_change(before: float, after: float) -> float:
    if before == 0:
        return 0.0 if after == 0 else float("inf")
    return (after - before) / before * 100


def compare(baseline: Score, candidate: Score, limits: Limits) -> GateResult:
    lines: list[str] = []
    ok = True

    drop = (baseline.accuracy - candidate.accuracy) * 100
    acc_ok = drop <= limits.max_accuracy_drop + 1e-9
    ok &= acc_ok
    lines.append(f"{'ok  ' if acc_ok else 'FAIL'} accuracy   {baseline.accuracy:.1%} -> {candidate.accuracy:.1%} "
                 f"({-drop:+.1f} points; allowed {(-limits.max_accuracy_drop or 0):g})")

    before, after = baseline.cost_per_case, candidate.cost_per_case
    if before is None or after is None:
        ok = False
        lines.append("FAIL cost/case  unknown: a model has no row in the price table, or a case has no "
                     "current recorded answer")
    else:
        change = _pct_change(before, after)
        cost_ok = change <= limits.max_cost_increase + 1e-9
        ok &= cost_ok
        lines.append(f"{'ok  ' if cost_ok else 'FAIL'} cost/case  ${before:.6f} -> ${after:.6f} "
                     f"({change:+.0f}%; allowed +{limits.max_cost_increase:g}%)")

    p_before, p_after = baseline.latency(95), candidate.latency(95)
    change = _pct_change(p_before, p_after)
    lat_ok = change <= limits.max_p95_increase + 1e-9
    ok &= lat_ok
    lines.append(f"{'ok  ' if lat_ok else 'FAIL'} p95        {p_before:.0f} ms -> {p_after:.0f} ms "
                 f"({change:+.0f}%; allowed +{limits.max_p95_increase:g}%)")

    base = {c.id: c for c in baseline.cases}
    newly_failing = [(c.id, c.reasons) for c in candidate.cases if not c.passed and base.get(c.id) and base[c.id].passed]
    newly_passing = [c.id for c in candidate.cases if c.passed and base.get(c.id) and not base[c.id].passed]
    return GateResult(ok, lines, newly_failing, newly_passing)


@dataclass
class Vacuous:
    case_id: str
    passes_on: list[str]
    cannot_fail: bool   # passes on an empty answer: the strongest finding


EMPTY_DECOYS: list[tuple[str, Any]] = [("an empty answer", ""), ("no answer", None), ("an empty object", {})]


def prove(cases: list[Case], recording: dict, empty_only: bool = False) -> list[Vacuous]:
    """A check never seen failing proves nothing. Each case is scored against decoys:
    empty answers (a case that passes one cannot fail at all), then every OTHER case's recorded
    answer that differs from its own (a case that passes one does not pin its answer down)."""
    outputs = {cid: entry.get("output") for cid, entry in recording["cases"].items() if not entry.get("error")}
    found: list[Vacuous] = []
    for case in cases:
        empty = [label for label, decoy in EMPTY_DECOYS if not run_checks(case, decoy)]
        if empty:
            found.append(Vacuous(case.id, empty, True))
            continue
        if empty_only:
            continue
        own = outputs.get(case.id)
        others = [label for label, decoy in
                  ((f"the answer recorded for {other!r}", out) for other, out in outputs.items()
                   if other != case.id and out != own)
                  if not run_checks(case, decoy)]
        if others:
            found.append(Vacuous(case.id, others, False))
    return found
