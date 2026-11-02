"""agent-eval-gate: score an AI agent on accuracy, cost and latency; fail the build when it gets worse.

    agent-eval-gate record CASES --agent file.py:run --out runs/v1.json   call the agent once per case
    agent-eval-gate score  CASES runs/v1.json --prices prices.toml         score a recording (free, no calls)
    agent-eval-gate gate   CASES --baseline runs/v1.json --candidate runs/v2.json --prices prices.toml
    agent-eval-gate prove  CASES runs/v1.json                              find cases that cannot fail
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .core import EvalError, load_agent, load_cases, load_prices, load_recording, record, save_recording, score, unpriced_models
from .gate import Limits, compare, prove
from .report import html_page, markdown


def _scored(cases_path: str, recording_path: str, prices_path: str | None):
    cases = load_cases(cases_path)
    recording = load_recording(recording_path)
    prices = load_prices(prices_path)
    missing = unpriced_models(recording, prices)
    if missing:
        print(f"note: no price row for {', '.join(missing)}; their cost is reported as unknown, never as $0",
              file=sys.stderr)
    return cases, recording, score(cases, recording, prices)


def cmd_record(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    agent = load_agent(args.agent)
    recording = record(cases, agent, args.name or args.agent)
    save_recording(recording, args.out)
    errors = sum(1 for e in recording["cases"].values() if e["error"])
    print(f"recorded {len(cases)} case(s) from {recording['agent']} to {args.out}"
          + (f" ({errors} raised)" if errors else ""))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    _cases, _recording, result = _scored(args.cases, args.recording, args.prices)
    print(markdown(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result.summary(), indent=2) + "\n", encoding="utf-8")
    if args.html:
        Path(args.html).write_text(html_page([result], title=f"agent-eval-gate: {result.agent}"), encoding="utf-8")
    return 0 if result.passed == len(result.cases) or not args.strict else 1


def cmd_gate(args: argparse.Namespace) -> int:
    _c, _r, baseline = _scored(args.cases, args.baseline, args.prices)
    _c, _r, candidate = _scored(args.cases, args.candidate, args.prices)
    limits = Limits(args.max_accuracy_drop, args.max_cost_increase, args.max_p95_increase)
    result = compare(baseline, candidate, limits)
    print(f"gate: {'passed' if result.passed else 'FAILED'} ({candidate.agent} against {baseline.agent})")
    for line in result.lines:
        print(f"  {line}")
    if result.newly_failing:
        print(f"newly failing ({len(result.newly_failing)}):")
        for cid, reasons in result.newly_failing:
            print(f"  - {cid}: {'; '.join(reasons)}")
    if result.newly_passing:
        print(f"newly passing ({len(result.newly_passing)}): {', '.join(result.newly_passing)}")
    if args.html:
        page = html_page([baseline, candidate], result, title=f"{candidate.agent} against {baseline.agent}")
        Path(args.html).write_text(page, encoding="utf-8")
        print(f"report: {args.html}")
    return 0 if result.passed else 1


def cmd_prove(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)
    recording = load_recording(args.recording)
    found = prove(cases, recording, empty_only=args.empty_only)
    decoys = "an empty answer" + ("" if args.empty_only else " and on every other case's different answer")
    if not found:
        print(f"proved: every one of {len(cases)} case(s) fails on {decoys}")
        return 0
    print(f"{len(found)} of {len(cases)} case(s) passed on an answer that was not theirs:")
    for item in found:
        shown = item.passes_on[:2] + ([f"{len(item.passes_on) - 2} more"] if len(item.passes_on) > 2 else [])
        verdict = "cannot fail" if item.cannot_fail else "does not pin its answer down"
        print(f"  - {item.case_id} {verdict}: it passes on {', '.join(shown)}")
    print("Add a check that only the case's own right answer meets, then run prove again.")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-eval-gate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("record", help="call the agent once per case and save what it answered")
    p.add_argument("cases")
    p.add_argument("--agent", required=True, help="module:function or path/to/file.py:function")
    p.add_argument("--out", required=True)
    p.add_argument("--name", help="a label for this agent version (default: the --agent value)")
    p.set_defaults(fn=cmd_record)

    p = sub.add_parser("score", help="score a recording; no agent call, so it costs nothing")
    p.add_argument("cases")
    p.add_argument("recording")
    p.add_argument("--prices")
    p.add_argument("--json", help="write the summary as JSON here")
    p.add_argument("--html", help="write a one-file HTML report here")
    p.add_argument("--strict", action="store_true", help="exit 1 when any case fails")
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("gate", help="compare a candidate recording against a baseline")
    p.add_argument("cases")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--prices")
    p.add_argument("--max-accuracy-drop", type=float, default=0.0, help="percentage points (default 0)")
    p.add_argument("--max-cost-increase", type=float, default=10.0, help="percent per case (default 10)")
    p.add_argument("--max-p95-increase", type=float, default=20.0, help="percent (default 20)")
    p.add_argument("--html", help="write a one-file HTML report of both runs here")
    p.set_defaults(fn=cmd_gate)

    p = sub.add_parser("prove", help="find cases that pass on an empty or a wrong answer")
    p.add_argument("cases")
    p.add_argument("recording")
    p.add_argument("--empty-only", action="store_true",
                   help="only the empty-answer decoys (for suites where many cases share one right answer)")
    p.set_defaults(fn=cmd_prove)

    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except EvalError as exc:
        print(f"agent-eval-gate: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
