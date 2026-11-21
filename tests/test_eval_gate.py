from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from agent_eval_gate.core import (Case, EvalError, Reply, load_agent, load_cases, load_prices, load_recording,
                                  percentile, record, run_checks, save_recording, score)
from agent_eval_gate.gate import EMPTY_DECOYS, Limits, compare, empty_decoys, prove
from agent_eval_gate.report import html_page, markdown

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "support_triage"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def case(checks, cid="c", value="x") -> Case:
    return Case(cid, value, tuple(checks))


def write_cases(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------- checks

def test_checks_read_dotted_paths_into_json_answers():
    c = case([{"type": "equals", "path": "a.b.0", "value": 3}])
    assert run_checks(c, {"a": {"b": [3]}}) == []
    assert run_checks(c, json.dumps({"a": {"b": [3]}})) == []
    assert run_checks(c, {"a": {"b": [4]}}) == ["equals a.b.0: expected 3, got 4"]
    assert "no field" in run_checks(c, {"a": {}})[0]
    assert "no field" in run_checks(c, "plain text")[0]


@pytest.mark.parametrize("check, good, bad", [
    ({"type": "contains", "value": ["refund", "today"]}, "A refund today", "A refund"),
    ({"type": "contains", "value": "Refund", "ignore_case": False}, "Refund", "refund"),
    ({"type": "not_contains", "value": "as an AI"}, "Happy to help", "As an AI, I"),
    ({"type": "regex", "value": r"^\d{4}-\d{2}$"}, "2026-11", "Nov 2026"),
    ({"type": "one_of", "value": ["a", "b"]}, "a", "c"),
    ({"type": "max_length", "value": 5}, "short", "longer"),
])
def test_each_check_passes_and_fails(check, good, bad):
    c = case([check])
    assert run_checks(c, good) == []
    assert run_checks(c, bad) != []


def test_a_case_file_that_cannot_work_says_why(tmp_path):
    with pytest.raises(EvalError, match="no checks"):
        load_cases(write_cases(tmp_path, [{"id": "a", "input": 1, "checks": []}]))
    with pytest.raises(EvalError, match="used twice"):
        load_cases(write_cases(tmp_path, [{"id": "a", "input": 1, "checks": [{"type": "equals", "value": 1}]}] * 2))
    with pytest.raises(EvalError, match="has no value"):
        load_cases(write_cases(tmp_path, [{"id": "a", "input": 1, "checks": [{"type": "equals"}]}]))
    with pytest.raises(EvalError, match="path must be text"):
        load_cases(write_cases(tmp_path, [{"id": "a", "input": 1, "checks": [{"type": "equals", "value": 1,
                                                                               "path": 3}]}]))
    with pytest.raises(EvalError, match="needs text"):
        load_cases(write_cases(tmp_path, [{"id": "a", "input": 1, "checks": [{"type": "regex", "value": 5}]}]))
    with pytest.raises(EvalError, match="unknown check type 'similar'"):
        load_cases(write_cases(tmp_path, [{"id": "a", "input": 1, "checks": [{"type": "similar"}]}]))
    with pytest.raises(EvalError, match="does not compile"):
        load_cases(write_cases(tmp_path, [{"id": "a", "input": 1, "checks": [{"type": "regex", "value": "("}]}]))
    with pytest.raises(EvalError, match="whole number"):
        load_cases(write_cases(tmp_path, [{"id": "a", "input": 1, "checks": [{"type": "max_length", "value": "9"}]}]))
    with pytest.raises(EvalError, match="not a JSON object"):
        load_cases(write_cases(tmp_path, [[1, 2]]))


def test_an_answer_that_is_not_json_is_kept_as_text(tmp_path):
    from agent_eval_gate.core import load_recording, save_recording

    class SdkObject:
        def __str__(self):
            return "ok from an sdk object"

    c = case([{"type": "contains", "value": "ok"}])
    save_recording(record([c], lambda _v: SdkObject(), "a"), tmp_path / "r.json")
    assert score([c], load_recording(tmp_path / "r.json"), {}).cases[0].passed


# ------------------------------------------------------------- recording and scoring

def test_record_keeps_usage_and_turns_an_exception_into_a_failing_case():
    cases = [case([{"type": "equals", "value": "ok"}], "good"), case([{"type": "equals", "value": "ok"}], "bad")]

    def agent(value):
        if value == "boom":
            raise RuntimeError("model timed out")
        return Reply("ok", "m", 10, 5)

    cases[1] = Case("bad", "boom", cases[1].checks)
    rec = record(cases, agent, "a")
    assert rec["cases"]["good"]["input_tokens"] == 10
    assert rec["cases"]["good"]["latency_source"] == "wall clock"
    assert rec["cases"]["bad"]["error"] == "RuntimeError: model timed out"
    result = score(cases, rec, {"m": (1.0, 2.0)})
    assert [c.passed for c in result.cases] == [True, False]
    assert result.cases[0].cost_usd == pytest.approx((10 * 1.0 + 5 * 2.0) / 1_000_000)


def test_a_changed_input_is_never_scored_against_an_old_answer():
    c = case([{"type": "equals", "value": "ok"}], "a", "first")
    rec = record([c], lambda _v: "ok", "agent")
    changed = Case("a", "second", c.checks)
    assert "record again" in score([changed], rec, {}).cases[0].reasons[0]


def test_an_unpriced_model_is_unknown_cost_never_free():
    c = case([{"type": "equals", "value": "ok"}])
    rec = record([c], lambda _v: {"output": "ok", "model": "mystery", "input_tokens": 9, "output_tokens": 9}, "a")
    result = score([c], rec, {})
    assert result.cases[0].cost_usd is None and result.cost_total is None


def test_percentile_is_nearest_rank():
    values = list(range(1, 21))
    assert percentile(values, 95) == 19
    assert percentile(values, 50) == 10
    assert percentile([], 95) == 0.0


def test_prices_need_both_columns(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text('[models."m"]\ninput_per_million = 1\n', encoding="utf-8")
    with pytest.raises(EvalError, match="output_per_million"):
        load_prices(path)


def test_load_agent_from_a_file_and_its_neighbors():
    agent = load_agent(f"{EXAMPLE / 'agent_v2.py'}:run")  # agent_v2 imports agent_v1 beside it
    assert agent({"subject": "Invoice", "body": "copy please"})["output"]["category"] == "billing"
    with pytest.raises(EvalError, match="no function"):
        load_agent(f"{EXAMPLE / 'agent_v1.py'}:missing")


# ------------------------------------------------------------- the gate and the proof

def example_scores():
    cases = load_cases(EXAMPLE / "cases.jsonl")
    prices = load_prices(EXAMPLE / "prices.toml")
    v1 = record(cases, load_agent(f"{EXAMPLE / 'agent_v1.py'}:run"), "v1")
    v2 = record(cases, load_agent(f"{EXAMPLE / 'agent_v2.py'}:run"), "v2")
    return cases, v1, score(cases, v1, prices), score(cases, v2, prices)


def test_the_gate_names_what_broke():
    _cases, _v1, base, cand = example_scores()
    result = compare(base, cand, Limits())
    assert not result.passed
    assert [cid for cid, _ in result.newly_failing] == ["crash-or-we-cancel", "third-time-invoice"]
    assert result.newly_passing == ["is-there-a-way"]
    assert sum(line.startswith("FAIL") for line in result.lines) == 4  # accuracy, cost, p95, newly failing


def test_the_gate_passes_a_run_against_itself():
    _cases, _v1, base, _cand = example_scores()
    assert compare(base, base, Limits()).passed


def test_loose_limits_let_a_tradeoff_through():
    _cases, _v1, base, cand = example_scores()
    loose = Limits(max_accuracy_drop=10, max_cost_increase=1000, max_p95_increase=100, max_newly_failing=2)
    assert compare(base, cand, loose).passed


def test_a_swap_that_holds_accuracy_still_fails_on_the_case_that_broke():
    cases = [case([{"type": "equals", "value": "ok"}], "a", "x"), case([{"type": "equals", "value": "ok"}], "b", "y")]

    def agent(right):
        return lambda v: {"output": "ok" if v == right else "no", "model": "m", "input_tokens": 1,
                          "output_tokens": 1, "latency_ms": 100}

    before = score(cases, record(cases, agent("x"), "v1"), {"m": (1, 1)})
    after = score(cases, record(cases, agent("y"), "v2"), {"m": (1, 1)})
    assert before.accuracy == after.accuracy == 0.5
    result = compare(before, after, Limits())
    assert not result.passed and [cid for cid, _ in result.newly_failing] == ["a"]
    assert result.lines[-1].startswith("FAIL newly failing 1")


def test_prove_finds_the_weak_cases_and_spares_the_strong_ones():
    cases, v1, _base, _cand = example_scores()
    found = {v.case_id: v for v in prove(cases, v1)}
    assert set(found) == {"is-there-a-way", "api-500", "no-robot-talk"}
    # an empty reply has no robot talk in it: the case cannot fail until it also asks for a reply
    assert {cid for cid, v in found.items() if v.cannot_fail} == {"no-robot-talk"}
    weak = case([{"type": "not_contains", "value": "as an AI"}], "weak")
    assert prove([weak], {"cases": {}})[0].cannot_fail


def test_the_html_report_escapes_answers():
    c = case([{"type": "equals", "value": "ok"}])
    rec = record([c], lambda _v: "<script>alert(1)</script>", "agent <b>")
    page = html_page([score([c], rec, {})])
    assert "<script>alert" not in page and "&lt;script&gt;" in page and "agent &lt;b&gt;" in page


# ------------------------------------------------------------- the command line, end to end

def cli(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_eval_gate", *args], cwd=cwd, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW)


def test_the_command_line_end_to_end(tmp_path):
    cases = str(EXAMPLE / "cases.jsonl")
    prices = str(EXAMPLE / "prices.toml")
    for version in ("v1", "v2"):
        result = cli("record", cases, "--agent", f"{EXAMPLE / f'agent_{version}.py'}:run",
                     "--out", str(tmp_path / f"{version}.json"), cwd=tmp_path)
        assert result.returncode == 0, result.stderr
    scored = cli("score", cases, str(tmp_path / "v1.json"), "--prices", prices, "--json", str(tmp_path / "s.json"),
                 cwd=tmp_path)
    assert scored.returncode == 0 and "90.9%" in scored.stdout
    assert json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))["passed"] == 10
    gated = cli("gate", cases, "--baseline", str(tmp_path / "v1.json"), "--candidate", str(tmp_path / "v2.json"),
                "--prices", prices, "--html", str(tmp_path / "r.html"), cwd=tmp_path)
    assert gated.returncode == 1 and "crash-or-we-cancel" in gated.stdout
    assert (tmp_path / "r.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    assert cli("prove", cases, str(tmp_path / "v1.json"), cwd=tmp_path).returncode == 1
    empty_only = cli("prove", cases, str(tmp_path / "v1.json"), "--empty-only", cwd=tmp_path)
    assert empty_only.returncode == 1 and "no-robot-talk cannot fail" in empty_only.stdout
    broken = cli("score", cases, str(tmp_path / "absent.json"), cwd=tmp_path)
    assert broken.returncode == 2 and "cannot read the recording" in broken.stderr
    nested = cli("score", cases, str(tmp_path / "v1.json"), "--html", str(tmp_path / "new" / "dir" / "r.html"),
                 cwd=tmp_path)
    assert nested.returncode == 0 and (tmp_path / "new" / "dir" / "r.html").exists()
    missing_module = cli("record", cases, "--agent", "no_such_module:run", "--out", str(tmp_path / "x.json"),
                         cwd=tmp_path)
    assert missing_module.returncode == 2 and "cannot load the agent" in missing_module.stderr


# ------------------------------------------------------------- shapes the close review found

def test_an_answer_with_no_token_counts_has_an_unknown_cost():
    c = case([{"type": "equals", "value": "ok"}])
    rec = record([c], lambda _v: {"output": "ok", "model": "m"}, "a")
    assert score([c], rec, {"m": (1.0, 1.0)}).cases[0].cost_usd is None


def test_equals_is_type_strict():
    c = case([{"type": "equals", "value": True}])
    assert run_checks(c, True) == []
    assert run_checks(c, 1) != []
    assert run_checks(case([{"type": "equals", "value": 1}]), True) != []


def test_text_checks_search_string_values_not_the_json_encoding():
    c = case([{"type": "contains", "value": "said \"hi\""}])
    assert run_checks(c, {"reply": "she said \"hi\" twice"}) == []
    banned = case([{"type": "not_contains", "value": "reply"}])
    assert run_checks(banned, {"text": "fine"}) == []


def test_a_latency_that_is_not_a_number_is_a_failing_case():
    c = case([{"type": "equals", "value": "ok"}])
    rec = record([c], lambda _v: {"output": "ok", "latency_ms": "fast"}, "a")
    assert rec["cases"]["c"]["error"].startswith("ValueError")


def test_record_hands_over_the_recording_after_every_case():
    cases = [case([{"type": "equals", "value": "ok"}], cid) for cid in ("a", "b", "c")]
    seen: list[int] = []
    record(cases, lambda _v: "ok", "a", on_case=lambda rec: seen.append(len(rec["cases"])))
    assert seen == [1, 2, 3]


def test_prove_fills_the_fields_a_check_reads():
    c = case([{"type": "not_contains", "path": "reply", "value": "as an AI"}])
    found = prove([c], {"cases": {}})
    assert found and found[0].cannot_fail and any("reply" in label for label in found[0].passes_on)


# ------------------------------------------------------------- numbers an agent reports

def answer_with(**fields) -> dict:
    return {"output": "ok", "model": "m", "input_tokens": 1, "output_tokens": 1, **fields}


def recorded(answer) -> dict:
    return record([case([{"type": "equals", "value": "ok"}])], lambda _v: answer, "a")["cases"]["c"]


@pytest.mark.parametrize("count", ["300", 2.9, -5, float("nan"), True],
                         ids=["text", "fractional", "negative", "nan", "true"])
def test_a_token_count_that_is_not_a_whole_number_of_zero_or_more_is_a_failing_case(count):
    entry = recorded(answer_with(input_tokens=count))
    assert (entry["error"] or "").startswith("ValueError: input_tokens must be a whole number of zero or more")
    assert entry["input_tokens"] is None


def test_output_tokens_and_a_reply_object_meet_the_same_rule():
    assert "output_tokens must be a whole number" in (recorded(answer_with(output_tokens=-1))["error"] or "")
    assert "input_tokens must be a whole number" in (recorded(Reply("ok", "m", 2.5, 1))["error"] or "")


@pytest.mark.parametrize("latency", [-3, float("inf"), float("nan"), "500", True],
                         ids=["negative", "infinite", "nan", "text", "true"])
def test_a_latency_that_is_not_a_finite_number_of_zero_or_more_is_a_failing_case(latency):
    entry = recorded(answer_with(latency_ms=latency))
    assert (entry["error"] or "").startswith("ValueError: latency_ms must be a finite number of zero or more")


def test_whole_token_counts_and_finite_latencies_are_kept():
    entry = recorded(answer_with(input_tokens=300.0, output_tokens=0, latency_ms=12.5))
    assert entry["error"] is None
    assert (entry["input_tokens"], entry["output_tokens"], entry["latency_ms"]) == (300, 0, 12.5)
    assert type(entry["input_tokens"]) is int
    assert recorded(answer_with(latency_ms=0))["latency_ms"] == 0


@pytest.mark.parametrize("count", [Decimal("300"), Decimal("300.0"), Fraction(300)],
                         ids=["decimal", "decimal-with-a-point", "fraction"])
def test_a_whole_count_of_any_number_type_is_kept_as_an_int(count):
    entry = recorded(answer_with(input_tokens=count, output_tokens=count))
    assert entry["error"] is None
    assert (entry["input_tokens"], entry["output_tokens"]) == (300, 300)
    assert type(entry["input_tokens"]) is int and type(entry["output_tokens"]) is int


@pytest.mark.parametrize("latency", [Decimal("500"), Fraction(1, 2)], ids=["decimal", "fraction"])
def test_a_finite_latency_of_any_number_type_is_kept_as_a_float(latency):
    entry = recorded(answer_with(latency_ms=latency))
    assert entry["error"] is None and entry["latency_ms"] == float(latency)
    assert type(entry["latency_ms"]) is float


def test_numpy_numbers_meet_the_same_rule():
    np = pytest.importorskip("numpy")
    entry = recorded(answer_with(input_tokens=np.float32(300.0), output_tokens=np.int64(2),
                                 latency_ms=np.float32(12.5)))
    assert entry["error"] is None
    assert (entry["input_tokens"], entry["output_tokens"], entry["latency_ms"]) == (300, 2, 12.5)
    assert "input_tokens must be a whole number" in (recorded(answer_with(input_tokens=np.float32(2.5)))["error"] or "")


@pytest.mark.parametrize("count", [Decimal("2.5"), Decimal("-1"), Decimal("NaN"), Decimal("Infinity"), Fraction(5, 2)],
                         ids=["decimal-fraction", "decimal-negative", "decimal-nan", "decimal-infinite", "fraction"])
def test_a_count_of_another_number_type_meets_the_same_rule(count):
    entry = recorded(answer_with(input_tokens=count))
    assert (entry["error"] or "").startswith("ValueError: input_tokens must be a whole number of zero or more")


@pytest.mark.parametrize("latency", [Decimal("-1"), Decimal("NaN"), Decimal("1e400"), 10 ** 400, 10 ** 5000],
                         ids=["decimal-negative", "decimal-nan", "decimal-too-large", "too-large", "too-long-to-print"])
def test_a_latency_of_another_number_type_or_too_large_for_a_float_meets_the_same_rule(latency):
    error = recorded(answer_with(latency_ms=latency))["error"] or ""
    assert error.startswith("ValueError: latency_ms must be a finite number of zero or more")
    assert len(error) < 200  # the value is shown cut short, never as thousands of digits


def test_a_case_failed_by_a_refused_number_keeps_no_answer_and_its_cost_is_unknown():
    c = case([{"type": "equals", "value": "ok"}])
    entry = recorded(answer_with(input_tokens=3, output_tokens=2, latency_ms="500"))
    assert (entry["output"], entry["input_tokens"], entry["output_tokens"]) == (None, None, None)
    scored = score([c], {"cases": {"c": entry}}, {"m": (1_000_000.0, 1_000_000.0)})
    assert (scored.cases[0].passed, scored.cases[0].cost_usd, scored.cost_total) == (False, None, None)


# ------------------------------------------------------------- the cost of a case whose numbers were refused

PRICED = {"m": (1.0, 1.0)}


def test_a_case_whose_token_count_was_refused_has_an_unknown_cost_in_the_score_and_the_report():
    c = case([{"type": "equals", "value": "ok"}])
    result = score([c], record([c], lambda _v: answer_with(input_tokens="300", output_tokens=20), "a"), PRICED)
    assert (result.cases[0].passed, result.cases[0].cost_usd, result.cost_total, result.cost_per_case) == (
        False, None, None, None)
    assert (result.summary()["cost_total_usd"], result.summary()["cost_per_case_usd"]) == (None, None)
    assert "| unknown | unknown |" in markdown(result)
    page = html_page([result])
    assert "<b>unknown</b><span>cost per case" in page and "<span>fail</span> unknown," in page


def test_a_candidate_that_starts_returning_bad_token_counts_never_reads_cheaper():
    cases = [case([{"type": "equals", "value": "ok"}], cid, cid) for cid in ("a", "b")]
    base = score(cases, record(cases, lambda _v: answer_with(input_tokens=1000, output_tokens=1000, latency_ms=100),
                               "v1"), PRICED)
    cand = score(cases, record(cases, lambda v: answer_with(input_tokens="1000" if v == "b" else 1000,
                                                            output_tokens=1000, latency_ms=100), "v2"), PRICED)
    result = compare(base, cand, Limits(max_accuracy_drop=100, max_newly_failing=2))  # accuracy loosened on purpose
    assert not result.passed
    assert result.lines[1].startswith("FAIL cost/case  unknown")


@pytest.mark.parametrize("stop", [RuntimeError("model timed out"), SystemExit("quota exceeded"),
                                  asyncio.CancelledError()], ids=["raised", "exited", "canceled"])
def test_a_case_whose_agent_raised_still_adds_nothing_to_the_cost(stop):
    def agent(value):
        if value == "b":
            raise stop
        return answer_with()

    result = score(THREE, record(THREE, agent, "a"), PRICED)
    assert [c.cost_usd for c in result.cases] == [pytest.approx(2e-6), 0.0, pytest.approx(2e-6)]
    assert result.cost_total == pytest.approx(4e-6)


@pytest.mark.parametrize("count", [10 ** 400, Decimal("1e400")], ids=["int", "decimal"])
def test_a_token_count_too_large_for_a_float_scores_as_an_unknown_cost(tmp_path, count):
    c = case([{"type": "equals", "value": "ok"}])
    rec = record([c], lambda _v: answer_with(input_tokens=count), "a")
    result = score([c], rec, PRICED)
    assert (result.cases[0].passed, result.cases[0].cost_usd, result.cost_total) == (True, None, None)
    save_recording(rec, tmp_path / "r.json")
    prices = tmp_path / "p.toml"
    prices.write_text('[models."m"]\ninput_per_million = 1\noutput_per_million = 1\n', encoding="utf-8")
    cases = write_cases(tmp_path, [{"id": "c", "input": "x", "checks": [{"type": "equals", "value": "ok"}]}])
    scored = cli("score", str(cases), str(tmp_path / "r.json"), "--prices", str(prices), cwd=tmp_path)
    assert scored.returncode == 0, scored.stderr
    assert "| unknown | unknown |" in scored.stdout


def test_a_cost_too_large_for_a_float_is_unknown_never_infinite():
    c = case([{"type": "equals", "value": "ok"}])
    rec = record([c], lambda _v: answer_with(input_tokens=10 ** 300), "a")
    assert score([c], rec, {"m": (1e300, 1.0)}).cases[0].cost_usd is None


def test_a_refused_number_reads_the_agent_returned_and_a_raise_still_reads_the_agent_raised():
    def agent(value):
        if value == "a":
            return answer_with(input_tokens="300")
        if value == "b":
            raise RuntimeError("model timed out")
        return answer_with(latency_ms=float("nan"))

    assert [c.reasons for c in score(THREE, record(THREE, agent, "x"), PRICED).cases] == [
        ["the agent returned a refused number: input_tokens must be a whole number of zero or more, got '300'"],
        ["the agent raised RuntimeError: model timed out"],
        ["the agent returned a refused number: latency_ms must be a finite number of zero or more that a float can "
         "hold, got nan"],
    ]


def test_a_hand_edited_refused_entry_whose_error_is_not_text_still_scores(tmp_path):
    c = case([{"type": "equals", "value": "ok"}], "double-charge")
    rec = record([c], lambda _v: answer_with(input_tokens="300"), "a")
    rec["cases"]["double-charge"].update(error=5, refused=True)
    save_recording(rec, tmp_path / "r.json")
    result = score([c], load_recording(tmp_path / "r.json"), PRICED)
    assert (result.cases[0].reasons, result.cases[0].cost_usd) == (["the agent returned a refused number: 5"], None)


def test_a_refused_case_in_a_recording_made_before_refusals_were_marked_still_reads_as_raised_at_0():
    rec = record(THREE, lambda v: answer_with(input_tokens="300") if v == "b" else answer_with(), "a")
    rec["cases"]["b"].pop("refused", None)  # what an earlier version wrote: no mark
    result = score(THREE, rec, PRICED)
    assert result.cases[1].cost_usd == 0.0
    assert result.cases[1].reasons == [
        "the agent raised ValueError: input_tokens must be a whole number of zero or more, got '300'"]


# ------------------------------------------------------------- a recording's numbers, when it loads

def saved(tmp_path: Path, field: str, value) -> Path:
    """A recording of one case, "double-charge", with one value set by hand."""
    c = case([{"type": "equals", "value": "ok"}], "double-charge")
    rec = record([c], lambda _v: answer_with(latency_ms=10), "a")
    rec["cases"]["double-charge"][field] = value
    save_recording(rec, tmp_path / "r.json")
    return tmp_path / "r.json"


@pytest.mark.parametrize("field, value", [
    ("input_tokens", "300"), ("input_tokens", -5), ("input_tokens", 2.9), ("output_tokens", True),
    ("latency_ms", "500"), ("latency_ms", -3), ("latency_ms", float("nan")), ("latency_ms", float("inf")),
], ids=["text-count", "negative-count", "fractional-count", "true-count",
        "text-latency", "negative-latency", "nan-latency", "infinite-latency"])
def test_a_recording_with_a_number_that_breaks_the_rules_is_refused_naming_the_case(tmp_path, field, value):
    with pytest.raises(EvalError, match=f"case 'double-charge': {field} must be"):
        load_recording(saved(tmp_path, field, value))


def test_a_recording_with_whole_counts_finite_latencies_and_errors_still_loads(tmp_path):
    for field, value in [("input_tokens", 300.0), ("output_tokens", None), ("latency_ms", 0), ("latency_ms", None)]:
        assert load_recording(saved(tmp_path, field, value))["cases"]["double-charge"][field] == value
    errors = record(THREE, lambda v: 1 / 0 if v == "b" else answer_with(), "a")
    save_recording(errors, tmp_path / "e.json")
    assert load_recording(tmp_path / "e.json")["cases"]["b"]["error"] == "ZeroDivisionError: division by zero"


def test_a_recording_whose_case_or_whole_file_is_not_an_object_is_refused(tmp_path):
    path = saved(tmp_path, "input_tokens", 1)
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["cases"]["double-charge"] = ["ok"]
    path.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(EvalError, match="case 'double-charge' is not a JSON object"):
        load_recording(path)
    (tmp_path / "list.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(EvalError, match="not a recording this version can read"):
        load_recording(tmp_path / "list.json")


def test_score_gate_and_prove_exit_2_on_a_recording_with_a_refused_number(tmp_path):
    path = saved(tmp_path, "input_tokens", -5)
    cases = write_cases(tmp_path, [{"id": "double-charge", "input": "x", "checks": [{"type": "equals", "value": "ok"}]}])
    for args in (["score", str(cases), str(path)], ["gate", str(cases), "--baseline", str(path), "--candidate", str(path)],
                 ["prove", str(cases), str(path)]):
        result = cli(*args, cwd=tmp_path)
        assert result.returncode == 2, (args[0], result.stdout, result.stderr)
        assert result.stderr.strip().splitlines() == [
            f"agent-eval-gate: {path}: case 'double-charge': input_tokens must be a whole number of zero or more, got -5"]


# ------------------------------------------------------------- an agent that exits or is canceled

def agent_file(tmp_path: Path, body: str) -> str:
    path = tmp_path / "agent_under_test.py"
    path.write_text(body, encoding="utf-8")
    return f"{path}:run"


EXITS_AS_IT_LOADS = 'import sys\nsys.exit("missing API key")\n\ndef run(v):\n    return v\n'
THREE = [case([{"type": "equals", "value": "ok"}], cid, cid) for cid in ("a", "b", "c")]


def test_an_agent_that_calls_sys_exit_as_it_loads_will_not_load(tmp_path):
    with pytest.raises(EvalError, match="cannot load the agent .*SystemExit: missing API key"):
        load_agent(agent_file(tmp_path, EXITS_AS_IT_LOADS))


def test_an_agent_canceled_as_it_loads_will_not_load(tmp_path):
    with pytest.raises(EvalError, match="cannot load the agent .*CancelledError"):
        load_agent(agent_file(tmp_path, "import asyncio\nraise asyncio.CancelledError()\n\ndef run(v):\n    return v\n"))


def test_the_command_line_exits_2_for_an_agent_that_calls_sys_exit_as_it_loads(tmp_path):
    result = cli("record", str(EXAMPLE / "cases.jsonl"), "--agent", agent_file(tmp_path, EXITS_AS_IT_LOADS),
                 "--out", str(tmp_path / "r.json"), cwd=tmp_path)
    assert result.returncode == 2, result.stderr
    assert "cannot load the agent" in result.stderr and "missing API key" in result.stderr


def test_an_agent_that_calls_sys_exit_on_a_case_fails_that_case_and_the_run_goes_on():
    def agent(value):
        if value == "b":
            sys.exit("quota exceeded")
        return "ok"

    rec = record(THREE, agent, "a")
    assert [rec["cases"][cid]["error"] for cid in ("a", "b", "c")] == [None, "SystemExit: quota exceeded", None]
    assert [c.passed for c in score(THREE, rec, {}).cases] == [True, False, True]


def test_an_agent_canceled_on_a_case_fails_that_case_and_the_run_goes_on():
    def agent(value):
        if value == "b":
            raise asyncio.CancelledError()
        return "ok"

    rec = record(THREE, agent, "a")
    assert (rec["cases"]["b"]["error"] or "").startswith("CancelledError")
    assert [rec["cases"][cid]["error"] for cid in ("a", "c")] == [None, None]


def test_an_exit_or_a_cancel_with_no_message_is_named_with_no_empty_reason(tmp_path):
    def agent(value):
        if value == "b":
            sys.exit()
        if value == "c":
            raise asyncio.CancelledError()
        return "ok"

    rec = record(THREE, agent, "a")
    assert [rec["cases"][cid]["error"] for cid in ("b", "c")] == ["SystemExit", "CancelledError"]
    with pytest.raises(EvalError) as raised:
        load_agent(agent_file(tmp_path, "import sys\nsys.exit()\n\ndef run(v):\n    return v\n"))
    assert str(raised.value).endswith(": SystemExit")


def test_the_command_line_records_past_an_agent_that_calls_sys_exit_on_a_case(tmp_path):
    cases = write_cases(tmp_path, [{"id": cid, "input": cid, "checks": [{"type": "equals", "value": "ok"}]}
                                   for cid in ("a", "b", "c")])
    spec = agent_file(tmp_path, 'import sys\n\ndef run(v):\n    if v == "b":\n        sys.exit("quota exceeded")\n'
                                '    return "ok"\n')
    result = cli("record", str(cases), "--agent", spec, "--out", str(tmp_path / "r.json"), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "recorded 3 case(s)" in result.stdout and "(1 raised)" in result.stdout


def test_ctrl_c_still_stops_the_run(tmp_path):
    def agent(value):
        if value == "b":
            raise KeyboardInterrupt
        return "ok"

    saved: list[list[str]] = []
    with pytest.raises(KeyboardInterrupt):
        record(THREE, agent, "a", on_case=lambda rec: saved.append(list(rec["cases"])))
    assert saved == [["a"]]  # the cases answered before it are kept
    with pytest.raises(KeyboardInterrupt):
        load_agent(agent_file(tmp_path, "raise KeyboardInterrupt\n\ndef run(v):\n    return v\n"))


def test_any_other_exception_that_is_not_an_exception_still_stops_the_run(tmp_path):
    def agent(value):
        if value == "b":
            raise GeneratorExit
        return "ok"

    with pytest.raises(GeneratorExit):
        record(THREE, agent, "a")
    with pytest.raises(GeneratorExit):
        load_agent(agent_file(tmp_path, "raise GeneratorExit\n\ndef run(v):\n    return v\n"))


def record_three(tmp_path: Path, body: str) -> subprocess.CompletedProcess:
    cases = write_cases(tmp_path, [{"id": cid, "input": cid, "checks": [{"type": "equals", "value": "ok"}]}
                                   for cid in ("a", "b", "c")])
    return cli("record", str(cases), "--agent", agent_file(tmp_path, body), "--out", str(tmp_path / "r.json"),
               cwd=tmp_path)


@pytest.mark.parametrize("body, last_line, kept", [
    ('def run(v):\n    if v == "b":\n        raise GeneratorExit\n    return "ok"\n', "GeneratorExit", ["a"]),
    ("raise GeneratorExit\n\ndef run(v):\n    return v\n", "GeneratorExit", None),
    ('class Halt(BaseException):\n    pass\n\ndef run(v):\n    if v == "b":\n        raise Halt("stop here")\n'
     '    return "ok"\n', "Halt: stop here", ["a"]),
], ids=["generator-exit-on-a-case", "generator-exit-as-it-loads", "its-own-base-exception"])
def test_the_command_line_exits_2_with_the_traceback_for_any_other_exception_that_is_not_an_exception(
        tmp_path, body, last_line, kept):
    result = record_three(tmp_path, body)
    assert result.returncode == 2, result.stderr  # never 1, the code kept for "the gate found a problem"
    assert result.stderr.startswith("Traceback (most recent call last):")
    assert result.stderr.strip().splitlines()[-1].endswith(last_line)
    out = tmp_path / "r.json"
    if kept is None:
        assert not out.exists()
    else:  # the cases answered before it are kept
        assert list(json.loads(out.read_text(encoding="utf-8"))["cases"]) == kept


def test_ctrl_c_from_an_agent_keeps_python_s_own_exit_through_the_command_line(tmp_path):
    result = record_three(tmp_path, "def run(v):\n    raise KeyboardInterrupt\n")
    assert result.returncode not in (0, 1, 2), result.returncode
    assert result.stderr.strip().splitlines()[-1] == "KeyboardInterrupt"


def unprintable(raises: type[BaseException]) -> type[Exception]:
    """An exception whose `__str__` raises `raises`."""
    class UnprintableError(Exception):
        def __str__(self):
            raise raises("this error cannot be printed")
    return UnprintableError


@pytest.mark.parametrize("raises", [RuntimeError, GeneratorExit], ids=["runtime-error", "generator-exit"])
def test_an_exception_that_cannot_be_printed_is_named_by_its_class_and_the_run_goes_on(raises):
    error = unprintable(raises)

    def agent(value):
        if value == "b":
            raise error()
        return "ok"

    rec = record(THREE, agent, "a")
    assert [rec["cases"][cid]["error"] for cid in ("a", "b", "c")] == [None, "UnprintableError", None]


def test_ctrl_c_while_an_error_is_printed_still_stops_the_run():
    error = unprintable(KeyboardInterrupt)

    def agent(_value):
        raise error()

    with pytest.raises(KeyboardInterrupt):
        record(THREE, agent, "a")


@pytest.mark.parametrize("raises", ["RuntimeError", "GeneratorExit"])
def test_an_agent_that_raises_an_unprintable_exception_as_it_loads_exits_2_with_its_name(tmp_path, raises):
    result = record_three(tmp_path, 'class Unprintable(Exception):\n    def __str__(self):\n'
                                    f'        raise {raises}("no")\n\nraise Unprintable()\n')
    assert result.returncode == 2, result.stderr  # never 1, the code kept for "the gate found a problem"
    assert result.stderr.strip().splitlines() == ["agent-eval-gate: Unprintable"]


# ------------------------------------------------------------- the empty answers prove builds

def filled(c: Case) -> list:
    """The empty answers that hold the fields the case's checks read, one per empty value."""
    return [decoy for _label, decoy in empty_decoys(c)[len(EMPTY_DECOYS):]]


def not_contains(*paths: str) -> Case:
    return case([{"type": "not_contains", "path": path, "value": "x"} for path in paths])


def test_prove_fills_two_fields_under_one_list_item():
    c = not_contains("actions.0.id", "actions.0.name")
    assert filled(c)[0] == {"actions": [{"id": "", "name": ""}]}
    assert prove([c], {"cases": {}}, empty_only=True)[0].cannot_fail


def test_prove_fills_a_path_into_a_list_at_the_top_of_the_answer():
    c = not_contains("0.name")
    assert filled(c) == [[{"name": ""}], [{"name": None}], [{"name": {}}], [{"name": []}]]
    assert prove([c], {"cases": {}}, empty_only=True)[0].cannot_fail


def test_prove_fills_two_items_of_one_list():
    c = not_contains("actions.0.id", "actions.1.id")
    assert all(run_checks(case([{"type": "equals", "path": p, "value": ""}]), filled(c)[0]) == []
               for p in ("actions.0.id", "actions.1.id"))
    assert prove([c], {"cases": {}}, empty_only=True)[0].cannot_fail


def test_prove_fills_a_list_path_and_a_key_at_the_top_of_one_answer():
    c = not_contains("0.name", "category")
    assert prove([c], {"cases": {}}, empty_only=True)[0].cannot_fail


def test_prove_fills_a_checked_field_beneath_another_for_every_empty_value():
    c = case([{"type": "not_contains", "path": "reply", "value": "sorry"},
              {"type": "one_of", "path": "reply.text", "value": [None, "sent"]}])
    assert filled(c) == [{"reply": {"text": value}} for value in ("", None, {}, [])]
    assert prove([c], {"cases": {}}, empty_only=True)[0].cannot_fail


def test_prove_names_only_the_fields_left_empty():
    c = not_contains("reply", "reply.text", "category")
    assert empty_decoys(c)[len(EMPTY_DECOYS)][0] == "an answer whose category, reply.text are empty text"


# ------------------------------------------------------------- a price that is not a finite number of zero or more

def price_table(tmp_path: Path, input_price: str, output_price: str = "1") -> Path:
    path = tmp_path / "prices.toml"
    path.write_text(f'[models."m"]\ninput_per_million = {input_price}\noutput_per_million = {output_price}\n',
                    encoding="utf-8")
    return path


@pytest.mark.parametrize("column, value, shown", [
    ("input_per_million", "inf", "inf"), ("input_per_million", "nan", "nan"), ("output_per_million", "-1", "-1"),
    ("input_per_million", "-inf", "-inf"), ("input_per_million", "1e400", "inf"),
    ("output_per_million", '"nan"', "'nan'"),
], ids=["infinite", "nan", "negative", "negative-infinite", "too-large-for-a-float", "nan-as-text"])
def test_a_price_that_is_not_a_finite_number_of_zero_or_more_is_refused_when_the_table_loads(
        tmp_path, column, value, shown):
    row = {"input_per_million": "1", "output_per_million": "1", column: value}
    with pytest.raises(EvalError) as raised:
        load_prices(price_table(tmp_path, row["input_per_million"], row["output_per_million"]))
    assert str(raised.value) == f"price row 'm': {column} must be a finite number of zero or more, got {shown}"


@pytest.mark.parametrize("value", ["inf", "nan", "-1"], ids=["infinite", "nan", "negative"])
def test_score_and_gate_exit_2_naming_the_model_on_a_price_that_is_not_a_finite_number_of_zero_or_more(
        tmp_path, value):
    cases = write_cases(tmp_path, [{"id": cid, "input": cid, "checks": [{"type": "equals", "value": "ok"}]}
                                   for cid in ("a", "b", "c")])
    rec = tmp_path / "r.json"
    save_recording(record(THREE, lambda _v: answer_with(), "a"), rec)
    prices = str(price_table(tmp_path, value))
    for args in (["score", str(cases), str(rec), "--prices", prices],
                 ["gate", str(cases), "--baseline", str(rec), "--candidate", str(rec), "--prices", prices]):
        result = cli(*args, cwd=tmp_path)
        assert result.returncode == 2, (args[0], result.stdout, result.stderr)
        assert result.stderr.strip().splitlines() == [
            f"agent-eval-gate: price row 'm': input_per_million must be a finite number of zero or more, got {value}"]


@pytest.mark.parametrize("value, loaded", [("0", 0.0), ("0.0", 0.0), ("-0.0", 0.0), ("0.25", 0.25), ("1e300", 1e300),
                                           ('"1.5"', 1.5)],
                         ids=["zero", "zero-point-zero", "negative-zero", "fraction", "large", "text"])
def test_a_finite_price_of_zero_or_more_still_loads(tmp_path, value, loaded):
    assert load_prices(price_table(tmp_path, value, value)) == {"m": (loaded, loaded)}


# ------------------------------------------------------------- why a cost is unknown, in the gate and in score's note

@pytest.mark.parametrize("fields, edit, price, cause", [
    ({"model": "unpriced"}, None, "1", "no price for unpriced"),
    ({"model": None}, None, "1", "1 answer(s) with token counts but no model (b)"),
    ({"input_tokens": None, "output_tokens": None}, None, "1", "1 answer(s) with no token counts (b)"),
    ({}, "drop", "1", "1 case(s) with no current recorded answer (b)"),
    ({}, "stale", "1", "1 case(s) with no current recorded answer (b)"),
    ({"input_tokens": "300"}, None, "1", "1 case(s) whose agent returned a refused number (b)"),
    ({"latency_ms": float("nan")}, None, "1", "1 case(s) whose agent returned a refused number (b)"),
    ({"input_tokens": 10 ** 400}, None, "1", "1 case(s) whose token count or cost is too large for a float (b)"),
    ({"input_tokens": 10 ** 300}, None, "1e300", "1 case(s) whose token count or cost is too large for a float (b)"),
], ids=["missing-price-row", "no-model", "no-token-counts", "missing-answer", "stale-answer", "refused-count",
        "refused-latency", "overflowing-count", "overflowing-cost"])
def test_the_gate_and_score_name_why_a_cost_is_unknown_and_the_gate_still_fails(tmp_path, fields, edit, price, cause):
    cases = write_cases(tmp_path, [{"id": cid, "input": cid, "checks": [{"type": "equals", "value": "ok"}]}
                                   for cid in ("a", "b", "c")])
    prices = str(price_table(tmp_path, price, price))
    base, cand = tmp_path / "base.json", tmp_path / "cand.json"
    save_recording(record(THREE, lambda _v: answer_with(), "v1"), base)
    rec = record(THREE, lambda v: answer_with(**fields) if v == "b" else answer_with(), "v2")
    if edit == "drop":
        del rec["cases"]["b"]
    elif edit == "stale":
        rec["cases"]["b"]["input_hash"] = "0" * 16
    save_recording(rec, cand)
    note = f"note: {cand}: {cause}; that cost is reported as unknown, never as $0"
    scored = cli("score", str(cases), str(cand), "--prices", prices, cwd=tmp_path)
    assert scored.returncode == 0, scored.stderr
    assert "| unknown | unknown |" in scored.stdout
    assert scored.stderr.strip().splitlines() == [note]
    gated = cli("gate", str(cases), "--baseline", str(base), "--candidate", str(cand), "--prices", prices,
                "--max-accuracy-drop", "100", "--max-newly-failing", "3", cwd=tmp_path)  # only the cost can fail it
    assert gated.returncode == 1, gated.stderr  # the gate never passes an unknown cost
    assert gated.stdout.splitlines()[0] == "gate: FAILED (v2 against v1)"
    assert f"  FAIL cost/case  unknown in the candidate: {cause}" in gated.stdout.splitlines()
    assert gated.stderr.strip().splitlines() == [note]


FIVE = [case([{"type": "equals", "value": "ok"}], cid, cid) for cid in ("a", "b", "c", "d", "e")]


def test_the_gate_names_the_causes_on_each_side_and_cuts_a_long_list_of_cases():
    base = score(FIVE, record(FIVE, lambda _v: answer_with(model="old"), "v1"), PRICED)
    cand = score(FIVE, record(FIVE, lambda v: answer_with(input_tokens=None, output_tokens=None) if v == "a"
                              else answer_with(input_tokens="300"), "v2"), PRICED)
    result = compare(base, cand, Limits(max_accuracy_drop=100, max_newly_failing=5))
    assert not result.passed
    assert result.lines[1] == ("FAIL cost/case  unknown in the baseline: no price for old; in the candidate: "
                               "1 answer(s) with no token counts (a); 4 case(s) whose agent returned a refused number "
                               "(b, c, d and 1 more)")


def test_the_gate_names_an_unknown_cost_it_cannot_explain_and_a_score_with_no_cases():
    from agent_eval_gate.core import CaseScore, Score

    by_hand = Score("v2", [CaseScore("x", True, [], None, 10.0)])  # a score built without a reason for its cost
    result = compare(Score("v1"), by_hand, Limits())
    assert not result.passed
    assert result.lines[1] == ("FAIL cost/case  unknown in the baseline: no cases; in the candidate: "
                               "1 case(s) with an unknown cost (x)")


def test_score_prints_no_note_when_every_cost_is_known(tmp_path):
    cases, prices = str(EXAMPLE / "cases.jsonl"), str(EXAMPLE / "prices.toml")
    assert cli("record", cases, "--agent", f"{EXAMPLE / 'agent_v1.py'}:run", "--out", str(tmp_path / "v1.json"),
               cwd=tmp_path).returncode == 0
    scored = cli("score", cases, str(tmp_path / "v1.json"), "--prices", prices, cwd=tmp_path)
    assert (scored.returncode, scored.stderr) == (0, "")
    unpriced = cli("score", cases, str(tmp_path / "v1.json"), cwd=tmp_path)
    assert unpriced.stderr.strip().splitlines() == [
        f"note: {tmp_path / 'v1.json'}: no price for example-small; that cost is reported as unknown, never as $0"]
