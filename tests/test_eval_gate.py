from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_eval_gate.core import (Case, EvalError, Reply, load_agent, load_cases, load_prices, percentile,
                                  record, run_checks, score)
from agent_eval_gate.gate import EMPTY_DECOYS, Limits, compare, empty_decoys, prove
from agent_eval_gate.report import html_page

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


# ------------------------------------------------------------- an agent that exits or is cancelled

def agent_file(tmp_path: Path, body: str) -> str:
    path = tmp_path / "agent_under_test.py"
    path.write_text(body, encoding="utf-8")
    return f"{path}:run"


EXITS_AS_IT_LOADS = 'import sys\nsys.exit("missing API key")\n\ndef run(v):\n    return v\n'
THREE = [case([{"type": "equals", "value": "ok"}], cid, cid) for cid in ("a", "b", "c")]


def test_an_agent_that_calls_sys_exit_as_it_loads_will_not_load(tmp_path):
    with pytest.raises(EvalError, match="cannot load the agent .*SystemExit: missing API key"):
        load_agent(agent_file(tmp_path, EXITS_AS_IT_LOADS))


def test_an_agent_cancelled_as_it_loads_will_not_load(tmp_path):
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


def test_an_agent_cancelled_on_a_case_fails_that_case_and_the_run_goes_on():
    def agent(value):
        if value == "b":
            raise asyncio.CancelledError()
        return "ok"

    rec = record(THREE, agent, "a")
    assert (rec["cases"]["b"]["error"] or "").startswith("CancelledError")
    assert [rec["cases"][cid]["error"] for cid in ("a", "c")] == [None, None]


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
