"""Reports: a markdown summary for a terminal or a pull request, and one self-contained HTML page."""
from __future__ import annotations

import html
import json
from typing import Any

from .core import Score
from .gate import GateResult


def _money(value: float | None) -> str:
    return "unknown" if value is None else f"${value:.6f}"


def markdown(score: Score) -> str:
    s = score.summary()
    lines = [
        f"## {s['agent']}",
        "",
        "| Accuracy | Passed | Cost total | Cost per case | p50 | p95 |",
        "|---|---|---|---|---|---|",
        f"| {score.accuracy:.1%} | {s['passed']} of {s['cases']} | {_money(score.cost_total)} | "
        f"{_money(score.cost_per_case)} | {s['latency_p50_ms']:.0f} ms | {s['latency_p95_ms']:.0f} ms |",
    ]
    if s["failures"]:
        lines += ["", "Failing:"]
        for failure in s["failures"]:
            lines.append(f"- `{failure['id']}`: {'; '.join(failure['reasons'])}")
    return "\n".join(lines)


def _short(value: Any, limit: int = 140) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def html_page(scores: list[Score], gate: GateResult | None = None, title: str = "agent-eval-gate") -> str:
    """One file, no scripts, no network: tiles per agent, the gate verdict, and every case."""
    esc = html.escape
    tiles = []
    for score in scores:
        tiles.append(
            f'<section class="agent"><h2>{esc(score.agent)}</h2><div class="tiles">'
            f'<div><b>{score.accuracy:.1%}</b><span>accuracy ({score.passed} of {len(score.cases)})</span></div>'
            f'<div><b>{esc(_money(score.cost_per_case))}</b><span>cost per case</span></div>'
            f'<div><b>{score.latency(50):.0f} ms</b><span>p50 latency</span></div>'
            f'<div><b>{score.latency(95):.0f} ms</b><span>p95 latency</span></div></div></section>'
        )
    verdict = ""
    if gate is not None:
        state = "pass" if gate.passed else "fail"
        items = "".join(f"<li><code>{esc(line)}</code></li>" for line in gate.lines)
        broke = "".join(f"<li><b>{esc(cid)}</b>: {esc('; '.join(r))}</li>" for cid, r in gate.newly_failing)
        verdict = (f'<section class="gate {state}"><h2>Gate: {"passed" if gate.passed else "failed"}</h2>'
                   f"<ul>{items}</ul>" + (f"<h3>Newly failing</h3><ul>{broke}</ul>" if broke else "") + "</section>")
    ids = [c.id for c in scores[0].cases] if scores else []
    by_agent = [{c.id: c for c in s.cases} for s in scores]
    head = "".join(f"<th>{esc(s.agent)}</th>" for s in scores)
    rows = []
    for cid in ids:
        cells = []
        for table in by_agent:
            c = table.get(cid)
            if c is None:
                cells.append("<td>-</td>")
                continue
            mark = "pass" if c.passed else "fail"
            why = "" if c.passed else f"<div class=\"why\">{esc('; '.join(c.reasons))}</div>"
            cells.append(f'<td class="{mark}"><span>{mark}</span> {esc(_money(c.cost_usd))}, '
                         f'{c.latency_ms:.0f} ms<div class="out">{esc(_short(c.output))}</div>{why}</td>')
        rows.append(f"<tr><th>{esc(cid)}</th>{''.join(cells)}</tr>")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>
:root {{ color-scheme: light dark; --bg:#fbfbfa; --fg:#1d1d1f; --muted:#6b6b70; --line:#e4e4e2; --card:#fff;
  --ok:#11743b; --okbg:#e8f5ed; --bad:#a3221b; --badbg:#fbeceb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141416; --fg:#ececee; --muted:#9a9aa2; --line:#2c2c31;
  --card:#1c1c20; --ok:#6fd49a; --okbg:#15291d; --bad:#ff8f87; --badbg:#2d1716; }} }}
body {{ margin:0; padding:32px clamp(16px,4vw,48px); background:var(--bg); color:var(--fg);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
h1 {{ font-size:22px; margin:0 0 20px; }} h2 {{ font-size:16px; margin:0 0 12px; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px 20px; margin:0 0 16px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }}
.tiles div {{ border:1px solid var(--line); border-radius:8px; padding:12px; }}
.tiles b {{ display:block; font-size:22px; }} .tiles span, .out {{ color:var(--muted); font-size:13px; }}
.gate.pass {{ border-color:var(--ok); }} .gate.fail {{ border-color:var(--bad); }}
.gate ul {{ margin:0; padding-left:18px; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th, td {{ border-top:1px solid var(--line); padding:8px; text-align:left; vertical-align:top; }}
td.pass span {{ color:var(--ok); font-weight:600; }} td.fail {{ background:var(--badbg); }}
td.fail span, .why {{ color:var(--bad); }} td.fail span {{ font-weight:600; }}
</style></head><body>
<h1>{esc(title)}</h1>
{verdict}
{''.join(tiles)}
<section><h2>Every case</h2><div class="scroll"><table><thead><tr><th>case</th>{head}</tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></section>
</body></html>
"""
