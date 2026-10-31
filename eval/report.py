"""Static HTML report for a comparison: deltas with CIs, categories, cost, case diffs."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent import constants as ac
from eval import constants as c
from eval.gate import evaluate_gate, load_thresholds
from eval.compare import compare_runs
from eval.io import read_json

MAX_CASES = 12
AXIS = 0.5  # delta axis spans [-AXIS, +AXIS]

CSS = """
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--line:#e7e5e4;--card:#fff;
--good:#15803d;--bad:#b91c1c;--neutral:#a8a29e;--goodbg:#dcfce7;--badbg:#fee2e2}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;
--line:#44403c;--card:#292524;--good:#4ade80;--bad:#f87171;--goodbg:#14532d;--badbg:#7f1d1d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:32px 0 10px;text-transform:uppercase;
letter-spacing:.06em;color:var(--muted)}.sub{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;overflow-x:auto}
table{border-collapse:collapse;width:100%}th,td{padding:7px 10px;border-bottom:1px solid var(--line);
text-align:left;vertical-align:top}th{font-weight:600;color:var(--muted);font-size:12px}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.badge{display:inline-block;padding:3px 10px;border-radius:99px;font-weight:600;font-size:13px}
.pass{background:var(--goodbg);color:var(--good)}.fail{background:var(--badbg);color:var(--bad)}
.warn{color:var(--bad);margin:4px 0}.good{color:var(--good)}.bad{color:var(--bad)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.kpi .v{font-size:20px;font-weight:600}.kpi .l{color:var(--muted);font-size:12px}
svg.ci{display:block}.reason{color:var(--muted);font-size:12px;max-width:380px}
.q{font-weight:500}code{font-size:12px}
"""


def _fmt(v: Optional[float], digits: int = 3) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def _ci_svg(delta: float, lo: float, hi: float, lower_better: bool) -> str:
    """Inline SVG of a delta with its CI on a fixed axis.

    Parameters
    ----------
    delta: float
        Mean delta.
    lo: float
        CI low.
    hi: float
        CI high.
    lower_better: bool
        Whether a negative delta is an improvement.

    Returns
    -------
    str
        SVG markup.
    """
    w, h = 220, 22

    def x(v: float) -> float:
        return max(4, min(w - 4, w / 2 + v / AXIS * (w / 2 - 4)))

    bad = (lo > 0) if lower_better else (hi < 0)
    good = (hi < 0) if lower_better else (lo > 0)
    color = "var(--bad)" if bad else "var(--good)" if good else "var(--neutral)"
    return (
        f'<svg class="ci" width="{w}" height="{h}" role="img" aria-label="delta {delta:+.3f}">'
        f'<line x1="{w/2}" y1="2" x2="{w/2}" y2="{h-2}" stroke="var(--line)"/>'
        f'<line x1="{x(lo):.1f}" y1="{h/2}" x2="{x(hi):.1f}" y2="{h/2}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
        f'<circle cx="{x(delta):.1f}" cy="{h/2}" r="4.5" fill="{color}"/></svg>'
    )


def _cell(delta: Optional[float], lower_better: bool) -> str:
    if delta is None:
        return '<td class="num">—</td>'
    improved = delta < 0 if lower_better else delta > 0
    cls = "good" if improved and abs(delta) >= 0.01 else "bad" if abs(delta) >= 0.01 else ""
    return f'<td class="num {cls}">{delta:+.2f}</td>'


def render(comparison: Dict[str, Any], gate_result: Dict[str, Any]) -> str:
    """Render the comparison as a standalone HTML page.

    Parameters
    ----------
    comparison: Dict[str, Any]
        Output of :func:`eval.compare.compare_runs`.
    gate_result: Dict[str, Any]
        Output of :func:`eval.gate.evaluate_gate`.

    Returns
    -------
    str
        HTML document.
    """
    a, b = comparison[c.CMP_RUN_A], comparison[c.CMP_RUN_B]
    ma = read_json(c.RUNS_DIR / a / c.RUN_MANIFEST_FILE)
    mb = read_json(c.RUNS_DIR / b / c.RUN_MANIFEST_FILE)
    e = html.escape
    passed = gate_result[c.GATE_PASSED]
    cur = comparison.get(c.SUM_CURRENCY) or ""

    def desc(m: Dict[str, Any]) -> str:
        ag = m[c.RM_AGENT]
        return f"{ag[ac.CFG_MODEL_ID]} · top_k={ag[ac.CFG_TOP_K]} · prompt {ag[c.RM_PROMPT_HASH][7:15]}"

    parts: List[str] = [
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Eval comparison</title><style>{CSS}</style></head><body><main>",
        f"<h1>Run comparison</h1><div class='sub'>A <code>{e(a)}</code> ({e(desc(ma))})<br>"
        f"B <code>{e(b)}</code> ({e(desc(mb))})<br>dataset {e(str(ma[c.RM_DATASET][c.RM_DATASET_VERSION]))} · "
        f"{comparison[c.CMP_SHARED_CASES]} shared cases · judge {e(str((ma.get(c.RM_JUDGE) or {}).get(c.JCFG_MODEL_ID, 'deterministic only')))}</div>",
        f"<p><span class='badge {'pass' if passed else 'fail'}'>Gate {'passed' if passed else 'failed'}</span></p>",
    ]
    for w in comparison[c.CMP_WARNINGS]:
        parts.append(f"<div class='warn'>⚠ {e(w)}</div>")
    if gate_result[c.GATE_BREACHES]:
        parts.append("<div class='card'><table><tr><th>Breached rule</th><th>Detail</th></tr>")
        for br in gate_result[c.GATE_BREACHES]:
            parts.append(f"<tr><td>{e(br[c.GATE_RULE])}</td><td>{e(br[c.GATE_DETAIL])}</td></tr>")
        parts.append("</table></div>")

    ops = {r[c.CMP_METRIC]: r for r in comparison[c.CMP_OPERATIONAL]}
    parts.append("<h2>Cost and latency</h2><div class='grid'>")
    for key, label, digits, unit in (
        (c.OP_AGENT_COST_MEAN, "Agent cost / question", 6, cur),
        (c.OP_COST_PER_CORRECT, "Cost / correct answer", 6, cur),
        (c.OP_LATENCY_P50, "Latency p50", 2, "s"),
        (c.OP_LATENCY_P95, "Latency p95", 2, "s"),
        (c.OP_TOKENS_IN, "Input tokens / question", 0, ""),
        (c.OP_ERROR_RATE, "Error rate", 3, ""),
    ):
        r = ops.get(key, {})
        rel = r.get(c.CMP_REL_CHANGE)
        rel_s = "" if rel is None else f" <span class='{'bad' if rel > 0 else 'good'}'>({rel:+.0%})</span>"
        parts.append(
            f"<div class='card kpi'><div class='l'>{e(label)}</div><div class='v'>"
            f"{_fmt(r.get(c.CMP_VALUE_A), digits)} → {_fmt(r.get(c.CMP_VALUE_B), digits)} {e(unit)}{rel_s}</div></div>"
        )
    parts.append("</div>")

    overall = [r for r in comparison[c.CMP_QUALITY] if r[c.CMP_SCOPE] == c.SCOPE_OVERALL]
    parts.append(
        "<h2>Quality (B − A, paired, 95% bootstrap CI)</h2><div class='card'><table>"
        "<tr><th>Metric</th><th class='num'>A</th><th class='num'>B</th><th class='num'>Δ</th>"
        f"<th>CI on [−{AXIS}, +{AXIS}]</th><th class='num'>n</th></tr>"
    )
    for r in overall:
        lb = r[c.CMP_METRIC] in c.LOWER_IS_BETTER
        name = r[c.CMP_METRIC] + (" (lower is better)" if lb else "")
        parts.append(
            f"<tr><td>{e(name)}</td><td class='num'>{_fmt(r[c.CMP_MEAN_A])}</td><td class='num'>{_fmt(r[c.CMP_MEAN_B])}</td>"
            f"<td class='num'>{r[c.CMP_DELTA]:+.3f}</td><td>{_ci_svg(r[c.CMP_DELTA], r[c.CMP_CI_LOW], r[c.CMP_CI_HIGH], lb)}</td>"
            f"<td class='num'>{r[c.CMP_N]}</td></tr>"
        )
    parts.append("</table></div>")

    metrics = [r[c.CMP_METRIC] for r in overall]
    by = {(r[c.CMP_METRIC], r[c.CMP_SCOPE]): r for r in comparison[c.CMP_QUALITY]}
    parts.append("<h2>Δ by category</h2><div class='card'><table><tr><th>Category</th>")
    parts += [f"<th class='num'>{e(m)}</th>" for m in metrics]
    parts.append("</tr>")
    for cat in c.CATEGORIES:
        if not any((m, cat) in by for m in metrics):
            continue
        parts.append(f"<tr><td>{e(cat)}</td>")
        for m in metrics:
            r = by.get((m, cat))
            parts.append(_cell(r[c.CMP_DELTA] if r else None, m in c.LOWER_IS_BETTER))
        parts.append("</tr>")
    parts.append("</table></div>")

    worst = [d for d in comparison[c.CMP_CASE_DIFFS] if any(v < 0 for k, v in d[c.CMP_DELTAS].items() if k not in c.LOWER_IS_BETTER)]
    parts.append(f"<h2>Most regressed cases (top {MAX_CASES})</h2><div class='card'><table>"
                 "<tr><th>Case</th><th>Question</th><th>Δ</th><th>Judge reason A → B (correctness)</th></tr>")
    for d in worst[:MAX_CASES]:
        deltas = ", ".join(f"{k} {v:+.2f}" for k, v in d[c.CMP_DELTAS].items() if abs(v) >= 0.01)
        reasons = ""
        if d.get(c.CMP_REASON_A) or d.get(c.CMP_REASON_B):
            reasons = f"<div class='reason'>A: {e(str(d.get(c.CMP_REASON_A) or ''))}</div><div class='reason'>B: {e(str(d.get(c.CMP_REASON_B) or ''))}</div>"
        parts.append(
            f"<tr><td><code>{e(d[c.CT_CASE_ID])}</code><br><span class='sub'>{e(d[c.CT_CATEGORY])}</span></td>"
            f"<td class='q'>{e(d[c.CMP_QUESTION])}</td><td>{e(deltas)}</td><td>{reasons}</td></tr>"
        )
    parts.append("</table></div></main></body></html>")
    return "".join(parts)


def write_report(run_a: str, run_b: str) -> Path:
    """Compare, gate and write the HTML report.

    Parameters
    ----------
    run_a: str
        Baseline run.
    run_b: str
        Candidate run.

    Returns
    -------
    Path
        Report file path.
    """
    th = load_thresholds()
    comparison = compare_runs(
        run_a, run_b,
        samples=int(th.get(c.TH_BOOTSTRAP_SAMPLES, 2000)),
        confidence=float(th.get(c.TH_CONFIDENCE, 0.95)),
        seed=int(th.get(c.TH_SEED, 7)),
    )
    gate_result = evaluate_gate(comparison, th)
    c.REPORTS_DIR.mkdir(exist_ok=True)
    path = c.REPORTS_DIR / c.REPORT_FILE_TEMPLATE.format(a=run_a, b=run_b)
    path.write_text(render(comparison, gate_result), encoding=c.TEXT_ENCODING)
    return path
