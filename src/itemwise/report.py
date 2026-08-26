"""Rendering: a terminal summary and a self-contained HTML report."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analysis import Report

_VERDICT_LABEL = {
    "dead": "DEAD",
    "backwards": "BACKWARDS",
    "weak": "weak",
    "acceptable": "ok",
    "strong": "STRONG",
}


def to_text(report: "Report", max_items: int = 20) -> str:
    """A terminal report. Leads with what to act on."""
    lines: list[str] = []
    add = lines.append

    add("=" * 72)
    add("ITEMWISE SUITE REPORT")
    add("=" * 72)
    add(f"  items                {report.n_items}")
    add(f"  models compared      {len(report.models)}")
    add(f"  Cronbach's alpha     {report.alpha:.3f}  ({report.alpha_verdict()})")
    add(f"  std error of meas.   {report.sem:.2f} items")
    add(f"  carrying signal      {report.signal_ratio:.0%}")
    add(f"  dead weight          {report.wasted_fraction:.0%} of every run")
    add("")

    dead = report.dead_items()
    backwards = report.backwards_items()

    if backwards:
        add("-" * 72)
        add(f"BACKWARDS ITEMS ({len(backwards)}) - fix these first")
        add("  Weaker models pass these more often than stronger ones, by more")
        add("  than chance explains. That is nearly always a broken grader.")
        add(f"  Tested against the null that the item is unrelated to ability,")
        add(f"  corrected across all {report.n_items} items at FDR {report.backwards_fdr:g}.")
        add("-" * 72)
        for s in backwards:
            pv = "" if s.backwards_p is None else f"{s.backwards_p:.5f}"
            add(f"  {s.item.id:<32} r_pb={s.point_biserial:+.3f}  "
                f"passed={s.difficulty:.2f}  p={pv}")
        add("")
    elif not report.backwards_detectable():
        add("-" * 72)
        add("BACKWARDS ITEMS - CANNOT BE ASSESSED")
        add("-" * 72)
        add(f"  {report.n_models} models is too few. With this many, even a perfectly")
        add(f"  inverted item cannot reach p <= {report.backwards_fdr:g}, so an empty list here")
        add("  means 'no evidence either way' - not 'no broken graders'.")
        shortlist = report.suspicious_items()[:5]
        if shortlist:
            add("  Pointing the wrong way, unproven, worth a look if you add models:")
            for s in shortlist:
                add(f"    {s.item.id:<30} r_pb={s.point_biserial:+.3f}  "
                    f"p={s.backwards_p:.3f}")
        add("")
    else:
        suspicious = report.suspicious_items()
        if suspicious:
            add(f"  no backwards items survive correction at FDR "
                f"{report.backwards_fdr:g} "
                f"({len(suspicious)} point the wrong way, none beyond chance)")
            add("")

    if dead:
        always = [s for s in dead if s.difficulty == 1.0]
        never = [s for s in dead if s.difficulty == 0.0]
        add("-" * 72)
        add(f"DEAD ITEMS ({len(dead)}) - retire or rewrite")
        add("-" * 72)
        if always:
            add(f"  every model passes ({len(always)}):")
            for s in always:
                add(f"    {s.item.id}")
        if never:
            add(f"  no model passes ({len(never)}) - check the grader before the prompt:")
            for s in never:
                add(f"    {s.item.id}")
        add("")

    add("-" * 72)
    add("ITEMS BY INFORMATION VALUE")
    add("-" * 72)
    add(f"  {'item':<32} {'r_pb':>7} {'D':>7} {'p':>6}  verdict")
    for s in report.ranked()[:max_items]:
        add(
            f"  {s.item.id[:32]:<32} {s.point_biserial:>+7.3f} "
            f"{s.discrimination:>+7.3f} {s.difficulty:>6.2f}  "
            f"{_VERDICT_LABEL[s.verdict]}"
        )
    if report.n_items > max_items:
        add(f"  ... {report.n_items - max_items} more")
    add("")

    add("-" * 72)
    add("MODEL TOTALS")
    add("-" * 72)
    ordered = sorted(report.model_totals.items(), key=lambda kv: kv[1], reverse=True)
    for model, total in ordered:
        add(f"  {model:<32} {total:>4} / {report.n_items}")
    if len(ordered) >= 2:
        add("")
        add(f"  A gap must exceed {report.min_real_gap():.2f} items to be real (95% conf).")
        for (a, ta), (b, tb) in zip(ordered, ordered[1:]):
            real = report.separates(a, b)
            mark = "real" if real else "NOT DISTINGUISHABLE"
            add(f"    {a} ({ta}) vs {b} ({tb}):  gap {abs(ta - tb)}  ->  {mark}")
    add("=" * 72)
    return "\n".join(lines)


def to_html(report: "Report", title: str = "Suite report") -> str:
    """A self-contained HTML report. No external assets, works offline."""
    e = html.escape

    def rows() -> str:
        out = []
        for s in report.ranked():
            out.append(
                f'<tr class="v-{s.verdict}">'
                f"<td>{e(s.item.id)}</td>"
                f'<td class="n">{s.point_biserial:+.3f}</td>'
                f'<td class="n">{s.discrimination:+.3f}</td>'
                f'<td class="n">{s.difficulty:.2f}</td>'
                f'<td class="n">{s.n_pass}/{s.n_models}</td>'
                f'<td><span class="tag t-{s.verdict}">{_VERDICT_LABEL[s.verdict]}</span></td>'
                f"<td class=\"dx\">{e(s.diagnosis)}</td>"
                "</tr>"
            )
        return "\n".join(out)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title>
<style>
:root{{--bg:#fbfbfa;--fg:#16191c;--mut:#6b7280;--line:#e3e5e4;--card:#fff;
--dead:#8a6d10;--back:#a33a2c;--strong:#0e6b52;--ok:#3f6d5f;--weak:#7a8a85}}
@media(prefers-color-scheme:dark){{:root{{--bg:#101314;--fg:#e6e9ea;--mut:#94a1a5;
--line:#252b2d;--card:#171b1d;--dead:#cba83e;--back:#d4816f;--strong:#5bb894;--ok:#7fa596;--weak:#7a8a85}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;padding:32px 20px}}
.wrap{{max-width:1080px;margin:0 auto}}
h1{{font-size:26px;margin:0 0 4px;letter-spacing:-.02em}}
.sub{{color:var(--mut);margin:0 0 28px;font-size:14px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:28px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:14px 16px}}
.kpi dt{{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--mut);margin-bottom:5px}}
.kpi dd{{margin:0;font-size:24px;font-variant-numeric:tabular-nums;letter-spacing:-.02em}}
.kpi .note{{font-size:12px;color:var(--mut);margin-top:3px}}
.tablewrap{{overflow-x:auto;border:1px solid var(--line);border-radius:4px;background:var(--card)}}
table{{border-collapse:collapse;width:100%;min-width:820px;font-size:13.5px}}
th,td{{text-align:left;padding:9px 13px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);font-weight:600;white-space:nowrap}}
tbody tr:last-child td{{border-bottom:0}}
td.n{{font-family:ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums;white-space:nowrap}}
td.dx{{color:var(--mut);font-size:12.5px;max-width:34ch}}
.tag{{font-family:ui-monospace,Menlo,monospace;font-size:10px;letter-spacing:.06em;
padding:2px 7px;border-radius:2px;border:1px solid currentColor;white-space:nowrap}}
.t-dead{{color:var(--dead)}}.t-backwards{{color:var(--back)}}.t-strong{{color:var(--strong)}}
.t-acceptable{{color:var(--ok)}}.t-weak{{color:var(--weak)}}
footer{{margin-top:22px;color:var(--mut);font-size:12.5px}}
</style></head><body><div class="wrap">
<h1>{e(title)}</h1>
<p class="sub">{report.n_items} items &middot; {len(report.models)} models compared</p>
<dl class="kpis">
<div class="kpi"><dt>Dead weight</dt><dd>{report.wasted_fraction:.0%}</dd>
<div class="note">of every run informs nothing</div></div>
<div class="kpi"><dt>Carrying signal</dt><dd>{report.signal_ratio:.0%}</dd>
<div class="note">{len(report.strong_items())} strong items</div></div>
<div class="kpi"><dt>Reliability</dt><dd>{report.alpha:.3f}</dd>
<div class="note">Cronbach&rsquo;s &alpha; &mdash; {e(report.alpha_verdict())}</div></div>
<div class="kpi"><dt>Min. real gap</dt><dd>{report.min_real_gap():.2f}</dd>
<div class="note">closer models are tied (95% conf)</div></div>
</dl>
<div class="tablewrap"><table>
<thead><tr><th>Item</th><th>r<sub>pb</sub></th><th>D</th><th>p</th><th>Passed</th><th>Verdict</th><th>What to do</th></tr></thead>
<tbody>
{rows()}
</tbody></table></div>
<footer>r<sub>pb</sub> = corrected point-biserial correlation &middot;
D = Kelley discrimination index &middot; p = difficulty (proportion passing).
Generated by itemwise.</footer>
</div></body></html>"""
