"""Eval history dashboard: renders evals/report.html from the JSONL log.

Each suite gets one SVG trend chart (pass rate over time, one line per
backend) plus a table of every recorded run. Pure stdlib — no JS, no
deps — so the report opens anywhere.

    python3 evals/eval_history.py   # writes evals/report.html
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from history import load

PALETTE = ["#0ea5e9", "#f59e0b", "#10b981", "#ef4444", "#a78bfa", "#f472b6"]
W, H, PAD = 640, 220, 36


def _svg(series, dates):
    # series: {backend: {date: rate}} — rates are 0..1, dates sorted ISO
    n = len(dates)
    def x(i):
        return PAD + i * (W - 2 * PAD) / max(n - 1, 1)
    def y(rate):
        return H - PAD - rate * (H - 2 * PAD)
    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart">']
    for g in (0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{PAD}" y1="{y(g):.1f}" x2="{W-PAD}" '
                     f'y2="{y(g):.1f}" class="grid"/>')
        parts.append(f'<text x="{PAD-6}" y="{y(g)+4:.1f}" class="tick">'
                     f'{int(g*100)}%</text>')
    for i, d in enumerate(dates):
        if i % max(n // 8, 1):
            continue  # too many dates to label — sample the ticks
        parts.append(f'<text x="{x(i):.1f}" y="{H-12}" class="tick" '
                     f'text-anchor="middle">{d[5:]}</text>')
    for k, (backend, vals) in enumerate(series.items()):
        pts = [(dates.index(d), vals[d]) for d in sorted(vals)]
        color = PALETTE[k % len(PALETTE)]
        path = " ".join(f"{x(i):.1f},{y(r):.1f}" for i, r in pts)
        parts.append(f'<polyline points="{path}" class="line" '
                     f'stroke="{color}"/>')
        for i, r in pts:
            parts.append(f'<circle cx="{x(i):.1f}" cy="{y(r):.1f}" r="3.5" '
                         f'fill="{color}"><title>{backend} {dates[i]}: '
                         f'{r:.0%}</title></circle>')
        parts.append(f'<text x="{W-PAD+6}" y="{y(pts[-1][1]):.1f}" '
                     f'class="legend" fill="{color}">{backend}</text>')
    return "".join(parts) + "</svg>"


def main():
    rows = load()
    suites = {}
    for r in rows:
        if r["total"]:
            suites.setdefault(r["suite"], {}).setdefault(
                r["backend"], {})[r["date"]] = r["passed"] / r["total"]

    css = ("body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem "
           "auto;padding:0 1rem;color:#1e293b}"
           "h1{font-size:1.4rem}h2{font-size:1.1rem;margin-top:2rem}"
           ".chart{width:100%;border:1px solid #e2e8f0;border-radius:8px}"
           ".grid{stroke:#e2e8f0}.tick{font-size:9px;fill:#94a3b8}"
           ".line{fill:none;stroke-width:2}.legend{font-size:11px}"
           "table{width:100%;border-collapse:collapse;font-size:.85rem}"
           "th,td{text-align:left;padding:.4rem;border-bottom:1px solid #e2e8f0}"
           ".src{color:#94a3b8;font-size:.75rem}"
           ".note{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;"
           "padding:.8rem 1rem;font-size:.85rem}")

    out = ["<html><head><meta charset='utf-8'><title>agent-service eval "
           "history</title><style>", css, "</style></head><body>",
           "<h1>agent-service — eval history</h1>"]
    if not rows:
        out.append("<p class='note'>No recorded runs yet — "
                   "<code>python3 evals/run_eval.py</code> appends them "
                   "automatically.</p>")
    else:
        backfilled = any(r.get("source") == "roadmap-backfill" for r in rows)
        out.append(f"<p class='note'>{len(rows)} recorded runs · newest first. "
                   + ("Rows marked <span class='src'>backfill</span> were "
                      "reconstructed from the roadmap's dated ship notes, not "
                      "re-run." if backfilled else "") + "</p>")
    for suite in sorted(suites):
        dates = sorted({d for s in suites[suite].values() for d in s})
        out.append(f"<h2>{suite}</h2>" + _svg(suites[suite], dates))
    out.append("<h2>all runs</h2><table><tr><th>date</th><th>suite</th>"
               "<th>backend</th><th>score</th><th>detail</th></tr>")
    for r in sorted(rows, key=lambda r: (r["date"], r["suite"], r["backend"]),
                    reverse=True):
        src = (" <span class='src'>backfill</span>"
               if r.get("source") == "roadmap-backfill" else "")
        out.append(f"<tr><td>{r['date']}</td><td>{r['suite']}</td>"
                   f"<td>{r['backend']}</td>"
                   f"<td>{r['passed']}/{r['total']}</td>"
                   f"<td>{r.get('detail', '')}{src}</td></tr>")
    out.append("</table></body></html>")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "report.html")
    with open(path, "w") as f:
        f.write("".join(out))
    print(f"report.html: {len(rows)} runs, {len(suites)} suite charts -> {path}")


if __name__ == "__main__":
    main()
