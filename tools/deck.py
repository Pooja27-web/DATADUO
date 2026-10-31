"""Builds the downloadable 6-page, 16:9 business report (one self-contained HTML file).

audience="Everyone": the 4 core charts per page with plain-language notes.
audience="Technical": up to 6 charts per page with technical notes.
"""
import copy
import html
from datetime import date

from tools.palette import ACCENT, LINE, MUTED, SURFACE, TEXT

_CSS = """
*{box-sizing:border-box}
body{margin:0;background:#000;font-family:"SF Mono","Menlo","Consolas","Liberation Mono",monospace;color:%TEXT%}
.tip{max-width:1280px;margin:14px auto 0;padding:10px 16px;border:1px solid %LINE%;background:%SURFACE%;font-size:12px;color:#BDBDBD}
.slide{position:relative;width:1280px;height:720px;margin:22px auto;padding:26px 32px 12px;overflow:hidden;
 display:flex;flex-direction:column;border:1px solid %LINE%;
 background-color:#000;%BGIMG%}
.bar{position:absolute;left:0;top:0;right:0;height:3px;background:%ACCENT%}
header{display:flex;justify-content:space-between;align-items:flex-start;height:62px}
h1{margin:0;font-size:24px;font-weight:600;color:#fff;letter-spacing:.02em}
header p{margin:4px 0 0;color:#BDBDBD;font-size:13px}
.pill{border:1px solid %ACCENT%;color:%ACCENT%;border-radius:2px;padding:4px 10px;font-size:11px;
 text-transform:uppercase;letter-spacing:.06em;white-space:nowrap}
.body{flex:1;min-height:0;display:grid;grid-template-columns:220px 1fr;gap:16px;margin-top:6px}
.body.nokpi{grid-template-columns:1fr}
aside{display:flex;flex-direction:column;gap:8px;min-height:0}
.kpi{flex:1;max-height:84px;padding:8px 14px;display:flex;flex-direction:column;justify-content:center;
 background:%SURFACE%;border:1px solid %LINE%;border-radius:2px}
.kpi span{font-size:10.5px;color:%MUTED%;text-transform:uppercase;letter-spacing:.06em}
.kpi b{font-size:22px;font-weight:600;line-height:1.15;color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.grid{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:12px;min-height:0;min-width:0}
.grid.n1{grid-template-columns:1fr;grid-template-rows:1fr}
.grid.n2{grid-template-rows:1fr}
.grid.n3 .card:nth-child(3){grid-column:span 2}
.grid.cols3{grid-template-columns:1fr 1fr 1fr}
.grid.cols3 .card p{font-size:9.5px;max-height:40px}
.card{background:#0A0A0A;border:1px solid %LINE%;border-radius:2px;padding:10px 14px 8px;display:flex;
 flex-direction:column;min-height:0;min-width:0}
.card h3{margin:0 0 2px;font-size:12px;font-weight:600;color:#fff;text-transform:uppercase;letter-spacing:.05em}
.chart{flex:1;min-height:0;min-width:0;position:relative}
.chart>div{position:absolute;inset:0}
.chart .plotly-graph-div{height:100%!important;width:100%!important}
.card p{margin:4px 0 0;font-size:10.5px;line-height:1.35;color:#BDBDBD;border-left:1px solid %ACCENT%;
 padding-left:8px;max-height:30px;overflow:hidden}
.empty{grid-column:1/-1;grid-row:1/-1;display:flex;align-items:center;justify-content:center;text-align:center;
 padding:30px;font-size:15px;color:#BDBDBD;border:1px dashed %LINE%}
footer{display:flex;justify-content:space-between;font-size:10.5px;color:%MUTED%;padding-top:6px;height:22px}
@page{size:1280px 720px;margin:0}
@media print{
 .tip{display:none} body{background:#000}
 .slide{margin:0;border:0;break-after:page;page-break-after:always}
 *{-webkit-print-color-adjust:exact;print-color-adjust:exact}
}
"""


def _chart_html(fig, first):
    f = copy.deepcopy(fig)
    f.update_layout(height=None, width=None, autosize=True, margin=dict(l=8, r=8, t=6, b=8),
                    font=dict(size=11), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return f.to_html(full_html=False, include_plotlyjs=True if first else False,
                     default_height="100%", default_width="100%",
                     config={"displayModeBar": False, "responsive": True})


def build_deck_html(pages, dataset_name, audience="Everyone", title="Business Performance Report", bg_uri=None):
    esc = html.escape
    technical = audience == "Technical"
    total = len(pages)
    bgimg = (f"background-image:linear-gradient(rgba(0,0,0,.62),rgba(0,0,0,.62)),url('{bg_uri}');"
             "background-size:cover;background-position:center right;") if bg_uri else ""
    css = (_CSS.replace("%TEXT%", TEXT).replace("%LINE%", LINE).replace("%SURFACE%", SURFACE)
           .replace("%ACCENT%", ACCENT).replace("%MUTED%", MUTED).replace("%BGIMG%", bgimg))
    edition = "Technical edition" if technical else "Plain-language edition"
    out = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{esc(title)}</title>"
           f"<style>{css}</style></head><body>",
           "<div class='tip'>To save as PDF: press Ctrl + P, choose Save as PDF, set margins to None and tick "
           "Background graphics. Each page is 16:9.</div>"]
    first = True
    for i, p in enumerate(pages, 1):
        kpis = "".join(f"<div class='kpi'><span>{esc(str(k))}</span><b>{esc(str(v))}</b></div>"
                       for k, v in p["kpis"][:6])
        charts = [c for c in p["charts"] if technical or c.get("level", "core") == "core"]
        cards = []
        for c in charts:
            text = c.get("tech") or c["note"] if technical else c["note"]
            cards.append(f"<div class='card'><h3>{esc(c['title'])}</h3>"
                         f"<div class='chart'>{_chart_html(c['fig'], first)}</div>"
                         f"<p>{esc(text)}</p></div>")
            first = False
        if not cards:
            cards.append(f"<div class='empty'>{esc(p['message'] or 'Not enough information for this page.')}</div>")
        n = len(charts)
        grid_cls = "cols3" if n >= 5 else f"n{n}"
        out.append(
            f"<section class='slide'><div class='bar'></div>"
            f"<header><div><h1>{esc(p['title'])}</h1><p>{esc(p['subtitle'])}</p></div>"
            f"<div class='pill'>Page {i} of {total}</div></header>"
            f"<div class='body{'' if kpis else ' nokpi'}'>"
            + (f"<aside>{kpis}</aside>" if kpis else "")
            + f"<div class='grid {grid_cls}'>{''.join(cards)}</div></div>"
            f"<footer><span>{esc(title)} | {esc(str(dataset_name))} | {edition}</span>"
            f"<span>Generated {date.today():%d %b %Y}</span></footer></section>"
        )
    out.append("<script>window.addEventListener('load',function(){window.dispatchEvent(new Event('resize'))});</script>")
    out.append("</body></html>")
    return "".join(out)
