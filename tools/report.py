import copy
import html
from datetime import date

from tools.charts import top_driver_columns
from tools.palette import ACCENT, LINE, MUTED, SURFACE, TEXT


def plain_summary(raw, cleaned, target, model_out, log, sampled=False):
    pts = [f"Your data has {len(raw):,} rows and {raw.shape[1]} columns."
           + (" A random sample was used to keep things fast." if sampled else "")]
    miss = raw.isna().mean().mean() * 100
    if log:
        pts.append(f"About {miss:.1f}% of the cells were empty. We found {len(log)} issue(s) and fixed them "
                   "automatically (see the cleaning section).")
    else:
        pts.append("The data was already clean, so nothing needed fixing.")

    if model_out:
        res = model_out["results"]
        base = float(res[res["model"].str.startswith("Baseline")]["mean"].iloc[0])
        best = float(res[res["model"] == model_out["best_model"]]["mean"].iloc[0])
        if model_out["task"] == "classification":
            lift = best - base
            verdict = ("barely better than guessing, so predictions should be treated with caution"
                       if lift < 0.03 else
                       "somewhat better than guessing" if lift < 0.10 else
                       "clearly better than guessing")
            if model_out.get("metric") == "balanced_accuracy":
                pts.append(
                    f"A model ({model_out['best_model']}) gets '{target}' right about {best:.0%} of the time "
                    f"when every outcome counts equally. That matters here because one outcome is rare. "
                    f"Guessing scores {base:.0%} on the same measure, so the model is {verdict}."
                )
            else:
                pts.append(
                    f"A model ({model_out['best_model']}) predicted '{target}' correctly about {best:.0%} of the time. "
                    f"Always guessing the most common answer would be right {base:.0%} of the time, "
                    f"so the model is {verdict}."
                )
        else:
            r2 = max(best, 0)
            verdict = "weak" if r2 < 0.2 else "moderate" if r2 < 0.5 else "strong"
            line = (f"The model ({model_out['best_model']}) explains about {r2:.0%} of the ups and downs "
                    f"in '{target}', which is a {verdict} result.")
            if model_out.get("split") == "time-ordered":
                line += (f" It was tested on later dates than it learned from. Simply repeating the "
                         f"previous value explains {max(base, 0):.0%}"
                         + (", so the model adds real value." if best > base + 0.02 else
                            ", which is as good or better, so treat the model with caution."))
            pts.append(line)
        fc = model_out.get("fit_check")
        if fc:
            pts.append(fc["plain"])
        drivers = top_driver_columns(cleaned, target, model_out["importance"])
        if any(str(f).startswith("target_") for f in model_out["importance"].index[:3]):
            drivers = [f"its own recent values"] + drivers[:2]
        if drivers:
            pts.append("The factors that matter most are: " + ", ".join(drivers) + ".")
    pts.append("These are patterns in the data, not proof of cause and effect.")
    return pts


_CSS = """
body{font-family:"SF Mono","Menlo","Consolas","Liberation Mono",monospace;margin:0;background:#000;color:%TEXT%;line-height:1.55;font-size:13px}
header{padding:30px 8vw;border-bottom:1px solid %LINE%;border-top:3px solid %ACCENT%}
header h1{margin:0;font-size:22px;color:#fff;font-weight:600} header p{margin:8px 0 0;color:#BDBDBD}
main{max-width:1000px;margin:0 auto;padding:10px 24px 60px}
h2{margin-top:36px;font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:%ACCENT%}
h3{margin:0 0 6px;font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:#fff}
.card{background:%SURFACE%;border:1px solid %LINE%;border-radius:2px;padding:16px 18px;margin:14px 0}
.note{border-left:1px solid %ACCENT%;padding:0 0 0 10px;margin:8px 0 0;color:#BDBDBD}
.summary li{margin:8px 0} dt{font-weight:600;margin-top:10px;color:#fff} dd{margin-left:0;color:#BDBDBD}
table{border-collapse:collapse;width:100%} th,td{padding:6px 10px;border-bottom:1px solid %LINE%;text-align:left}
details{background:%SURFACE%;border:1px solid %LINE%;border-radius:2px;padding:12px 18px;margin:14px 0}
summary{cursor:pointer;color:%ACCENT%;text-transform:uppercase;letter-spacing:.06em;font-size:12px}
@media print{*{-webkit-print-color-adjust:exact;print-color-adjust:exact} .card,details{break-inside:avoid}}
"""

_GLOSSARY = [
    ("Baseline", "The score you get by simply guessing. A useful model has to beat it."),
    ("Accuracy", "The share of predictions that are correct."),
    ("R²", "How much of the ups and downs in the data the model explains (0% none, 100% all)."),
    ("Cross-validation", "The model is tested 5 times on data it has not seen, so the score is not a lucky accident."),
    ("Missing value", "An empty cell in the data."),
    ("Outlier", "A value that is unusually large or small compared to the rest."),
]


def build_report_html(title, dataset_name, target, summary_points, ai_summary, insights,
                      charts, log, model_out, audience="Everyone"):
    esc = html.escape
    technical = audience == "Technical"
    css = (_CSS.replace("%TEXT%", TEXT).replace("%LINE%", LINE).replace("%SURFACE%", SURFACE)
           .replace("%ACCENT%", ACCENT).replace("%MUTED%", MUTED))
    edition = "Technical edition" if technical else "Plain-language edition"
    p = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{esc(title)}</title>"
         f"<style>{css}</style></head><body>",
         f"<header><h1>{esc(title)}</h1><p>Dataset: {esc(str(dataset_name))} | Question: {esc(str(target))} "
         f"| {edition} | Generated {date.today():%d %b %Y}</p></header><main>",
         "<h2>In a nutshell</h2><div class='card'>"]
    if ai_summary:
        p.append(f"<p>{esc(ai_summary).replace(chr(10), '<br>')}</p>")
    p.append("<ul class='summary'>" + "".join(f"<li>{esc(s)}</li>" for s in summary_points) + "</ul></div>")

    if insights:
        p.append("<h2>Key findings</h2><div class='card'><ul class='summary'>"
                 + "".join(f"<li>{esc(s)}</li>" for s in insights) + "</ul></div>")

    first, current = True, None
    for c in charts:
        if not technical and c.get("level", "core") != "core":
            continue
        if c["section"] != current:
            current = c["section"]
            p.append(f"<h2>{esc(current)}</h2>")
        fig = copy.deepcopy(c["fig"])
        fig.update_layout(height=420)
        div = fig.to_html(full_html=False, include_plotlyjs=True if first else False)
        first = False
        text = (c.get("tech") or c["note"]) if technical else c["note"]
        p.append(f"<section class='card'><h3>{esc(c['title'])}</h3>{div}"
                 f"<p class='note'>{esc(text)}</p></section>")

    p.append("<h2>Data cleaning</h2><div class='card'>")
    p.append("<ul>" + "".join(f"<li>{esc(s)}</li>" for s in log) + "</ul>" if log
             else "<p>No cleaning was needed.</p>")
    p.append("</div><h2>Glossary</h2><div class='card'><dl>"
             + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in _GLOSSARY) + "</dl></div>")

    if model_out and technical:
        p.append("<h2>Technical details</h2><div class='card'>")
        p.append(model_out["results"].round(3).to_html(index=False, border=0))
        p.append(f"<p><b>Task:</b> {esc(model_out['task'])} | <b>Metric:</b> {esc(model_out['metric'])} "
                 f"| <b>Rows used for modeling:</b> {model_out['modeled_rows']:,}</p>")
        if model_out["dropped_columns"]:
            p.append("<p><b>Not used as inputs (IDs or free text):</b> "
                     + esc(", ".join(model_out["dropped_columns"])) + "</p>")
        p.append("<p><b>Inputs used:</b> " + esc(", ".join(model_out["features_used"])) + "</p></div>")

    p.append(f"<p style='color:{MUTED}'>Generated automatically. Patterns are not proof of cause and effect.</p>")
    p.append("</main></body></html>")
    return "".join(p)