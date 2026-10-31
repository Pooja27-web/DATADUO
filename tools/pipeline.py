import pandas as pd

from tools.analysis import auto_analyze, regression_analyze
from tools.charts import build_story_charts
from tools.cleaning import clean_dataframe
from tools.modeling import auto_model
from tools.report import plain_summary

MAX_ROWS = 100_000


def run_pipeline(raw: pd.DataFrame, target: str) -> dict:
    rows_total = len(raw)
    data = raw.dropna(subset=[target])
    sampled = False
    if len(data) > MAX_ROWS:
        data = data.sample(MAX_ROWS, random_state=42)
        sampled = True

    missing_pct = float(data.isna().mean().mean() * 100)
    dup_pct = float(data.duplicated().mean() * 100)
    health = int(max(0, min(100, round(100 - 2 * missing_pct - 2 * dup_pct))))

    cleaned, log = clean_dataframe(data)

    # Text findings and extra charts
    insights, extra = [], {}
    try:
        t = cleaned[target]
        if pd.api.types.is_numeric_dtype(t) and t.nunique() > 10:
            insights, extra = regression_analyze(cleaned, target)
        elif t.nunique() == 2:
            tmp = cleaned.copy()
            tmp[target] = (t == t.max()).astype(int)
            insights, extra = auto_analyze(tmp, target)
    except Exception:
        insights, extra = [], {}

    # Models
    model_out, model_error = None, None
    try:
        model_out = auto_model(cleaned, target)
    except Exception as e:
        model_error = str(e)

    story = build_story_charts(data, cleaned, target, model_out)
    summary = plain_summary(data, cleaned, target, model_out, log, sampled)

    score = baseline = None
    if model_out:
        res = model_out["results"]
        baseline = float(res[res["model"].str.startswith("Baseline")]["mean"].iloc[0])
        score = float(res[res["model"] == model_out["best_model"]]["mean"].iloc[0])

    return {
        "target": target, "raw": data, "cleaned": cleaned, "log": log,
        "insights": insights, "extra_figures": extra,
        "model_out": model_out, "model_error": model_error,
        "story": story, "summary": summary,
        "health": health, "score": score, "baseline": baseline,
        "rows_total": rows_total, "sampled": sampled,
    }