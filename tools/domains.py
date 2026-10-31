"""tools/domains.py — dataset domain detection and dashboard routing.

DATADUO used to have exactly one dashboard (sales) plus generic EDA, so any
dataset that was not sales-shaped fell through to "needs X column" messages.

This module looks at a dataset first, decides what KIND of data it is, and
routes to the dashboard built for that kind. Every builder returns the same
page shape the app already renders:

    {"id", "title", "subtitle", "kpis": [(label, value)],
     "charts": [{"title", "fig", "note", "tech", "level"}], "message"}

Adding a new domain means writing a _build_<name>() and registering it.
"""

import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from tools.palette import ACCENT, BAD, COLORWAY, GOOD, MUTED, OLIVE, SAND, SCALE

MAX_CHARTS_PER_PAGE = 6
CORE_CHARTS = 4          # first N charts shown to everyone, rest are "technical"
TOP_N_BUCKET = 8


# ───────────────────────────── shared helpers ─────────────────────────────

def _norm(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _fmt(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "n/a"
    a = abs(v)
    for lim, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= lim:
            return f"{v / lim:.1f}{suf}"
    return f"{v:,.0f}" if a >= 100 else f"{v:,.2f}"


def _clip(s, n=28):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _layout(fig, legend=False):
    fig.update_layout(margin=dict(l=8, r=8, t=10, b=8), showlegend=legend)
    if legend:
        fig.update_layout(legend=dict(orientation="h", y=1.12, x=0))
    return fig


def _chart(title, fig, note, tech=None, level="core"):
    return {"title": title, "fig": fig, "note": note, "tech": tech or note, "level": level}


def _bucket_top(s, n=TOP_N_BUCKET):
    top = s.value_counts().nlargest(n).index
    return s.where(s.isin(top), "Other")


def _is_datetime(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype == object:
        sample = s.dropna().head(200)
        if sample.empty:
            return False
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            return parsed.notna().mean() > 0.8
        except Exception:
            return False
    return False


def _to_datetime(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    try:
        return pd.to_datetime(s, errors="coerce", format="mixed")
    except Exception:
        return pd.to_datetime(s, errors="coerce")


def _looks_like_id(s, name, nunique, n):
    """An all-unique INTEGER column that marches upward, or is named like a
    key, is an identifier. A continuous float being all-unique is normal and
    must not be mistaken for an ID — that distinction matters, because
    treating 'Sales' as an ID silently removes it from every analysis."""
    if nunique < n or n <= 20:
        return False
    if not pd.api.types.is_integer_dtype(s):
        return False
    norm = _norm(name)
    if norm.endswith("_id") or norm in ("id", "index", "sr_no", "serial", "row"):
        return True
    clean = s.dropna()
    return bool(clean.is_monotonic_increasing or clean.is_monotonic_decreasing)


def profile_columns(df):
    """Classify every column once, so each domain can reason about the data
    without re-deriving types. Returns a dict of lists keyed by role."""
    out = {"numeric": [], "discrete": [], "categorical": [], "high_card": [],
           "datetime": [], "id_like": [], "constant": []}
    n = len(df)
    for c in df.columns:
        s = df[c]
        nunique = s.nunique(dropna=True)

        if nunique <= 1:
            out["constant"].append(c)
        elif _is_datetime(s):
            out["datetime"].append(c)
        elif pd.api.types.is_numeric_dtype(s):
            if _looks_like_id(s, c, nunique, n):
                out["id_like"].append(c)
            else:
                (out["numeric"] if nunique > 10 else out["discrete"]).append(c)
        elif nunique >= n and n > 20:
            out["id_like"].append(c)
        elif nunique <= 20:
            out["categorical"].append(c)
        elif nunique <= 50:
            out["high_card"].append(c)
        else:
            out["id_like"].append(c)
    return out


def _page(pid, title, subtitle, kpis, builders, data, missing=""):
    """Run each chart builder defensively; a chart that cannot be built is
    skipped rather than taking the whole page down."""
    page = {"id": pid, "title": title, "subtitle": subtitle,
            "kpis": [], "charts": [], "message": ""}
    try:
        page["kpis"] = kpis(data) or []
    except Exception:
        page["kpis"] = []
    for build in builders:
        if len(page["charts"]) >= MAX_CHARTS_PER_PAGE:
            break
        try:
            c = build(data)
        except Exception:
            c = None
        if c:
            page["charts"].append(c)
    for i, c in enumerate(page["charts"]):
        c["level"] = "core" if i < CORE_CHARTS else "adv"
    if not page["charts"]:
        page["message"] = missing or "Not enough information in the data for this page."
    return page


# ─────────────────────────── domain detection ─────────────────────────────

DOMAIN_LABELS = {
    "sales": "Sales & Commerce",
    "web": "Web & Product Analytics",
    "people": "People & HR",
    "finance": "Finance & Transactions",
    "timeseries": "Time Series & Operations",
    "generic": "General Dataset",
}

_SIGNALS = {
    "sales": {
        "core": ["sales", "revenue", "profit", "amount", "price", "order_id", "invoice"],
        "support": ["cost", "discount", "quantity", "qty", "customer", "product", "sku",
                    "category", "segment", "order", "margin", "unit_price", "subtotal"],
    },
    "web": {
        "core": ["session", "sessions", "pageview", "page_views", "pageviews", "visits",
                 "visitors", "traffic_source", "trafficsource", "bounce_rate", "bounces"],
        "support": ["source", "medium", "campaign", "referrer", "channel", "device",
                    "browser", "os", "landing_page", "exit_page", "page", "clicks",
                    "impressions", "ctr", "conversion", "conversions", "duration",
                    "time_on_page", "session_duration", "new_user", "returning"],
    },
    "people": {
        "core": ["employee", "employee_id", "staff", "attrition", "headcount", "job_role",
                 "jobrole", "department"],
        "support": ["salary", "wage", "compensation", "tenure", "hire_date", "joining_date",
                    "performance", "rating", "manager", "promotion", "designation",
                    "experience", "education", "resigned", "termination", "overtime"],
    },
    "finance": {
        "core": ["balance", "transaction", "account", "credit", "debit", "portfolio",
                 "interest_rate", "loan"],
                "support": ["amount", "category", "merchant", "payee", "narration",
                    "open", "high", "low", "close", "volume", "ticker", "symbol",
                    "return", "yield", "principal", "emi", "tenure", "default",
                    "currency", "exchange_rate"],
    },
}


def _score_keywords(df, spec):
    """Weighted keyword score: core signals count double, and a domain needs
    more than one distinct hit to register at all."""
    names = [_norm(c) for c in df.columns]
    tokens = set()
    for n in names:
        tokens.add(n)
        tokens.update(n.split("_"))

    def hit(kw):
        if kw in tokens:
            return True
        return any(kw in n for n in names if len(kw) > 4)

    core = sum(1 for kw in spec["core"] if hit(kw))
    support = sum(1 for kw in spec["support"] if hit(kw))
    if core == 0 and support < 3:
        return 0.0
    raw = core * 2.0 + support * 0.6
    return min(raw / 6.0, 1.0)


def _score_timeseries(df, prof):
    """A dataset is time-series-shaped when a date column covers most rows,
    there is at least one numeric metric, and rows are mostly one per period
    rather than many transactions per period."""
    if not prof["datetime"] or not (prof["numeric"] or prof["discrete"]):
        return 0.0
    col = prof["datetime"][0]
    dt = _to_datetime(df[col]).dropna()
    if len(dt) < 10:
        return 0.0
    coverage = len(dt) / max(len(df), 1)
    uniqueness = dt.nunique() / max(len(dt), 1)
    few_cats = 1.0 if len(prof["categorical"]) + len(prof["high_card"]) <= 3 else 0.5
    return min(coverage * uniqueness * few_cats * 1.2, 1.0)


def detect_domain(df):
    """Return [(key, label, confidence)] ranked best-first. 'generic' always
    appears last as a guaranteed fallback, so there is never a dead end."""
    prof = profile_columns(df)
    scores = []
    for key, spec in _SIGNALS.items():
        scores.append((key, DOMAIN_LABELS[key], _score_keywords(df, spec)))
        # price data (open/high/low/close) is unmistakably finance
    names = {_norm(c) for c in df.columns}
    if len(names & {"open", "high", "low", "close", "adj_close"}) >= 3:
        scores = [(k, l, 1.0 if k == "finance" else c) for k, l, c in scores]
    # Almost any dataset with dates looks time-series-shaped. A dated bank
    # statement or HR file is still a bank statement or HR file, so time
    # series only wins outright when no named domain is a reasonable fit.
    best_named = max((c for _, _, c in scores), default=0.0)
    ts_weight = 0.95 if best_named < 0.5 else 0.45
    scores.append(("timeseries", DOMAIN_LABELS["timeseries"], _score_timeseries(df, prof) * ts_weight))
    scores = [s for s in scores if s[2] > 0.15]
    scores.sort(key=lambda x: x[2], reverse=True)
    scores.append(("generic", DOMAIN_LABELS["generic"], 0.0))
    return scores


# ─────────────────────────── generic dashboard ────────────────────────────
# Works on literally any tabular data. This is what makes "it handles
# everything" true: when no specialised domain fits, this still has substance.

def _g_types(ctx):
    prof = ctx["prof"]
    counts = {"Numeric": len(prof["numeric"]) + len(prof["discrete"]),
              "Text / category": len(prof["categorical"]) + len(prof["high_card"]),
              "Date": len(prof["datetime"]),
              "ID-like": len(prof["id_like"])}
    counts = {k: v for k, v in counts.items() if v}
    if len(counts) < 2:
        return None
    fig = go.Figure(go.Pie(labels=list(counts), values=list(counts.values()),
                           hole=0.55, textinfo="percent"))
    biggest = max(counts, key=counts.get)
    return _chart(
        "What kind of columns are in this data?", _layout(fig, legend=True),
        f"Mostly {biggest.lower()} columns ({counts[biggest]} of {sum(counts.values())}).",
        "Column type breakdown after classification; ID-like columns are excluded from analysis.")


def _g_missing(ctx):
    df = ctx["df"]
    miss = df.isna().mean().mul(100)
    miss = miss[miss > 0].sort_values(ascending=False).head(15)
    if miss.empty:
        return None
    fig = go.Figure(go.Bar(x=miss.values, y=[_clip(i, 22) for i in miss.index],
                           orientation="h", marker_color=BAD))
    fig.update_xaxes(title_text="% missing", ticksuffix="%")
    return _chart(
        "Which columns have gaps?", _layout(fig),
        f"{len(miss)} column(s) have missing values. '{_clip(miss.index[0], 20)}' is the worst at {miss.iloc[0]:.0f}%.",
        f"Share of missing values per column, top {len(miss)}. Columns above ~40% missing are usually worth dropping or flagging.")


def _g_cardinality(ctx):
    df, prof = ctx["df"], ctx["prof"]
    cols = prof["categorical"] + prof["high_card"]
    if len(cols) < 2:
        return None
    card = df[cols].nunique().sort_values(ascending=False).head(12)
    fig = go.Figure(go.Bar(x=[_clip(i, 20) for i in card.index], y=card.values,
                           marker=dict(color=card.values, colorscale=SCALE)))
    fig.update_yaxes(title_text="Distinct values")
    return _chart(
        "How many different values per column?", _layout(fig),
        f"'{_clip(card.index[0], 20)}' has the most variety with {card.iloc[0]:,} distinct values.",
        "Distinct value counts for text columns. Very high counts behave like IDs; very low counts behave like flags.")


def _interesting_numeric(df, cols, n=4):
    """Rank numeric columns by relative spread, so flat columns don't win a
    slot over ones that actually vary."""
    scored = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 5 or s.nunique() < 3:
            continue
        m = abs(s.mean())
        cv = (s.std() / m) if m else s.std()
        scored.append((float(cv or 0), c))
    scored.sort(reverse=True)
    return [c for _, c in scored[:n]]


def _g_dist(ctx):
    df, prof = ctx["df"], ctx["prof"]
    cols = _interesting_numeric(df, prof["numeric"], n=1)
    if not cols:
        return None
    c = cols[0]
    s = pd.to_numeric(df[c], errors="coerce").dropna()
    skew = float(s.skew() or 0)
    if abs(skew) > 1.5:
        fig = go.Figure(go.Box(y=s.values, marker_color=ACCENT, boxpoints=False))
        note = (f"'{c}' is lopsided — most values sit near {_fmt(s.median())}, "
                f"but a few extreme ones stretch the average to {_fmt(s.mean())}.")
        tech = f"Box plot chosen over a histogram: skew = {skew:.2f}. Median {_fmt(s.median())}, IQR {_fmt(s.quantile(.75) - s.quantile(.25))}."
    else:
        fig = go.Figure(go.Histogram(x=s.values, nbinsx=40, marker_color=ACCENT))
        note = f"Most '{c}' values are around {_fmt(s.median())}, ranging from {_fmt(s.min())} to {_fmt(s.max())}."
        tech = f"Histogram, 40 bins. Mean {_fmt(s.mean())}, std {_fmt(s.std())}, skew {skew:.2f}."
    fig.update_xaxes(title_text=c)
    return _chart(f"Spread of {c}", _layout(fig), note, tech)


def _g_dist_more(ctx):
    """Small-multiples view of the next most variable numeric columns."""
    df, prof = ctx["df"], ctx["prof"]
    cols = _interesting_numeric(df, prof["numeric"], n=4)[1:4]
    if not cols:
        return None
    frames = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        z = (s - s.mean()) / (s.std() or 1)
        frames.append(pd.DataFrame({"value": z.values, "column": c}))
    if not frames:
        return None
    long = pd.concat(frames)
    fig = px.violin(long, x="column", y="value", box=True, color="column",
                    color_discrete_sequence=COLORWAY)
    fig.update_yaxes(title_text="Standardised value")
    return _chart(
        "Comparing the other numeric columns", _layout(fig),
        "Each column is rescaled so they can sit on one chart. Wider shapes mean values are more spread out.",
        f"Z-scored distributions of {', '.join(cols)}. Standardisation makes shape comparable, not magnitude.")


def _g_category(ctx):
    df, prof = ctx["df"], ctx["prof"]
    cols = prof["categorical"] + prof["high_card"]
    if not cols:
        return None
    # pick the most balanced column: one dominant value is a boring chart
    best, best_score = None, -1
    for c in cols:
        vc = df[c].value_counts(normalize=True)
        if len(vc) < 2:
            continue
        score = (1 - vc.iloc[0]) * min(len(vc), 10) / 10
        if score > best_score:
            best, best_score = c, score
    if best is None:
        return None
    s = df[best]
    bucketed = s.nunique() > TOP_N_BUCKET
    if bucketed:
        s = _bucket_top(s)
    vc = s.value_counts().head(10).iloc[::-1]   # largest ends up at the top
    fig = go.Figure(go.Bar(x=vc.values, y=[_clip(i, 22) for i in vc.index],
                           orientation="h", marker=dict(color=vc.values, colorscale=SCALE)))
    fig.update_xaxes(title_text="Rows")
    vc = vc.iloc[::-1]                           # back to largest-first for the note
    share = vc.iloc[0] / len(df)
    return _chart(
        f"Breakdown by {best}", _layout(fig),
        f"'{_clip(vc.index[0], 20)}' is the most common, covering {share:.0%} of all rows."
        + (" Smaller values are grouped as 'Other'." if bucketed else ""),
        f"Row counts per level of {best}" + (f", top {TOP_N_BUCKET} plus Other." if bucketed else ".")
        + f" {df[best].nunique():,} distinct values in total.")


def _g_category_2(ctx):
    df, prof = ctx["df"], ctx["prof"]
    cols = [c for c in prof["categorical"] if df[c].nunique() <= 8]
    if len(cols) < 2:
        return None
    c = cols[1]
    vc = df[c].value_counts()
    fig = go.Figure(go.Pie(labels=[_clip(i, 20) for i in vc.index], values=vc.values,
                           hole=0.55, textinfo="percent"))
    return _chart(
        f"Share by {c}", _layout(fig, legend=True),
        f"'{_clip(vc.index[0], 20)}' is the largest group at {vc.iloc[0] / vc.sum():.0%}.",
        f"Proportion of rows per level of {c}, {len(vc)} levels.")


def _g_corr(ctx):
    df, prof = ctx["df"], ctx["prof"]
    cols = prof["numeric"] + prof["discrete"]
    if len(cols) < 3:
        return None
    corr = df[cols].corr(numeric_only=True)
    if corr.empty or corr.shape[0] < 3:
        return None
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                    color_continuous_scale=[BAD, "#161616", GOOD])
    m = corr.where(~np.eye(len(corr), dtype=bool))
    try:
        flat = m.unstack().dropna()
        pair = flat.abs().idxmax()
        val = flat[pair]
        note = (f"'{pair[0]}' and '{pair[1]}' move together most strongly "
                f"({'same' if val > 0 else 'opposite'} direction, r = {val:.2f}).")
    except Exception:
        note = "Green means two columns rise together; red means one rises as the other falls."
    return _chart(
        "Which numbers move together?", fig, note,
        f"Pearson correlation matrix over {len(cols)} numeric columns. |r| > 0.7 between inputs signals multicollinearity.")


def _g_scatter(ctx):
    df, prof = ctx["df"], ctx["prof"]
    cols = prof["numeric"]
    if len(cols) < 2:
        return None
    corr = df[cols].corr(numeric_only=True)
    m = corr.where(~np.eye(len(corr), dtype=bool))
    try:
        flat = m.unstack().dropna()
        a, b = flat.abs().idxmax()
        r = float(flat[(a, b)])
    except Exception:
        return None
    sample = df[[a, b]].dropna()
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)
    if len(sample) < 10:
        return None
    fig = go.Figure(go.Scatter(x=sample[a], y=sample[b], mode="markers",
                               marker=dict(color=ACCENT, size=5, opacity=0.5)))
    fig.update_xaxes(title_text=a)
    fig.update_yaxes(title_text=b)
    strength = "strong" if abs(r) > 0.7 else ("moderate" if abs(r) > 0.4 else "weak")
    return _chart(
        f"{b} against {a}", _layout(fig),
        f"Each dot is a row. The link between them is {strength} (r = {r:.2f}).",
        f"Scatter of the most correlated numeric pair, r = {r:.2f}, n = {len(sample):,}"
        + (" (sampled)." if len(df) > 5000 else ".") + " Correlation is not causation.")


def _g_time_volume(ctx):
    df, prof = ctx["df"], ctx["prof"]
    if not prof["datetime"]:
        return None
    col = prof["datetime"][0]
    dt = _to_datetime(df[col]).dropna()
    if len(dt) < 5:
        return None
    span = (dt.max() - dt.min()).days
    freq = "D" if span <= 120 else ("W" if span <= 730 else "M")
    g = dt.dt.to_period(freq).value_counts().sort_index()
    if len(g) < 2:
        return None
    idx = [str(i) for i in g.index]
    fig = go.Figure(go.Scatter(x=idx, y=g.values, mode="lines", fill="tozeroy",
                               line=dict(color=ACCENT, width=2)))
    fig.update_yaxes(title_text="Records")
    label = {"D": "day", "W": "week", "M": "month"}[freq]
    return _chart(
        "How many records over time?", _layout(fig),
        f"Data runs from {dt.min():%b %Y} to {dt.max():%b %Y}. The busiest {label} had {g.max():,} records.",
        f"Record counts per {label} from '{col}', {len(g)} periods. Gaps here mean missing collection, not zero activity.")


def _g_time_metric(ctx):
    df, prof = ctx["df"], ctx["prof"]
    if not prof["datetime"] or not prof["numeric"]:
        return None
    col = prof["datetime"][0]
    metric = _interesting_numeric(df, prof["numeric"], n=1)
    if not metric:
        return None
    metric = metric[0]
    t = df[[col, metric]].copy()
    t[col] = _to_datetime(t[col])
    t = t.dropna()
    if len(t) < 5:
        return None
    span = (t[col].max() - t[col].min()).days
    freq = "D" if span <= 120 else ("W" if span <= 730 else "M")
    g = t.groupby(t[col].dt.to_period(freq))[metric].mean()
    if len(g) < 2:
        return None
    idx = [str(i) for i in g.index]
    fig = go.Figure()
    fig.add_scatter(x=idx, y=g.values, mode="lines+markers", name=metric,
                    line=dict(color=ACCENT, width=2))
    if len(g) >= 4:
        fig.add_scatter(x=idx, y=g.rolling(3, min_periods=1).mean().values,
                        mode="lines", name="3-period average",
                        line=dict(color=SAND, width=3, dash="dot"))
    fig.update_yaxes(title_text=metric)
    direction = "rising" if g.iloc[-1] > g.iloc[0] else "falling"
    return _chart(
        f"{metric} over time", _layout(fig, legend=len(g) >= 4),
        f"Average {metric} is {direction} overall, from {_fmt(g.iloc[0])} to {_fmt(g.iloc[-1])}.",
        f"Mean {metric} per period from '{col}'. Dotted line is a 3-period rolling mean; a short series makes the trend unreliable.")


def _g_seasonality(ctx):
    df, prof = ctx["df"], ctx["prof"]
    if not prof["datetime"]:
        return None
    col = prof["datetime"][0]
    dt = _to_datetime(df[col]).dropna()
    if len(dt) < 30 or (dt.max() - dt.min()).days < 14:
        return None
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    g = dt.dt.dayofweek.value_counts().reindex(range(7)).fillna(0)
    if (g > 0).sum() < 3:
        return None
    fig = go.Figure(go.Bar(x=days, y=g.values,
                           marker=dict(color=g.values, colorscale=SCALE)))
    fig.update_yaxes(title_text="Records")
    best = int(g.idxmax())
    return _chart(
        "Which day of the week is busiest?", _layout(fig),
        f"{days[best]} has the most activity, with {g.max() / g.sum():.0%} of all records.",
        f"Record counts by weekday from '{col}'. Check the source actually records real timestamps before reading into this.")


def _g_outliers(ctx):
    df, prof = ctx["df"], ctx["prof"]
    cols = prof["numeric"]
    if not cols:
        return None
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 20:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        rows.append((c, n_out / len(s) * 100))
    if not rows:
        return None
    rows.sort(key=lambda x: x[1], reverse=True)
    rows = rows[:12]
    fig = go.Figure(go.Bar(x=[r[1] for r in rows], y=[_clip(r[0], 22) for r in rows],
                           orientation="h",
                           marker_color=[BAD if r[1] > 5 else OLIVE for r in rows]))
    fig.update_xaxes(title_text="% unusual values", ticksuffix="%")
    worst = rows[0]
    return _chart(
        "Which columns have unusual values?", _layout(fig),
        f"'{_clip(worst[0], 20)}' has the most, with {worst[1]:.1f}% of its values far from the rest."
        if worst[1] > 0 else "No column has an unusual number of extreme values.",
        "Share of values outside 1.5×IQR per column. Red marks above 5%, which is worth investigating before modelling.")


def _g_duplicates(ctx):
    df = ctx["df"]
    dup = int(df.duplicated().sum())
    total = len(df)
    const = ctx["prof"]["constant"]
    if not dup and not const:
        return None
    if dup:
        fig = go.Figure(go.Bar(x=["Duplicate rows", "Unique rows"],
                               y=[dup, total - dup], marker_color=[BAD, GOOD]))
        fig.update_yaxes(title_text="Rows")
        return _chart(
            "Are rows repeated?", _layout(fig),
            f"{dup:,} of {total:,} rows ({dup / total:.1%}) are exact duplicates of another row.",
            f"Exact duplicate row count. {len(const)} constant column(s) also carry no information: {', '.join(const[:5]) or 'none'}.")
    fig = go.Figure(go.Bar(x=[_clip(c, 20) for c in const[:10]],
                           y=[1] * len(const[:10]), marker_color=MUTED))
    fig.update_yaxes(title_text="", showticklabels=False)
    return _chart(
        "Columns with only one value", _layout(fig),
        f"{len(const)} column(s) contain the same value in every row, so they tell you nothing.",
        "Constant columns should be dropped before modelling; they add dimensions without information.")


def _kpi_generic_overview(ctx):
    df, prof = ctx["df"], ctx["prof"]
    total = max(df.shape[0] * df.shape[1], 1)
    miss = df.isna().sum().sum() / total * 100
    dup = df.duplicated().mean() * 100
    health = int(max(0, min(100, round(100 - 2 * miss - 2 * dup))))
    return [
        ("Rows", f"{len(df):,}"),
        ("Columns", f"{df.shape[1]:,}"),
        ("Data health", f"{health}/100"),
        ("Missing", f"{miss:.1f}%"),
        ("Duplicate rows", f"{dup:.1f}%"),
        ("Usable columns", f"{len(prof['numeric']) + len(prof['discrete']) + len(prof['categorical']) + len(prof['high_card']):,}"),
    ]


def _kpi_generic_numeric(ctx):
    df, prof = ctx["df"], ctx["prof"]
    cols = prof["numeric"] + prof["discrete"]
    if not cols:
        return [("Numeric columns", "0")]
    out = [("Numeric columns", f"{len(cols):,}")]
    best = _interesting_numeric(df, prof["numeric"], n=1)
    if best:
        s = pd.to_numeric(df[best[0]], errors="coerce").dropna()
        out += [("Most variable", _clip(best[0], 18)),
                ("Its average", _fmt(s.mean())),
                ("Its range", f"{_fmt(s.min())} – {_fmt(s.max())}")]
    return out


def _kpi_generic_category(ctx):
    df, prof = ctx["df"], ctx["prof"]
    cols = prof["categorical"] + prof["high_card"]
    if not cols:
        return [("Text columns", "0")]
    card = df[cols].nunique().sort_values(ascending=False)
    return [("Text columns", f"{len(cols):,}"),
            ("Most varied", _clip(card.index[0], 18)),
            ("Its distinct values", f"{card.iloc[0]:,}"),
            ("Least varied", _clip(card.index[-1], 18))]


def _kpi_generic_time(ctx):
    df, prof = ctx["df"], ctx["prof"]
    if not prof["datetime"]:
        return []
    dt = _to_datetime(df[prof["datetime"][0]]).dropna()
    if dt.empty:
        return []
    span = (dt.max() - dt.min()).days
    return [("Date column", _clip(prof["datetime"][0], 18)),
            ("From", f"{dt.min():%d %b %Y}"),
            ("To", f"{dt.max():%d %b %Y}"),
            ("Span", f"{span:,} days")]


def _kpi_generic_quality(ctx):
    df, prof = ctx["df"], ctx["prof"]
    miss_cols = int((df.isna().mean() > 0).sum())
    return [("Columns with gaps", f"{miss_cols:,}"),
            ("Constant columns", f"{len(prof['constant']):,}"),
            ("ID-like columns", f"{len(prof['id_like']):,}"),
            ("Duplicate rows", f"{int(df.duplicated().sum()):,}")]


def _build_generic(df, prof):
    ctx = {"df": df, "prof": prof}
    pages = [
        _page("overview", "Dataset Overview", "What is in this data?",
              _kpi_generic_overview, [_g_types, _g_cardinality, _g_category, _g_dist], ctx),
        _page("numbers", "Numeric Columns", "How are the numbers spread out?",
              _kpi_generic_numeric, [_g_dist, _g_dist_more, _g_outliers], ctx,
              missing="This dataset has no numeric columns to summarise."),
        _page("categories", "Categories", "What are the main groups?",
              _kpi_generic_category, [_g_category, _g_category_2], ctx,
              missing="This dataset has no text or category columns to break down."),
        _page("relationships", "Relationships", "Which columns are connected?",
              lambda c: [], [_g_corr, _g_scatter], ctx,
              missing="At least three numeric columns are needed to look for relationships."),
    ]
    if prof["datetime"]:
        pages.append(_page("time", "Over Time", "How does this data change over time?",
                           _kpi_generic_time, [_g_time_volume, _g_time_metric, _g_seasonality], ctx,
                           missing="No usable date column was found."))
    pages.append(_page("quality", "Data Quality", "What should you fix before trusting this?",
                       _kpi_generic_quality, [_g_missing, _g_outliers, _g_duplicates], ctx,
                       missing="No quality problems were detected in this dataset."))
    return pages


# ═════════════════════ DOMAIN-SPECIFIC DASHBOARDS ═════════════════════════

_DATE_ROLES = {"date", "hire_date", "exit_date"}


def _map_roles(df, rules, numeric_roles=(), date_roles=()):
    """Map columns to domain roles by name, checking the column's type can
    actually carry that role. A column is claimed by one role only."""
    names = {c: _norm(c) for c in df.columns}
    used, found = set(), {}
    for role, kws in rules.items():
        for kw in kws:
            hit = None
            for c in df.columns:
                if c in used:
                    continue
                n = names[c]
                if n == kw or kw in n.split("_") or (len(kw) > 4 and kw in n):
                    s = df[c]
                    if role in numeric_roles:
                        if pd.to_numeric(s, errors="coerce").notna().mean() < 0.6:
                            continue
                    if role in date_roles:
                        try:
                            if _to_datetime(s).notna().mean() < 0.6:
                                continue
                        except Exception:
                            continue
                    hit = c
                    break
            if hit is not None:
                found[role] = hit
                used.add(hit)
                break
    return found


def _clean_cat(s):
    s = s.astype(str).str.strip()
    return s.where(~s.isin(["", "nan", "None", "NaN", "<NA>"]), np.nan)


def _as_rate(s):
    """Bounce/conversion columns arrive as either 0-1 or 0-100. Normalise to
    percent so the axis label is never a lie."""
    s = pd.to_numeric(s, errors="coerce")
    top = s.max()
    return s * 100 if pd.notna(top) and top <= 1.5 else s


def _period_freq(dt):
    span = (dt.max() - dt.min()).days
    if span <= 3:
        return "h", "hour"
    if span <= 120:
        return "D", "day"
    if span <= 730:
        return "W", "week"
    return "M", "month"


def _hbar(g, title_x="Value", color=None):
    g = g.iloc[::-1]
    if color:
        bar = go.Bar(x=g.values, y=[_clip(i, 24) for i in g.index], orientation="h",
                     marker_color=color)
    else:
        bar = go.Bar(x=g.values, y=[_clip(i, 24) for i in g.index], orientation="h",
                     marker=dict(color=g.values, colorscale=SCALE))
    fig = go.Figure(bar)
    fig.update_xaxes(title_text=title_x)
    return _layout(fig)


# ══════════════════════════ WEB & PRODUCT ANALYTICS ═══════════════════════

_WEB_RULES = {
    "date": ["date", "timestamp", "datetime", "visit_date", "session_date", "day"],
    "sessions": ["sessions", "session_count", "visits", "visit_count"],
    "pageviews": ["pageviews", "page_views", "pageview", "views", "screen_views"],
    "bounce": ["bounce_rate", "bounces", "bounce"],
    "duration": ["session_duration", "avg_session_duration", "time_on_page",
                 "time_on_site", "duration", "avg_time"],
    "conversions": ["conversions", "conversion", "goal_completions", "signups", "purchases"],
    "revenue": ["revenue", "transaction_revenue", "order_value"],
    "source": ["traffic_source", "trafficsource", "source", "channel", "medium",
               "utm_source", "referrer"],
    "campaign": ["campaign", "utm_campaign", "ad_group"],
    "device": ["device", "device_category", "devicetype", "platform"],
    "browser": ["browser"],
    "os": ["os", "operating_system"],
    "country": ["country", "geo_country", "location"],
    "page": ["page_path", "landing_page", "page_title", "page", "url"],
    "visitor_type": ["user_type", "new_returning", "visitor_type", "is_new", "returning"],
    "users": ["users", "visitors", "unique_visitors", "user_id", "visitor_id", "client_id"],
}
_WEB_NUM = {"sessions", "pageviews", "bounce", "duration", "conversions", "revenue"}


def _prep_web(df, roles):
    d = pd.DataFrame(index=df.index)
    if "date" in roles:
        d["_date"] = _to_datetime(df[roles["date"]])
    for r in ("sessions", "pageviews", "duration", "conversions", "revenue"):
        if r in roles:
            s = pd.to_numeric(df[roles[r]], errors="coerce")
            if s.notna().mean() > 0.5:
                d["_" + r] = s
    if "bounce" in roles:
        d["_bounce"] = _as_rate(df[roles["bounce"]])
    for r in ("source", "campaign", "device", "browser", "os", "country", "page", "visitor_type"):
        if r in roles:
            d["_" + r] = _clean_cat(df[roles[r]])
    if "users" in roles:
        u = df[roles["users"]]
        if pd.api.types.is_numeric_dtype(u) and u.nunique() < len(u):
            d["_users"] = pd.to_numeric(u, errors="coerce")
        else:
            d["_user_id"] = u.astype(str)
    return d


def _w_sessions(d):
    """Each row is one session unless a pre-aggregated sessions column exists."""
    return d["_sessions"] if "_sessions" in d else pd.Series(1, index=d.index)


def _w_total_sessions(d):
    return float(_w_sessions(d).sum())


def _kpi_web_traffic(d):
    out = [("Sessions", _fmt(_w_total_sessions(d)))]
    if "_users" in d:
        out.append(("Users", _fmt(d["_users"].sum())))
    elif "_user_id" in d:
        out.append(("Users", f"{d['_user_id'].nunique():,}"))
    if "_pageviews" in d:
        pv = d["_pageviews"].sum()
        out.append(("Pageviews", _fmt(pv)))
        out.append(("Pages / session", f"{pv / max(_w_total_sessions(d), 1):.1f}"))
    if "_duration" in d:
        avg = d["_duration"].mean()
        out.append(("Avg duration", f"{avg / 60:.1f} min" if avg > 180 else f"{avg:.0f}s"))
    if "_bounce" in d:
        out.append(("Bounce rate", f"{d['_bounce'].mean():.0f}%"))
    if "_conversions" in d:
        out.append(("Conv. rate", f"{d['_conversions'].sum() / max(_w_total_sessions(d), 1):.1%}"))
    return out[:6]


def _w_traffic_time(d):
    if "_date" not in d:
        return None
    t = d.dropna(subset=["_date"]).copy()
    if t.empty:
        return None
    t["_s"] = _w_sessions(t)
    dt = t["_date"]
    freq, label = _period_freq(dt)
    g = t.groupby(dt.dt.to_period(freq))["_s"].sum()
    if len(g) < 2:
        return None
    idx = [str(i) for i in g.index]
    fig = go.Figure(go.Scatter(x=idx, y=g.values, mode="lines", fill="tozeroy",
                               line=dict(color=ACCENT, width=2)))
    fig.update_yaxes(title_text="Sessions")
    peak = g.idxmax()
    trend = "up" if g.iloc[-len(g) // 3:].mean() > g.iloc[:len(g) // 3].mean() else "down"
    return _chart(
        "Traffic over time", _layout(fig),
        f"Busiest {label}: {peak} with {_fmt(g.max())} sessions. Overall the trend is {trend}.",
        f"Sessions per {label} over {len(g)} periods. Mean {_fmt(g.mean())}, "
        f"coefficient of variation {(g.std() / g.mean() if g.mean() else 0):.0%}.")


def _w_by_source(d):
    if "_source" not in d:
        return None
    t = d.dropna(subset=["_source"]).copy()
    t["_s"] = _w_sessions(t)
    g = t.groupby("_source")["_s"].sum().sort_values(ascending=False).head(10)
    if len(g) < 2:
        return None
    total = g.sum()
    return _chart(
        "Where does traffic come from?", _hbar(g, "Sessions"),
        f"'{_clip(g.index[0], 20)}' sends the most traffic, {g.iloc[0] / total:.0%} of all sessions.",
        f"Sessions by acquisition source, top {len(g)} of {t['_source'].nunique()}. "
        f"Top 3 account for {g.head(3).sum() / total:.0%}.")


def _w_source_time(d):
    if "_source" not in d or "_date" not in d:
        return None
    t = d.dropna(subset=["_source", "_date"]).copy()
    if t.empty:
        return None
    t["_s"] = _w_sessions(t)
    top = t.groupby("_source")["_s"].sum().nlargest(5).index
    t = t[t["_source"].isin(top)]
    freq, label = _period_freq(t["_date"])
    g = t.groupby([t["_date"].dt.to_period(freq), "_source"])["_s"].sum().unstack(fill_value=0)
    if len(g) < 3:
        return None
    fig = go.Figure()
    for i, col in enumerate(g.columns):
        fig.add_scatter(x=[str(x) for x in g.index], y=g[col].values, name=_clip(col, 18),
                        mode="lines", stackgroup="one",
                        line=dict(color=COLORWAY[i % len(COLORWAY)], width=1))
    fig.update_yaxes(title_text="Sessions")
    return _chart(
        "How each source changes over time", _layout(fig, legend=True),
        "Each band is one traffic source. A band that widens is growing; one that narrows is shrinking.",
        f"Stacked sessions per {label} for the top {len(g.columns)} sources. "
        "Stacking shows total and mix together, but makes individual trends harder to read.")


def _w_device(d):
    for key, title in (("_device", "Which devices do people use?"),
                       ("_browser", "Which browsers do people use?"),
                       ("_os", "Which operating systems?")):
        if key in d:
            t = d.dropna(subset=[key]).copy()
            t["_s"] = _w_sessions(t)
            g = t.groupby(key)["_s"].sum().sort_values(ascending=False).head(7)
            if len(g) < 2:
                continue
            fig = go.Figure(go.Pie(labels=[_clip(i, 18) for i in g.index], values=g.values,
                                   hole=0.55, textinfo="percent"))
            share = g.iloc[0] / g.sum()
            return _chart(
                title, _layout(fig, legend=True),
                f"'{_clip(g.index[0], 18)}' leads with {share:.0%} of sessions.",
                f"Session share by {key.strip('_')}, top {len(g)} values.")
    return None


def _w_engagement(d):
    """Bounce rate or session duration broken down by source — the chart
    that tells you whether traffic is actually any good."""
    metric = "_bounce" if "_bounce" in d else ("_duration" if "_duration" in d else None)
    key = "_source" if "_source" in d else ("_device" if "_device" in d else None)
    if metric is None or key is None:
        return None
    t = d.dropna(subset=[metric, key])
    g = t.groupby(key)[metric].agg(["mean", "count"])
    g = g[g["count"] >= max(5, len(t) * 0.01)]
    if len(g) < 2:
        return None
    g = g.sort_values("mean", ascending=False).head(10)
    is_bounce = metric == "_bounce"
    colors = [BAD if v > g["mean"].median() else GOOD for v in g["mean"]] if is_bounce else \
             [GOOD if v > g["mean"].median() else SAND for v in g["mean"]]
    fig = go.Figure(go.Bar(x=[_clip(i, 16) for i in g.index], y=g["mean"].values,
                           marker_color=colors))
    fig.update_yaxes(title_text="Bounce rate" if is_bounce else "Avg duration (s)",
                     ticksuffix="%" if is_bounce else "")
    name = key.strip("_")
    worst, best = g.index[0], g.index[-1]
    note = (f"'{_clip(worst, 18)}' has the highest bounce rate ({g['mean'].iloc[0]:.0f}%) — "
            f"that traffic leaves fastest. '{_clip(best, 18)}' sticks around most."
            if is_bounce else
            f"'{_clip(worst, 18)}' visitors stay longest ({g['mean'].iloc[0]:.0f}s), "
            f"'{_clip(best, 18)}' the shortest.")
    return _chart(
        f"Engagement by {name}", _layout(fig), note,
        f"Mean {'bounce rate' if is_bounce else 'session duration'} per {name}, "
        f"groups under 1% of rows hidden. Averages hide the spread; check the distribution too.")


def _w_duration_dist(d):
    col = "_duration" if "_duration" in d else ("_pageviews" if "_pageviews" in d else None)
    if col is None:
        return None
    s = d[col].dropna()
    s = s[s > 0]
    if len(s) < 30:
        return None
    cap = s.quantile(0.99)
    fig = go.Figure(go.Histogram(x=s[s <= cap].values, nbinsx=40, marker_color=OLIVE))
    name = "Session duration" if col == "_duration" else "Pageviews per session"
    fig.update_xaxes(title_text=name)
    fig.update_yaxes(title_text="Sessions")
    return _chart(
        "How long do visits last?" if col == "_duration" else "How many pages per visit?",
        _layout(fig),
        f"A typical visit is {_fmt(s.median())}{'s' if col == '_duration' else ' pages'}, "
        f"but a few long ones pull the average up to {_fmt(s.mean())}.",
        f"{name} distribution clipped at the 99th percentile. "
        f"Median {_fmt(s.median())}, mean {_fmt(s.mean())}, skew {s.skew():.1f}.")


def _w_conv_by(d):
    if "_conversions" not in d:
        return None
    key = "_source" if "_source" in d else ("_device" if "_device" in d else None)
    if key is None:
        return None
    t = d.dropna(subset=[key]).copy()
    t["_s"] = _w_sessions(t)
    g = t.groupby(key).agg(conv=("_conversions", "sum"), sess=("_s", "sum"))
    g = g[g["sess"] >= max(5, g["sess"].sum() * 0.01)]
    if len(g) < 2:
        return None
    g["rate"] = g["conv"] / g["sess"] * 100
    g = g.sort_values("rate", ascending=False).head(10)
    fig = go.Figure(go.Bar(x=[_clip(i, 16) for i in g.index], y=g["rate"].values,
                           marker_color=[GOOD if v >= g["rate"].median() else SAND
                                         for v in g["rate"]],
                           text=[f"{v:.1f}%" for v in g["rate"]], textposition="outside"))
    fig.update_yaxes(title_text="Conversion rate", ticksuffix="%")
    name = key.strip("_")
    return _chart(
        f"Conversion rate by {name}", _layout(fig),
        f"'{_clip(g.index[0], 18)}' converts best at {g['rate'].iloc[0]:.1f}%, "
        f"against {g['rate'].iloc[-1]:.1f}% for '{_clip(g.index[-1], 18)}'.",
        f"Conversions divided by sessions per {name}. Rates on small session counts are noisy; "
        f"groups under 1% of traffic are hidden.")


def _w_conv_time(d):
    if "_conversions" not in d or "_date" not in d:
        return None
    t = d.dropna(subset=["_date"]).copy()
    t["_s"] = _w_sessions(t)
    freq, label = _period_freq(t["_date"])
    g = t.groupby(t["_date"].dt.to_period(freq)).agg(conv=("_conversions", "sum"),
                                                     sess=("_s", "sum"))
    g = g[g["sess"] > 0]
    if len(g) < 2:
        return None
    rate = g["conv"] / g["sess"] * 100
    idx = [str(i) for i in g.index]
    fig = go.Figure()
    fig.add_bar(x=idx, y=g["conv"].values, name="Conversions", marker_color=ACCENT)
    fig.add_scatter(x=idx, y=rate.values, name="Rate %", mode="lines+markers",
                    yaxis="y2", line=dict(color=GOOD, width=3))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", ticksuffix="%",
                                  showgrid=False))
    fig.update_yaxes(title_text="Conversions")
    return _chart(
        "Conversions over time", _layout(fig, legend=True),
        f"Best {label}: {g['conv'].idxmax()} with {_fmt(g['conv'].max())} conversions. "
        f"The line shows the rate, which can fall even while the count rises.",
        f"Conversion counts (bars) and rate (line) per {label}. "
        f"Mean rate {rate.mean():.2f}%, std {rate.std():.2f}pp.")


def _w_pages(d):
    if "_page" not in d:
        return None
    t = d.dropna(subset=["_page"]).copy()
    t["_s"] = _w_sessions(t)
    g = t.groupby("_page")["_s"].sum().sort_values(ascending=False).head(10)
    if len(g) < 2:
        return None
    return _chart(
        "Most visited pages", _hbar(g, "Sessions"),
        f"'{_clip(g.index[0], 22)}' is the most visited, with {g.iloc[0] / g.sum():.0%} of listed traffic.",
        f"Sessions per page, top {len(g)} of {t['_page'].nunique():,} pages.")


def _w_geo(d):
    if "_country" not in d:
        return None
    t = d.dropna(subset=["_country"]).copy()
    t["_s"] = _w_sessions(t)
    g = t.groupby("_country")["_s"].sum().sort_values(ascending=False).head(10)
    if len(g) < 2:
        return None
    return _chart(
        "Where are visitors from?", _hbar(g, "Sessions"),
        f"'{_clip(g.index[0], 20)}' leads with {g.iloc[0] / g.sum():.0%} of sessions.",
        f"Sessions by country, top {len(g)} of {t['_country'].nunique()}.")


def _w_visitor_type(d):
    if "_visitor_type" not in d:
        return None
    t = d.dropna(subset=["_visitor_type"]).copy()
    t["_s"] = _w_sessions(t)
    g = t.groupby("_visitor_type")["_s"].sum()
    if len(g) < 2:
        return None
    fig = go.Figure(go.Pie(labels=[_clip(i, 18) for i in g.index], values=g.values,
                           hole=0.55, textinfo="percent",
                           marker=dict(colors=[GOOD, ACCENT, SAND, OLIVE])))
    return _chart(
        "New versus returning visitors", _layout(fig, legend=True),
        f"'{_clip(g.idxmax(), 18)}' makes up {g.max() / g.sum():.0%} of sessions.",
        "Session share by visitor type. A very low returning share suggests weak retention.")


def _w_hour(d):
    if "_date" not in d:
        return None
    dt = d["_date"].dropna()
    if len(dt) < 30 or dt.dt.hour.nunique() < 4:
        return None
    g = dt.dt.hour.value_counts().reindex(range(24)).fillna(0)
    fig = go.Figure(go.Bar(x=list(range(24)), y=g.values,
                           marker=dict(color=g.values, colorscale=SCALE)))
    fig.update_xaxes(title_text="Hour of day", dtick=3)
    fig.update_yaxes(title_text="Sessions")
    peak = int(g.idxmax())
    return _chart(
        "What time of day is busiest?", _layout(fig),
        f"Traffic peaks around {peak:02d}:00, which carries {g.max() / g.sum():.0%} of all sessions.",
        "Session counts by hour of day. Check the timestamps are in the viewer's timezone before acting on this.")


def _w_weekday(d):
    if "_date" not in d:
        return None
    dt = d["_date"].dropna()
    if len(dt) < 20 or (dt.max() - dt.min()).days < 10:
        return None
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    g = dt.dt.dayofweek.value_counts().reindex(range(7)).fillna(0)
    fig = go.Figure(go.Bar(x=days, y=g.values, marker=dict(color=g.values, colorscale=SCALE)))
    fig.update_yaxes(title_text="Sessions")
    best = int(g.idxmax())
    weekend = g[5:].sum() / g.sum() if g.sum() else 0
    return _chart(
        "Which day of the week is busiest?", _layout(fig),
        f"{days[best]} is the busiest day. Weekends carry {weekend:.0%} of traffic.",
        f"Sessions by weekday. Weekend share {weekend:.1%} against a 28.6% flat expectation.")


def _kpi_web_acq(d):
    out = []
    if "_source" in d:
        t = d.dropna(subset=["_source"]).copy()
        t["_s"] = _w_sessions(t)
        g = t.groupby("_source")["_s"].sum().sort_values(ascending=False)
        out += [("Sources", f"{len(g):,}"), ("Top source", _clip(g.index[0], 16)),
                ("Its share", f"{g.iloc[0] / g.sum():.0%}")]
    if "_campaign" in d:
        out.append(("Campaigns", f"{d['_campaign'].nunique():,}"))
    return out or [("Sessions", _fmt(_w_total_sessions(d)))]


def _kpi_web_eng(d):
    out = []
    if "_bounce" in d:
        out.append(("Bounce rate", f"{d['_bounce'].mean():.0f}%"))
    if "_duration" in d:
        s = d["_duration"].dropna()
        out += [("Avg duration", f"{s.mean():.0f}s"), ("Median duration", f"{s.median():.0f}s")]
    if "_pageviews" in d:
        out.append(("Pages / session",
                    f"{d['_pageviews'].sum() / max(_w_total_sessions(d), 1):.1f}"))
    return out or [("Sessions", _fmt(_w_total_sessions(d)))]


def _kpi_web_conv(d):
    if "_conversions" not in d:
        return []
    conv = d["_conversions"].sum()
    sess = max(_w_total_sessions(d), 1)
    out = [("Conversions", _fmt(conv)), ("Conv. rate", f"{conv / sess:.2%}")]
    if "_revenue" in d:
        rev = d["_revenue"].sum()
        out += [("Revenue", _fmt(rev)), ("Revenue / session", _fmt(rev / sess))]
    return out


def _kpi_web_aud(d):
    out = []
    for key, label in (("_device", "Devices"), ("_browser", "Browsers"),
                       ("_country", "Countries")):
        if key in d:
            out.append((label, f"{d[key].nunique():,}"))
    if "_user_id" in d:
        out.append(("Unique users", f"{d['_user_id'].nunique():,}"))
    elif "_users" in d:
        out.append(("Users", _fmt(d["_users"].sum())))
    return out


def _build_web(df, prof):
    roles = _map_roles(df, _WEB_RULES, numeric_roles=_WEB_NUM, date_roles={"date"})
    d = _prep_web(df, roles)
    return [
        _page("traffic", "Traffic Overview", "How much traffic is there, and is it growing?",
              _kpi_web_traffic, [_w_traffic_time, _w_by_source, _w_device, _w_weekday,
                                 _w_hour, _w_visitor_type], d,
              missing="No traffic metrics found. Check the dataset has sessions, visits or a date column."),
        _page("acquisition", "Acquisition", "Where does the traffic come from?",
              _kpi_web_acq, [_w_by_source, _w_source_time, _w_conv_by, _w_geo], d,
              missing="This page needs a traffic source, channel or campaign column."),
        _page("engagement", "Engagement", "Is the traffic any good once it arrives?",
              _kpi_web_eng, [_w_engagement, _w_duration_dist, _w_visitor_type], d,
              missing="This page needs a bounce rate, session duration or pageviews column."),
        _page("audience", "Audience", "Who is visiting?",
              _kpi_web_aud, [_w_device, _w_geo, _w_visitor_type], d,
              missing="This page needs a device, browser, OS or country column."),
        _page("conversion", "Conversion", "Does the traffic turn into anything?",
              _kpi_web_conv, [_w_conv_time, _w_conv_by], d,
              missing="This page needs a conversions, goals or purchases column."),
        _page("content", "Content", "Which pages do people actually look at?",
              lambda x: [("Pages", f"{x['_page'].nunique():,}")] if "_page" in x else [],
              [_w_pages], d,
              missing="This page needs a page, URL or landing page column."),
    ]


# ═══════════════════════════════ PEOPLE & HR ══════════════════════════════

_HR_RULES = {
    "hire_date": ["hire_date", "hiredate", "joining_date", "joiningdate", "date_of_joining",
                  "dateofjoining", "start_date", "doj"],
    "exit_date": ["exit_date", "exitdate", "termination_date", "last_working_day", "leaving_date"],
    "department": ["department", "dept", "division", "function", "business_unit"],
    "role": ["job_role", "jobrole", "role", "designation", "position", "job_title", "title"],
    "salary": ["salary", "monthly_income", "monthlyincome", "annual_salary", "compensation",
               "wage", "pay", "ctc", "income"],
    "tenure": ["years_at_company", "yearsatcompany", "tenure", "years_of_service",
               "total_working_years", "experience", "years_experience"],
    "attrition": ["attrition", "left", "resigned", "churn", "turnover", "is_active", "status"],
    "performance": ["performance_rating", "performancerating", "performance", "rating",
                    "appraisal", "score"],
    "age": ["age"],
    "gender": ["gender", "sex"],
    "education": ["education", "education_level", "qualification", "degree"],
    "overtime": ["overtime", "over_time", "extra_hours"],
    "satisfaction": ["job_satisfaction", "jobsatisfaction", "satisfaction", "engagement_score"],
    "level": ["job_level", "joblevel", "grade", "band", "seniority"],
    "location": ["location", "office", "city", "site", "country"],
}
_HR_NUM = {"salary", "tenure", "performance", "age", "satisfaction", "level"}


def _prep_hr(df, roles):
    d = pd.DataFrame(index=df.index)
    for r in ("hire_date", "exit_date"):
        if r in roles:
            d["_" + r] = _to_datetime(df[roles[r]])
    for r in ("salary", "tenure", "performance", "age", "satisfaction", "level"):
        if r in roles:
            s = pd.to_numeric(df[roles[r]], errors="coerce")
            if s.notna().mean() > 0.5:
                d["_" + r] = s
    for r in ("department", "role", "gender", "education", "overtime", "location"):
        if r in roles:
            d["_" + r] = _clean_cat(df[roles[r]])
    if "attrition" in roles:
        s = df[roles["attrition"]]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() <= 3:
            d["_attrition"] = pd.to_numeric(s, errors="coerce").clip(0, 1)
        else:
            txt = s.astype(str).str.strip().str.lower()
            yes = {"yes", "y", "true", "1", "left", "resigned", "terminated", "inactive"}
            if txt.isin(yes | {"no", "n", "false", "0", "active", "current", "stayed"}).mean() > 0.7:
                d["_attrition"] = txt.isin(yes).astype(int)
            else:
                d["_attrition_label"] = _clean_cat(s)
    return d


def _kpi_hr_overview(d):
    out = [("Headcount", f"{len(d):,}")]
    if "_department" in d:
        out.append(("Departments", f"{d['_department'].nunique():,}"))
    if "_role" in d:
        out.append(("Roles", f"{d['_role'].nunique():,}"))
    if "_tenure" in d:
        out.append(("Avg tenure", f"{d['_tenure'].mean():.1f} yrs"))
    if "_age" in d:
        out.append(("Avg age", f"{d['_age'].mean():.0f}"))
    if "_attrition" in d:
        out.append(("Attrition rate", f"{d['_attrition'].mean():.1%}"))
    return out[:6]


def _h_headcount(d):
    key = "_department" if "_department" in d else ("_role" if "_role" in d else None)
    if key is None:
        return None
    g = d[key].value_counts().head(12)
    if len(g) < 2:
        return None
    name = key.strip("_")
    return _chart(
        f"Headcount by {name}", _hbar(g, "Employees"),
        f"'{_clip(g.index[0], 20)}' is the largest {name} with {g.iloc[0]:,} people "
        f"({g.iloc[0] / len(d):.0%} of staff).",
        f"Employee counts per {name}, top {len(g)} of {d[key].nunique()}.")


def _h_tenure_dist(d):
    if "_tenure" not in d:
        return None
    s = d["_tenure"].dropna()
    if len(s) < 20:
        return None
    fig = go.Figure(go.Histogram(x=s.values, nbinsx=25, marker_color=ACCENT))
    fig.update_xaxes(title_text="Years at company")
    fig.update_yaxes(title_text="Employees")
    newish = (s <= 2).mean()
    return _chart(
        "How long have people been here?", _layout(fig),
        f"Typical tenure is {s.median():.0f} years. {newish:.0%} of staff joined within the last 2 years.",
        f"Tenure distribution. Median {s.median():.1f}, mean {s.mean():.1f}, skew {s.skew():.2f}. "
        "A heavy left tail means a young workforce or high churn — attrition tells you which.")


def _h_age_dist(d):
    if "_age" not in d:
        return None
    s = d["_age"].dropna()
    if len(s) < 20:
        return None
    fig = go.Figure(go.Histogram(x=s.values, nbinsx=25, marker_color=OLIVE))
    fig.update_xaxes(title_text="Age")
    fig.update_yaxes(title_text="Employees")
    return _chart(
        "Age profile", _layout(fig),
        f"Most staff are around {s.median():.0f} years old, ranging from {s.min():.0f} to {s.max():.0f}.",
        f"Age distribution. Mean {s.mean():.1f}, std {s.std():.1f}.")


def _h_salary_dist(d):
    if "_salary" not in d:
        return None
    s = d["_salary"].dropna()
    s = s[s > 0]
    if len(s) < 20:
        return None
    cap = s.quantile(0.99)
    fig = go.Figure(go.Histogram(x=s[s <= cap].values, nbinsx=35, marker_color=ACCENT))
    fig.update_xaxes(title_text="Salary", tickformat="~s")
    fig.update_yaxes(title_text="Employees")
    ratio = s.quantile(0.9) / s.quantile(0.1) if s.quantile(0.1) else 0
    return _chart(
        "How is pay spread out?", _layout(fig),
        f"Half of staff earn under {_fmt(s.median())}. The top 10% earn more than {_fmt(s.quantile(0.9))}.",
        f"Salary distribution clipped at the 99th percentile. Median {_fmt(s.median())}, "
        f"mean {_fmt(s.mean())}, P90/P10 ratio {ratio:.1f}.")


def _h_salary_by(d):
    if "_salary" not in d:
        return None
    key = "_department" if "_department" in d else ("_role" if "_role" in d else
                                                    ("_level" if "_level" in d else None))
    if key is None:
        return None
    t = d.dropna(subset=["_salary", key])
    g = t.groupby(key)["_salary"].agg(["median", "count"])
    g = g[g["count"] >= 5]
    if len(g) < 2:
        return None
    g = g.sort_values("median", ascending=False).head(10)
    name = key.strip("_")
    gap = g["median"].iloc[0] / g["median"].iloc[-1] if g["median"].iloc[-1] else 0
    fig = go.Figure(go.Box(
        x=[_clip(v, 14) for v in t[t[key].isin(g.index)][key]],
        y=t[t[key].isin(g.index)]["_salary"], marker_color=ACCENT, boxpoints=False))
    fig.update_yaxes(title_text="Salary", tickformat="~s")
    return _chart(
        f"Pay by {name}", _layout(fig),
        f"'{_clip(g.index[0], 18)}' is paid most (median {_fmt(g['median'].iloc[0])}), "
        f"'{_clip(g.index[-1], 18)}' least — a {gap:.1f}x gap.",
        f"Salary distribution per {name}, groups under 5 people hidden. "
        "Box plots show spread, which averages hide; overlapping boxes mean the gap is not clear-cut.")


def _h_salary_tenure(d):
    if "_salary" not in d or "_tenure" not in d:
        return None
    t = d[["_salary", "_tenure"]].dropna()
    if len(t) < 20:
        return None
    if len(t) > 4000:
        t = t.sample(4000, random_state=42)
    r = t["_tenure"].corr(t["_salary"])
    fig = go.Figure(go.Scatter(x=t["_tenure"], y=t["_salary"], mode="markers",
                               marker=dict(color=ACCENT, size=5, opacity=0.45)))
    fig.update_xaxes(title_text="Years at company")
    fig.update_yaxes(title_text="Salary", tickformat="~s")
    strength = "strongly" if abs(r) > 0.6 else ("somewhat" if abs(r) > 0.3 else "barely")
    return _chart(
        "Does pay rise with time served?", _layout(fig),
        f"Pay is {strength} linked to tenure (r = {r:.2f}). "
        + ("Long service is rewarded." if r > 0.3 else
           "Time at the company is not what decides pay here."),
        f"Salary against tenure, n = {len(t):,}, Pearson r = {r:.2f}. "
        "Role and level usually explain more than tenure alone; this is not a controlled comparison.")


def _h_attrition_by(d):
    if "_attrition" not in d:
        return None
    for key in ("_department", "_role", "_overtime", "_gender", "_education", "_location"):
        if key not in d:
            continue
        t = d.dropna(subset=[key])
        g = t.groupby(key)["_attrition"].agg(["mean", "count"])
        g = g[g["count"] >= max(5, len(t) * 0.02)]
        if len(g) < 2:
            continue
        g = g.sort_values("mean", ascending=False).head(10)
        overall = d["_attrition"].mean()
        fig = go.Figure(go.Bar(
            x=[_clip(i, 16) for i in g.index], y=g["mean"].mul(100).values,
            marker_color=[BAD if v > overall else GOOD for v in g["mean"]],
            text=[f"{v:.0%}" for v in g["mean"]], textposition="outside"))
        fig.add_hline(y=overall * 100, line_dash="dash", line_color=MUTED)
        fig.update_yaxes(title_text="Attrition rate", ticksuffix="%")
        name = key.strip("_")
        return _chart(
            f"Who leaves? Attrition by {name}", _layout(fig),
            f"'{_clip(g.index[0], 18)}' loses the most people ({g['mean'].iloc[0]:.0%}), "
            f"against {overall:.0%} across the company (dashed line).",
            f"Attrition rate per {name}, groups under 2% of staff hidden. "
            "Descriptive only — no test for whether these differences are significant.")
    return None


def _h_attrition_tenure(d):
    if "_attrition" not in d or "_tenure" not in d:
        return None
    t = d[["_attrition", "_tenure"]].dropna()
    if len(t) < 30:
        return None
    bands = pd.cut(t["_tenure"], [-0.1, 1, 2, 5, 10, 20, 100],
                   labels=["<1 yr", "1-2", "2-5", "5-10", "10-20", "20+"])
    g = t.groupby(bands, observed=True)["_attrition"].agg(["mean", "count"])
    g = g[g["count"] >= 5]
    if len(g) < 2:
        return None
    overall = t["_attrition"].mean()
    fig = go.Figure(go.Bar(x=[str(i) for i in g.index], y=g["mean"].mul(100).values,
                           marker_color=[BAD if v > overall else GOOD for v in g["mean"]]))
    fig.add_hline(y=overall * 100, line_dash="dash", line_color=MUTED)
    fig.update_yaxes(title_text="Attrition rate", ticksuffix="%")
    fig.update_xaxes(title_text="Years at company")
    peak = g["mean"].idxmax()
    return _chart(
        "When do people leave?", _layout(fig),
        f"Risk is highest in the '{peak}' band ({g['mean'].max():.0%} leave). "
        "That is where retention effort pays off most.",
        f"Attrition rate by tenure band, bands under 5 people hidden. Overall rate {overall:.1%}. "
        "Survivorship matters here: current staff are not a random sample of everyone hired.")


def _h_perf_dist(d):
    if "_performance" not in d:
        return None
    s = d["_performance"].dropna()
    if s.nunique() < 2:
        return None
    g = s.value_counts().sort_index()
    fig = go.Figure(go.Bar(x=[str(i) for i in g.index], y=g.values,
                           marker=dict(color=g.values, colorscale=SCALE)))
    fig.update_xaxes(title_text="Performance rating")
    fig.update_yaxes(title_text="Employees")
    top = g.idxmax()
    return _chart(
        "How are performance ratings spread?", _layout(fig),
        f"Most people are rated {top} ({g.max() / g.sum():.0%} of staff). "
        + ("Ratings are heavily bunched, which limits how useful they are."
           if g.max() / g.sum() > 0.6 else "Ratings are reasonably spread out."),
        f"Rating distribution, {s.nunique()} levels, mean {s.mean():.2f}. "
        "Bunching at one value is a sign of rating inflation or a blunt scale.")


def _h_perf_vs(d):
    if "_performance" not in d:
        return None
    other = "_salary" if "_salary" in d else ("_satisfaction" if "_satisfaction" in d else None)
    if other is None:
        return None
    t = d[["_performance", other]].dropna()
    if len(t) < 20 or t["_performance"].nunique() < 2:
        return None
    g = t.groupby("_performance")[other].median()
    fig = go.Figure(go.Bar(x=[str(i) for i in g.index], y=g.values, marker_color=ACCENT))
    fig.update_xaxes(title_text="Performance rating")
    fig.update_yaxes(title_text=f"Median {other.strip('_')}",
                     tickformat="~s" if other == "_salary" else None)
    r = t["_performance"].corr(t[other])
    name = other.strip("_")
    return _chart(
        f"Does {name} follow performance?", _layout(fig),
        f"Higher-rated staff {'do' if r > 0.2 else 'do not really'} have higher {name} "
        f"(correlation {r:.2f}).",
        f"Median {name} per rating level, n = {len(t):,}, r = {r:.2f}. "
        "Correlation across levels says nothing about cause.")


def _h_hiring_trend(d):
    if "_hire_date" not in d:
        return None
    dt = d["_hire_date"].dropna()
    if len(dt) < 10:
        return None
    g = dt.dt.to_period("Y").value_counts().sort_index()
    if len(g) < 2:
        return None
    fig = go.Figure(go.Bar(x=[str(i) for i in g.index], y=g.values, marker_color=ACCENT))
    fig.update_yaxes(title_text="People hired")
    return _chart(
        "Hiring over the years", _layout(fig),
        f"The biggest intake was {g.idxmax()} with {g.max():,} hires.",
        "Hires per year among people in this dataset. If leavers are excluded, "
        "older years look smaller than they really were.")


def _h_demo(d):
    for key, title in (("_gender", "Gender split"), ("_education", "Education levels"),
                       ("_location", "Where people are based")):
        if key in d:
            g = d[key].value_counts().head(8)
            if len(g) < 2:
                continue
            fig = go.Figure(go.Pie(labels=[_clip(i, 18) for i in g.index], values=g.values,
                                   hole=0.55, textinfo="percent"))
            return _chart(
                title, _layout(fig, legend=True),
                f"'{_clip(g.index[0], 18)}' is the largest group at {g.iloc[0] / g.sum():.0%}.",
                f"Headcount share by {key.strip('_')}, {d[key].nunique()} values.")
    return None


def _kpi_hr_comp(d):
    if "_salary" not in d:
        return []
    s = d["_salary"].dropna()
    return [("Median salary", _fmt(s.median())), ("Average", _fmt(s.mean())),
            ("Lowest", _fmt(s.min())), ("Highest", _fmt(s.max()))]


def _kpi_hr_attr(d):
    if "_attrition" not in d:
        return []
    rate = d["_attrition"].mean()
    left = int(d["_attrition"].sum())
    out = [("Attrition rate", f"{rate:.1%}"), ("People left", f"{left:,}"),
           ("People stayed", f"{len(d) - left:,}")]
    if "_tenure" in d:
        t = d[d["_attrition"] == 1]["_tenure"].dropna()
        if len(t):
            out.append(("Avg tenure at exit", f"{t.mean():.1f} yrs"))
    return out


def _kpi_hr_perf(d):
    out = []
    if "_performance" in d:
        s = d["_performance"].dropna()
        out += [("Avg rating", f"{s.mean():.2f}"), ("Rating levels", f"{s.nunique()}")]
    if "_satisfaction" in d:
        out.append(("Avg satisfaction", f"{d['_satisfaction'].mean():.2f}"))
    return out


def _build_people(df, prof):
    roles = _map_roles(df, _HR_RULES, numeric_roles=_HR_NUM, date_roles=_DATE_ROLES)
    d = _prep_hr(df, roles)
    return [
        _page("workforce", "Workforce Overview", "Who works here?",
              _kpi_hr_overview, [_h_headcount, _h_tenure_dist, _h_age_dist, _h_demo], d,
              missing="No workforce columns found. Check for department, role or tenure columns."),
        _page("pay", "Compensation", "How is pay distributed?",
              _kpi_hr_comp, [_h_salary_dist, _h_salary_by, _h_salary_tenure], d,
              missing="This page needs a salary, income or compensation column."),
        _page("attrition", "Attrition", "Who leaves, and when?",
              _kpi_hr_attr, [_h_attrition_by, _h_attrition_tenure], d,
              missing="This page needs an attrition, resigned or status column."),
        _page("performance", "Performance", "How is performance rated?",
              _kpi_hr_perf, [_h_perf_dist, _h_perf_vs], d,
              missing="This page needs a performance rating or satisfaction column."),
        _page("hiring", "Hiring & Tenure", "How has the workforce been built up?",
              lambda x: [("Hires on record", f"{int(x['_hire_date'].notna().sum()):,}")]
              if "_hire_date" in x else [],
              [_h_hiring_trend, _h_tenure_dist], d,
              missing="This page needs a hire or joining date column."),
    ]


# ═══════════════════════════ TIME SERIES & OPS ════════════════════════════

def _ts_pick(df, prof):
    """The date column with the best coverage, plus the numeric metrics."""
    if not prof["datetime"]:
        return None, []
    best, best_cov = None, -1
    for c in prof["datetime"]:
        cov = _to_datetime(df[c]).notna().mean()
        if cov > best_cov:
            best, best_cov = c, cov
    metrics = [c for c in prof["numeric"] + prof["discrete"]][:6]
    return best, metrics


def _prep_ts(df, prof):
    col, metrics = _ts_pick(df, prof)
    d = pd.DataFrame(index=df.index)
    if col is None:
        return d, []
    d["_date"] = _to_datetime(df[col])
    for m in metrics:
        d[m] = pd.to_numeric(df[m], errors="coerce")
    d = d.dropna(subset=["_date"]).sort_values("_date")
    d.attrs["date_col"] = col
    d.attrs["metrics"] = metrics
    return d, metrics


def _ts_main(d):
    m = d.attrs.get("metrics") or []
    return m[0] if m else None


def _kpi_ts_overview(d):
    if "_date" not in d or d.empty:
        return []
    dt = d["_date"]
    span = (dt.max() - dt.min()).days
    out = [("Records", f"{len(d):,}"), ("From", f"{dt.min():%d %b %Y}"),
           ("To", f"{dt.max():%d %b %Y}"), ("Span", f"{span:,} days")]
    m = _ts_main(d)
    if m:
        out.append((f"Avg {_clip(m, 14)}", _fmt(d[m].mean())))
    return out[:6]


def _t_trend(d):
    m = _ts_main(d)
    if m is None or "_date" not in d or d.empty:
        return None
    freq, label = _period_freq(d["_date"])
    g = d.groupby(d["_date"].dt.to_period(freq))[m].mean().dropna()
    if len(g) < 3:
        return None
    idx = [str(i) for i in g.index]
    fig = go.Figure()
    fig.add_scatter(x=idx, y=g.values, name=m, mode="lines",
                    line=dict(color=ACCENT, width=2))
    if len(g) >= 5:
        fig.add_scatter(x=idx, y=g.rolling(3, min_periods=1).mean().values,
                        name="3-period average", mode="lines",
                        line=dict(color=SAND, width=3, dash="dot"))
    fig.update_yaxes(title_text=m)
    slope = np.polyfit(np.arange(len(g)), g.values, 1)[0]
    rel = slope / abs(g.mean()) if g.mean() else 0
    direction = "rising" if slope > 0 else "falling"
    return _chart(
        f"{m} over time", _layout(fig, legend=len(g) >= 5),
        f"{m} is {direction}, from {_fmt(g.iloc[0])} at the start to {_fmt(g.iloc[-1])} now.",
        f"Mean {m} per {label}, {len(g)} periods. Linear trend {rel:+.1%} of the mean per {label}; "
        "a rolling mean lags real turning points.")


def _t_multi(d):
    metrics = (d.attrs.get("metrics") or [])[1:4]
    if not metrics or "_date" not in d:
        return None
    freq, label = _period_freq(d["_date"])
    fig = go.Figure()
    drawn = 0
    for i, m in enumerate(metrics):
        g = d.groupby(d["_date"].dt.to_period(freq))[m].mean().dropna()
        if len(g) < 3:
            continue
        z = (g - g.mean()) / (g.std() or 1)
        fig.add_scatter(x=[str(x) for x in g.index], y=z.values, name=_clip(m, 16),
                        mode="lines", line=dict(color=COLORWAY[i % len(COLORWAY)], width=2))
        drawn += 1
    if drawn < 2:
        return None
    fig.update_yaxes(title_text="Standardised value")
    return _chart(
        "Comparing the other measures", _layout(fig, legend=True),
        "Each measure is rescaled onto the same axis so you can see which ones move together.",
        f"Z-scored {label} means of {', '.join(metrics[:drawn])}. "
        "Standardising compares shape, not size.")


def _t_change(d):
    m = _ts_main(d)
    if m is None or "_date" not in d:
        return None
    freq, label = _period_freq(d["_date"])
    g = d.groupby(d["_date"].dt.to_period(freq))[m].mean().dropna()
    if len(g) < 4:
        return None
    ch = g.pct_change().mul(100).replace([np.inf, -np.inf], np.nan).dropna()
    if ch.empty:
        return None
    fig = go.Figure(go.Bar(x=[str(i) for i in ch.index], y=ch.values,
                           marker_color=[GOOD if v >= 0 else BAD for v in ch.values]))
    fig.update_yaxes(title_text=f"Change vs previous {label}", ticksuffix="%")
    return _chart(
        "Period-on-period change", _layout(fig),
        f"Biggest rise: {ch.idxmax()} ({ch.max():+.0f}%). Biggest fall: {ch.idxmin()} ({ch.min():+.0f}%). "
        f"{(ch < 0).mean():.0%} of periods went down.",
        f"Percent change per {label}. Mean {ch.mean():+.1f}%, std {ch.std():.1f}%. "
        "Percent changes explode when the base is near zero.")


def _t_season_month(d):
    m = _ts_main(d)
    if m is None or "_date" not in d:
        return None
    dt = d["_date"]
    if (dt.max() - dt.min()).days < 120:
        return None
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    g = d.groupby(dt.dt.month)[m].mean().reindex(range(1, 13))
    if g.notna().sum() < 4:
        return None
    fig = go.Figure(go.Bar(x=months, y=g.values,
                           marker=dict(color=g.fillna(0).values, colorscale=SCALE)))
    fig.update_yaxes(title_text=f"Average {m}")
    best = int(g.idxmax())
    ratio = g.max() / g.min() if g.min() else 0
    return _chart(
        "Which months run highest?", _layout(fig),
        f"{months[best - 1]} is the strongest month on average. "
        f"The best month runs {ratio:.1f}x the weakest.",
        f"Mean {m} by calendar month across all years. "
        "With under two full years this mixes trend with seasonality.")


def _t_season_weekday(d):
    m = _ts_main(d)
    if m is None or "_date" not in d:
        return None
    dt = d["_date"]
    if (dt.max() - dt.min()).days < 14:
        return None
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    g = d.groupby(dt.dt.dayofweek)[m].mean().reindex(range(7))
    if g.notna().sum() < 4:
        return None
    fig = go.Figure(go.Bar(x=days, y=g.values,
                           marker=dict(color=g.fillna(0).values, colorscale=SCALE)))
    fig.update_yaxes(title_text=f"Average {m}")
    best = int(g.idxmax())
    return _chart(
        "Which weekday runs highest?", _layout(fig),
        f"{days[best]} has the highest average {m}.",
        f"Mean {m} by weekday. Confirm the timestamps carry real days before reading into this.")


def _t_volatility(d):
    m = _ts_main(d)
    if m is None or "_date" not in d:
        return None
    freq, label = _period_freq(d["_date"])
    g = d.groupby(d["_date"].dt.to_period(freq))[m].mean().dropna()
    if len(g) < 8:
        return None
    roll = g.rolling(4).std().dropna()
    if roll.empty:
        return None
    fig = go.Figure(go.Scatter(x=[str(i) for i in roll.index], y=roll.values, mode="lines",
                               fill="tozeroy", line=dict(color=OLIVE, width=2)))
    fig.update_yaxes(title_text="Rolling standard deviation")
    return _chart(
        "How steady is it?", _layout(fig),
        f"Swings are largest around {roll.idxmax()}. A rising line means the measure is "
        "becoming less predictable.",
        f"4-period rolling standard deviation of {m}. Mean {_fmt(roll.mean())}. "
        "Rising volatility often precedes a regime change.")


def _t_anomalies(d):
    m = _ts_main(d)
    if m is None or "_date" not in d:
        return None
    freq, label = _period_freq(d["_date"])
    g = d.groupby(d["_date"].dt.to_period(freq))[m].mean().dropna()
    if len(g) < 10:
        return None
    mu, sd = g.mean(), g.std()
    if not sd:
        return None
    out = g[(g - mu).abs() > 2 * sd]
    idx = [str(i) for i in g.index]
    fig = go.Figure()
    fig.add_scatter(x=idx, y=g.values, mode="lines", name=m,
                    line=dict(color=ACCENT, width=2))
    fig.add_hline(y=mu + 2 * sd, line_dash="dash", line_color=MUTED)
    fig.add_hline(y=mu - 2 * sd, line_dash="dash", line_color=MUTED)
    if len(out):
        fig.add_scatter(x=[str(i) for i in out.index], y=out.values, mode="markers",
                        name="Unusual", marker=dict(color=BAD, size=10))
    fig.update_yaxes(title_text=m)
    return _chart(
        "Unusual periods", _layout(fig, legend=True),
        f"{len(out)} period(s) sit outside the normal range (dashed lines)."
        + (f" The most extreme was {(out - mu).abs().idxmax()}." if len(out) else
           " Nothing looks out of the ordinary."),
        f"Points beyond ±2 standard deviations of the series mean ({_fmt(mu)} ± {_fmt(2 * sd)}). "
        "A fixed-mean band flags trend as anomaly; treat it as a prompt to look, not a verdict.")


def _t_gaps(d):
    if "_date" not in d or len(d) < 10:
        return None
    dt = d["_date"].dropna().sort_values()
    diffs = dt.diff().dropna().dt.total_seconds() / 3600
    if diffs.empty or diffs.nunique() < 2:
        return None
    med = diffs.median()
    big = diffs[diffs > med * 3]
    fig = go.Figure(go.Histogram(x=diffs[diffs <= diffs.quantile(0.99)].values,
                                 nbinsx=30, marker_color=OLIVE))
    fig.update_xaxes(title_text="Hours between records")
    fig.update_yaxes(title_text="Count")
    return _chart(
        "Is the data evenly spaced?", _layout(fig),
        f"Records usually arrive every {med:.1f} hours. "
        + (f"{len(big)} unusually long gap(s) suggest missing data." if len(big)
           else "Spacing is consistent, with no large gaps."),
        f"Distribution of intervals between consecutive timestamps, clipped at the 99th percentile. "
        f"Median {med:.2f}h. Irregular spacing breaks most forecasting methods.")


def _t_corr(d):
    metrics = d.attrs.get("metrics") or []
    if len(metrics) < 3:
        return None
    corr = d[metrics].corr()
    if corr.empty:
        return None
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                    color_continuous_scale=[BAD, "#161616", GOOD])
    return _chart(
        "Which measures move together?", fig,
        "Green means two measures rise together; red means one rises as the other falls.",
        f"Pearson correlation across {len(metrics)} measures. In time series, two trending "
        "series correlate even when unrelated — check against differenced data before believing it.")


def _kpi_ts_change(d):
    m = _ts_main(d)
    if m is None or "_date" not in d or d.empty:
        return []
    freq, label = _period_freq(d["_date"])
    g = d.groupby(d["_date"].dt.to_period(freq))[m].mean().dropna()
    if len(g) < 2:
        return []
    total = (g.iloc[-1] / g.iloc[0] - 1) if g.iloc[0] else 0
    return [("Measure", _clip(m, 16)), ("Start", _fmt(g.iloc[0])), ("Latest", _fmt(g.iloc[-1])),
            ("Overall change", f"{total:+.0%}"), ("Periods", f"{len(g):,}")]


def _build_timeseries(df, prof):
    d, metrics = _prep_ts(df, prof)
    if d.empty or not metrics:
        return _build_generic(df, prof)
    return [
        _page("ts_overview", "Series Overview", "What does this series look like?",
              _kpi_ts_overview, [_t_trend, _t_multi, _t_gaps], d,
              missing="No usable date and measure pair was found."),
        _page("ts_trend", "Trend & Change", "Which way is it going?",
              _kpi_ts_change, [_t_trend, _t_change, _t_volatility], d,
              missing="At least four periods of data are needed to show a trend."),
        _page("ts_season", "Seasonality", "Does it follow a repeating pattern?",
              lambda x: [], [_t_season_month, _t_season_weekday], d,
              missing="At least a few weeks of data are needed to look for seasonal patterns."),
        _page("ts_anom", "Anomalies", "When did it behave oddly?",
              lambda x: [], [_t_anomalies, _t_volatility], d,
              missing="At least ten periods are needed before anomalies mean anything."),
        _page("ts_rel", "Relationships", "Which measures are connected?",
              lambda x: [], [_t_corr, _t_multi], d,
              missing="At least three numeric measures are needed."),
    ]


# ═══════════════════════════ FINANCE & TRANSACTIONS ═══════════════════════

_FIN_RULES = {
    "date": ["date", "transaction_date", "posting_date", "value_date", "timestamp", "day"],
    "amount": ["amount", "transaction_amount", "value", "debit", "credit", "txn_amount"],
    "balance": ["balance", "closing_balance", "running_balance", "account_balance"],
    "category": ["category", "transaction_type", "type", "expense_category", "merchant_category"],
    "account": ["account", "account_id", "account_number", "iban", "wallet"],
    "merchant": ["merchant", "payee", "description", "narration", "counterparty"],
    "direction": ["direction", "debit_credit", "dr_cr", "flow", "in_out"],
    "close": ["close", "closing_price", "adj_close", "price"],
    "open": ["open", "opening_price"],
    "high": ["high"],
    "low": ["low"],
    "volume": ["volume", "qty_traded", "shares"],
    "ticker": ["ticker", "symbol", "instrument", "scrip"],
    "currency": ["currency", "ccy"],
}
_FIN_NUM = {"amount", "balance", "close", "open", "high", "low", "volume"}


def _prep_fin(df, roles):
    d = pd.DataFrame(index=df.index)
    if "date" in roles:
        d["_date"] = _to_datetime(df[roles["date"]])
    for r in ("amount", "balance", "close", "open", "high", "low", "volume"):
        if r in roles:
            s = pd.to_numeric(df[roles[r]], errors="coerce")
            if s.notna().mean() > 0.5:
                d["_" + r] = s
    for r in ("category", "account", "merchant", "direction", "ticker", "currency"):
        if r in roles:
            d["_" + r] = _clean_cat(df[roles[r]])
    return d


def _kpi_fin_overview(d):
    out = []
    if "_amount" in d:
        s = d["_amount"].dropna()
        out += [("Transactions", f"{len(s):,}"), ("Total value", _fmt(s.abs().sum())),
                ("Average", _fmt(s.mean()))]
        if (s < 0).any():
            out.append(("Net flow", _fmt(s.sum())))
    if "_balance" in d:
        out.append(("Latest balance", _fmt(d["_balance"].dropna().iloc[-1])
                    if d["_balance"].notna().any() else "n/a"))
    if "_account" in d:
        out.append(("Accounts", f"{d['_account'].nunique():,}"))
    return out[:6]


def _f_flow_time(d):
    if "_amount" not in d or "_date" not in d:
        return None
    t = d.dropna(subset=["_date", "_amount"])
    if t.empty:
        return None
    freq, label = _period_freq(t["_date"])
    g = t.groupby(t["_date"].dt.to_period(freq))["_amount"].sum()
    if len(g) < 2:
        return None
    has_neg = (g < 0).any()
    fig = go.Figure(go.Bar(x=[str(i) for i in g.index], y=g.values,
                           marker_color=[GOOD if v >= 0 else BAD for v in g.values]
                           if has_neg else ACCENT))
    fig.update_yaxes(title_text="Net amount", tickformat="~s")
    return _chart(
        f"Money moved per {label}", _layout(fig),
        f"Highest {label}: {g.idxmax()} at {_fmt(g.max())}."
        + (f" {int((g < 0).sum())} period(s) were net negative." if has_neg else ""),
        f"Sum of amounts per {label} over {len(g)} periods. Mean {_fmt(g.mean())}, "
        f"std {_fmt(g.std())}.")


def _f_balance(d):
    if "_date" not in d:
        return None
    if "_balance" in d:
        t = d.dropna(subset=["_date", "_balance"]).sort_values("_date")
        if len(t) < 3:
            return None
        fig = go.Figure(go.Scatter(x=t["_date"], y=t["_balance"], mode="lines",
                                   line=dict(color=ACCENT, width=2)))
        fig.update_yaxes(title_text="Balance", tickformat="~s")
        return _chart(
            "Balance over time", _layout(fig),
            f"Balance moved from {_fmt(t['_balance'].iloc[0])} to {_fmt(t['_balance'].iloc[-1])}, "
            f"touching a low of {_fmt(t['_balance'].min())}.",
            f"Recorded balance over {len(t):,} points. Minimum {_fmt(t['_balance'].min())}, "
            f"maximum {_fmt(t['_balance'].max())}.")
    if "_amount" in d:
        t = d.dropna(subset=["_date", "_amount"]).sort_values("_date")
        if len(t) < 3:
            return None
        cum = t["_amount"].cumsum()
        fig = go.Figure(go.Scatter(x=t["_date"], y=cum.values, mode="lines",
                                   fill="tozeroy", line=dict(color=GOOD, width=2)))
        fig.update_yaxes(title_text="Cumulative amount", tickformat="~s")
        return _chart(
            "Running total", _layout(fig),
            f"Adding every transaction in order ends at {_fmt(cum.iloc[-1])}.",
            "Cumulative sum of amounts in date order. Only meaningful if signs distinguish "
            "money in from money out.")
    return None


def _f_by_category(d):
    key = "_category" if "_category" in d else ("_merchant" if "_merchant" in d else None)
    if key is None or "_amount" not in d:
        return None
    t = d.dropna(subset=[key, "_amount"])
    g = t.groupby(key)["_amount"].sum().abs().sort_values(ascending=False).head(10)
    if len(g) < 2:
        return None
    name = key.strip("_")
    return _chart(
        f"Where the money goes, by {name}", _hbar(g, "Total amount"),
        f"'{_clip(g.index[0], 20)}' accounts for the most at {_fmt(g.iloc[0])} "
        f"({g.iloc[0] / g.sum():.0%} of the top {len(g)}).",
        f"Absolute total per {name}, top {len(g)} of {t[key].nunique():,}. "
        "Absolute values mix inflows and outflows if signs are used.")


def _f_amount_dist(d):
    if "_amount" not in d:
        return None
    s = d["_amount"].dropna().abs()
    s = s[s > 0]
    if len(s) < 30:
        return None
    cap = s.quantile(0.99)
    fig = go.Figure(go.Histogram(x=s[s <= cap].values, nbinsx=40, marker_color=OLIVE))
    fig.update_xaxes(title_text="Transaction size", tickformat="~s")
    fig.update_yaxes(title_text="Transactions")
    return _chart(
        "How big is a typical transaction?", _layout(fig),
        f"Most transactions are around {_fmt(s.median())}, though a few reach {_fmt(s.max())}.",
        f"Transaction size clipped at the 99th percentile. Median {_fmt(s.median())}, "
        f"mean {_fmt(s.mean())}, skew {s.skew():.1f}.")


def _f_price(d):
    if "_close" not in d or "_date" not in d:
        return None
    t = d.dropna(subset=["_date", "_close"]).sort_values("_date")
    if len(t) < 5:
        return None
    fig = go.Figure()
    fig.add_scatter(x=t["_date"], y=t["_close"], mode="lines", name="Close",
                    line=dict(color=ACCENT, width=2))
    if len(t) >= 20:
        fig.add_scatter(x=t["_date"], y=t["_close"].rolling(20).mean(), mode="lines",
                        name="20-period average", line=dict(color=SAND, width=2, dash="dot"))
    fig.update_yaxes(title_text="Price")
    ret = (t["_close"].iloc[-1] / t["_close"].iloc[0] - 1) if t["_close"].iloc[0] else 0
    return _chart(
        "Price over time", _layout(fig, legend=len(t) >= 20),
        f"Price went from {_fmt(t['_close'].iloc[0])} to {_fmt(t['_close'].iloc[-1])}, "
        f"a {ret:+.1%} move overall.",
        f"Closing price across {len(t):,} points with a 20-period moving average. "
        "Past movement is not a forecast.")


def _f_returns(d):
    if "_close" not in d or "_date" not in d:
        return None
    t = d.dropna(subset=["_date", "_close"]).sort_values("_date")
    if len(t) < 20:
        return None
    r = t["_close"].pct_change().dropna().mul(100)
    if r.empty:
        return None
    fig = go.Figure(go.Histogram(x=r.values, nbinsx=50, marker_color=OLIVE))
    fig.update_xaxes(title_text="Period return", ticksuffix="%")
    fig.update_yaxes(title_text="Periods")
    return _chart(
        "How much does it move each period?", _layout(fig),
        f"A typical period moves {r.abs().median():.2f}%. The worst was {r.min():.1f}%, "
        f"the best {r.max():+.1f}%.",
        f"Period return distribution, n = {len(r):,}. Mean {r.mean():+.3f}%, "
        f"std {r.std():.2f}%, kurtosis {r.kurtosis():.1f} — fat tails mean extremes are "
        "more common than a normal curve implies.")


def _f_volume(d):
    if "_volume" not in d or "_date" not in d:
        return None
    t = d.dropna(subset=["_date", "_volume"]).sort_values("_date")
    if len(t) < 5:
        return None
    fig = go.Figure(go.Bar(x=t["_date"], y=t["_volume"], marker_color=MUTED))
    fig.update_yaxes(title_text="Volume", tickformat="~s")
    return _chart(
        "Trading volume", _layout(fig),
        f"Heaviest activity was {t.loc[t['_volume'].idxmax(), '_date']:%d %b %Y} "
        f"at {_fmt(t['_volume'].max())}.",
        f"Volume per period, mean {_fmt(t['_volume'].mean())}. "
        "Volume spikes usually accompany news or rebalancing.")


def _f_by_account(d):
    if "_account" not in d or "_amount" not in d:
        return None
    t = d.dropna(subset=["_account", "_amount"])
    g = t.groupby("_account")["_amount"].sum().abs().sort_values(ascending=False).head(10)
    if len(g) < 2:
        return None
    return _chart(
        "Activity by account", _hbar(g, "Total amount"),
        f"'{_clip(g.index[0], 20)}' is the busiest account at {_fmt(g.iloc[0])}.",
        f"Absolute total per account, top {len(g)} of {t['_account'].nunique():,}.")


def _f_outliers(d):
    if "_amount" not in d or "_date" not in d:
        return None
    t = d.dropna(subset=["_date", "_amount"])
    if len(t) < 30:
        return None
    s = t["_amount"].abs()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return None
    thr = q3 + 3 * iqr
    out = t[s > thr]
    fig = go.Figure()
    fig.add_scatter(x=t["_date"], y=t["_amount"], mode="markers", name="Transactions",
                    marker=dict(color=MUTED, size=4, opacity=0.4))
    if len(out):
        fig.add_scatter(x=out["_date"], y=out["_amount"], mode="markers", name="Unusually large",
                        marker=dict(color=BAD, size=9))
    fig.update_yaxes(title_text="Amount", tickformat="~s")
    return _chart(
        "Unusually large transactions", _layout(fig, legend=True),
        f"{len(out)} transaction(s) are far larger than the rest (above {_fmt(thr)})."
        if len(out) else "No transaction stands out as unusually large.",
        f"Points above Q3 + 3×IQR ({_fmt(thr)}). A statistical flag, not evidence of anything wrong.")


def _kpi_fin_market(d):
    if "_close" not in d:
        return []
    s = d["_close"].dropna()
    if s.empty:
        return []
    r = s.pct_change().dropna()
    out = [("Latest", _fmt(s.iloc[-1])), ("High", _fmt(s.max())), ("Low", _fmt(s.min()))]
    if len(r):
        out += [("Total change", f"{(s.iloc[-1] / s.iloc[0] - 1):+.1%}" if s.iloc[0] else "n/a"),
                ("Volatility", f"{r.std() * 100:.2f}%")]
    return out


def _build_finance(df, prof):
    roles = _map_roles(df, _FIN_RULES, numeric_roles=_FIN_NUM, date_roles={"date"})
    d = _prep_fin(df, roles)
    if "_close" in d and "_amount" not in d:
        # market / price data asks different questions from bank transactions
        return [
            _page("mkt_overview", "Market Overview", "How has the price behaved?",
                  _kpi_fin_market, [_f_price, _f_volume, _f_returns], d,
                  missing="This page needs a price or close column with dates."),
            _page("mkt_returns", "Returns & Risk", "How much does it swing?",
                  _kpi_fin_market, [_f_returns, _f_price], d,
                  missing="At least 20 price points are needed to measure returns."),
        ] + _build_timeseries(df, prof)[2:4]
    return [
        _page("fin_overview", "Overview", "What is moving, and how much?",
              _kpi_fin_overview, [_f_flow_time, _f_balance, _f_amount_dist, _f_by_category], d,
              missing="No amount, balance or price column found."),
        _page("fin_flow", "Flow Over Time", "How does it change over time?",
              _kpi_fin_overview, [_f_balance, _f_flow_time], d,
              missing="This page needs a date column alongside amounts or balances."),
        _page("fin_cat", "Categories", "Where does the money go?",
              lambda x: [("Categories", f"{x['_category'].nunique():,}")] if "_category" in x else [],
              [_f_by_category, _f_by_account, _f_amount_dist], d,
              missing="This page needs a category, type or merchant column."),
        _page("fin_quality", "Unusual Activity", "What stands out?",
              lambda x: [], [_f_outliers, _f_amount_dist], d,
              missing="At least 30 transactions with dates are needed."),
    ]


# ═══════════════════════════════ ROUTING ══════════════════════════════════

BUILDERS = {
    "web": _build_web,
    "people": _build_people,
    "finance": _build_finance,
    "timeseries": _build_timeseries,
    "generic": _build_generic,
}


def build_dashboard(df, domain=None, mapping=None):
    """Build dashboard pages for the chosen domain. If domain is None the
    best detected one is used. Sales keeps its purpose-built module
    (tools/business.py); every other domain has its own builder above, and
    'generic' works on any table, so no dataset ever ends on an empty page."""
    prof = profile_columns(df)

    if domain is None:
        ranked = detect_domain(df)
        domain = ranked[0][0] if ranked else "generic"

    if domain == "sales":
        from tools.business import build_pages, detect_columns
        return build_pages(df, mapping if mapping is not None else detect_columns(df))

    pages = BUILDERS.get(domain, _build_generic)(df, prof)
    # A page whose required column doesn't exist adds nothing but a "needs X"
    # message, so it is left out. If a domain produced nothing at all, the
    # generic dashboard takes over rather than showing an empty screen.
    pages = [p for p in pages if p["charts"]]
    return pages or [p for p in _build_generic(df, prof) if p["charts"]]