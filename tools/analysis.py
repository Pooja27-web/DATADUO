"""tools/analysis.py — data-aware chart selection for DATADUO.

Instead of applying one fixed recipe to every dataset (bar chart for every
low-cardinality column, histogram for every numeric column, always one
heatmap), this module classifies each column first and picks the chart type
that actually fits it, then ranks candidates by effect size and caps the
total so results stay readable on both a 4-column and a 40-column dataset.
"""

import numpy as np
import pandas as pd
import plotly.express as px

MAX_CAT_CHARTS = 5     # categorical-vs-target charts to keep
MAX_NUM_CHARTS = 3     # numeric-vs-target charts to keep
CAT_LOW = 6             # <= this many categories: show as-is
CAT_MID = 20            # <= this many: still show as-is, horizontal bar
CAT_HIGH_MAX = 50       # above CAT_MID and up to this: bucket to top-8 + Other
TOP_N_BUCKET = 8
MIN_GROUP_DEFAULT = 10


# ───────────────────────── column classification ──────────────────────────

def _classify(df: pd.DataFrame, target: str) -> dict:
    """Map each usable column to a role: cat_low, cat_mid, cat_high,
    numeric, numeric_discrete, or datetime. Columns that are ID-like
    (all unique) or constant (one value) are left out entirely."""
    roles = {}
    n_rows = len(df)
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        nunique = s.nunique(dropna=True)
        if nunique <= 1 or nunique >= n_rows:
            continue  # constant or ID-like: not informative

        if pd.api.types.is_datetime64_any_dtype(s):
            roles[c] = "datetime"
            continue

        if pd.api.types.is_numeric_dtype(s):
            roles[c] = "numeric" if nunique > 10 else "numeric_discrete"
            continue

        # try to parse as date before treating as plain categorical
        if s.dtype == object and nunique > CAT_HIGH_MAX:
            try:
                parsed = pd.to_datetime(s, errors="coerce", format="mixed")
                if parsed.notna().mean() > 0.9:
                    roles[c] = "datetime"
                    continue
            except Exception:
                pass

        if nunique <= CAT_LOW:
            roles[c] = "cat_low"
        elif nunique <= CAT_MID:
            roles[c] = "cat_mid"
        elif nunique <= CAT_HIGH_MAX:
            roles[c] = "cat_high"
        # above CAT_HIGH_MAX: too sparse to summarize meaningfully, skip

    return roles


def _bucket_top(s: pd.Series, n: int = TOP_N_BUCKET) -> pd.Series:
    """Collapse a high-cardinality column into its top-n values + 'Other'."""
    top = s.value_counts().nlargest(n).index
    return s.where(s.isin(top), "Other")


def _skew(s: pd.Series) -> float:
    try:
        v = float(s.skew())
        return 0.0 if np.isnan(v) else v
    except Exception:
        return 0.0


# ──────────────────────── classification target (binary) ──────────────────

def auto_analyze(df: pd.DataFrame, target: str, min_group_size: int = MIN_GROUP_DEFAULT):
    """Find patterns related to a binary target column.
    Returns (insights, figures): text findings ranked by effect size, and a
    dict of Plotly charts chosen per-column instead of one fixed template.
    """
    insights, figures = [], {}
    cat_candidates, num_candidates = [], []

    roles = _classify(df, target)
    overall = df[target].mean()
    insights.append(f"Overall '{target}' rate: {overall:.1%}")

    for col, role in roles.items():
        if role in ("cat_low", "cat_mid", "cat_high", "numeric_discrete"):
            s = df[col]
            bucketed = role == "cat_high"
            if bucketed:
                s = _bucket_top(s)
            g = pd.DataFrame({col: s, target: df[target]}).groupby(col)[target].agg(["mean", "count"])
            g = g[g["count"] >= min_group_size]
            if len(g) < 2:
                continue
            g = g.sort_values("mean")
            worst, best = g.index[0], g.index[-1]
            spread = g["mean"].iloc[-1] - g["mean"].iloc[0]
            label = f"top {TOP_N_BUCKET} + Other" if bucketed else col
            text = (f"{col}: '{target}' rate is highest for '{best}' ({g['mean'].iloc[-1]:.0%}) "
                    f"and lowest for '{worst}' ({g['mean'].iloc[0]:.0%})" + (f" [{label}]" if bucketed else ""))
            orientation = "h" if len(g) > 6 else "v"
            if orientation == "h":
                fig = px.bar(g.reset_index(), y=col, x="mean", orientation="h",
                             title=f"'{target}' rate by {col}" + (" (top 8 + Other)" if bucketed else ""),
                             labels={"mean": f"{target} rate"})
            else:
                fig = px.bar(g.reset_index(), x=col, y="mean",
                             title=f"'{target}' rate by {col}",
                             labels={"mean": f"{target} rate"})
                fig.update_xaxes(type="category")
            key = f"rate_top_{col}" if bucketed else f"rate_by_{col}"
            cat_candidates.append((spread, key, fig, text))

        elif role == "datetime":
            tmp = df[[col, target]].dropna()
            tmp["_period"] = pd.to_datetime(tmp[col], errors="coerce").dt.to_period("M").astype(str)
            trend = tmp.groupby("_period")[target].mean().reset_index()
            if len(trend) >= 2:
                fig = px.line(trend, x="_period", y=target, markers=True,
                              title=f"'{target}' rate over time (by {col})",
                              labels={target: f"{target} rate", "_period": col})
                variance = float(trend[target].var() or 0)
                cat_candidates.append((variance * 5, f"trend_{col}", fig,
                                       f"{col}: '{target}' rate changes over time — see trend chart"))

        elif role == "numeric":
            s = df[col]
            try:
                r = float(df[[col, target]].corr().iloc[0, 1])
            except Exception:
                r = 0.0
            score = abs(r) if not np.isnan(r) else 0.0
            skewed = abs(_skew(s)) > 1.0
            if skewed:
                fig = px.box(df, x=df[target].astype(str), y=col,
                            title=f"{col} by {target} (box plot — skewed distribution)")
                key = f"box_{col}"
            else:
                fig = px.histogram(df, x=col, color=df[target].astype(str),
                                   barmode="overlay", opacity=0.6,
                                   title=f"Distribution of {col} by {target}")
                key = f"dist_{col}"
            direction = "positively" if r > 0 else "negatively"
            text = f"{col} is {direction} correlated with {target} (r = {r:.2f})" if r else f"{col}: distribution shown by {target}"
            num_candidates.append((score, key, fig, text))

    cat_candidates.sort(key=lambda x: x[0], reverse=True)
    num_candidates.sort(key=lambda x: x[0], reverse=True)

    for _, key, fig, text in cat_candidates[:MAX_CAT_CHARTS]:
        figures[key] = fig
        insights.append(text)
    for _, key, fig, text in num_candidates[:MAX_NUM_CHARTS]:
        figures[key] = fig
        insights.append(text)

    numeric_cols = [c for c, r in roles.items() if r in ("numeric", "numeric_discrete")]
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols + [target]].corr()
        figures["correlation_heatmap"] = px.imshow(
            corr_df, text_auto=".2f",
            color_continuous_scale=["#E8604C", "#161616", "#7FD1A0"], zmin=-1, zmax=1,
            title="Correlation heatmap",
        )

    return insights, figures


# ────────────────────── regression target (continuous) ─────────────────────

def regression_analyze(df: pd.DataFrame, target: str, min_group_size: int = MIN_GROUP_DEFAULT):
    """Insights and charts for a continuous numeric target (like Age or Fare)."""
    insights, figures = [], {}
    cat_candidates, num_candidates = [], []

    roles = _classify(df, target)
    insights.append(
        f"Average '{target}': {df[target].mean():.2f} "
        f"(median {df[target].median():.2f}, std {df[target].std():.2f})"
    )
    figures["target_distribution"] = px.histogram(df, x=target, title=f"Distribution of {target}")

    for col, role in roles.items():
        if role in ("cat_low", "cat_mid", "cat_high", "numeric_discrete"):
            s = df[col]
            bucketed = role == "cat_high"
            if bucketed:
                s = _bucket_top(s)
            tmp = pd.DataFrame({col: s, target: df[target]})
            g = tmp.groupby(col)[target].agg(["mean", "count"])
            g = g[g["count"] >= min_group_size]
            if len(g) < 2:
                continue
            spread = g["mean"].max() - g["mean"].min()
            text = (f"{col}: average '{target}' is highest for '{g['mean'].idxmax()}' ({g['mean'].max():.1f}) "
                    f"and lowest for '{g['mean'].idxmin()}' ({g['mean'].min():.1f})" +
                    (" [top 8 + Other]" if bucketed else ""))
            large = (tmp.groupby(col)[target].count() >= 30).mean() > 0.5
            if large:
                fig = px.violin(tmp, x=col, y=target, box=True,
                                title=f"{target} by {col}" + (" (top 8 + Other)" if bucketed else ""))
                key = f"violin_{col}"
            else:
                fig = px.box(tmp, x=col, y=target,
                            title=f"{target} by {col}" + (" (top 8 + Other)" if bucketed else ""))
                key = f"box_{col}"
            fig.update_xaxes(type="category")
            cat_candidates.append((spread, key, fig, text))

        elif role == "datetime":
            tmp = df[[col, target]].dropna()
            tmp["_period"] = pd.to_datetime(tmp[col], errors="coerce").dt.to_period("M").astype(str)
            trend = tmp.groupby("_period")[target].mean().reset_index()
            if len(trend) >= 2:
                fig = px.line(trend, x="_period", y=target, markers=True,
                              title=f"Average {target} over time (by {col})")
                variance = float(trend[target].var() or 0)
                cat_candidates.append((variance, f"trend_{col}", fig,
                                       f"{col}: average {target} changes over time — see trend chart"))

        elif role == "numeric":
            try:
                r = float(df[[col, target]].corr().iloc[0, 1])
            except Exception:
                r = 0.0
            score = abs(r) if not np.isnan(r) else 0.0
            fig = px.scatter(df, x=col, y=target, opacity=0.5, title=f"{target} vs {col}")
            direction = "positively" if r > 0 else "negatively"
            text = f"{col} is {direction} correlated with {target} (r = {r:.2f})"
            num_candidates.append((score, f"scatter_{col}", fig, text))

    cat_candidates.sort(key=lambda x: x[0], reverse=True)
    num_candidates.sort(key=lambda x: x[0], reverse=True)

    for _, key, fig, text in cat_candidates[:MAX_CAT_CHARTS]:
        figures[key] = fig
        insights.append(text)
    for _, key, fig, text in num_candidates[:MAX_NUM_CHARTS]:
        figures[key] = fig
        insights.append(text)

    all_numeric = [c for c, r in roles.items() if r in ("numeric", "numeric_discrete")]
    if len(all_numeric) >= 1:
        corr_df = df[all_numeric + [target]].corr()
        figures["correlation_heatmap"] = px.imshow(
            corr_df, text_auto=".2f",
            color_continuous_scale=["#E8604C", "#161616", "#7FD1A0"], zmin=-1, zmax=1,
            title="Correlation heatmap",
        )

    return insights, figures