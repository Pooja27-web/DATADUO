import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import cross_val_predict

from tools.modeling import detect_task
from tools.palette import ACCENT, BAD, COLORWAY, GOOD, MUTED, SCALE

S1 = "1. Can we trust the data?"
S2 = "2. What are we looking at?"
S3 = "3. What matters most?"
S4 = "4. How well does the model work?"


def _chart(key, section, title, fig, note, tech=None, level="core"):
    """note = plain language (1 to 2 sentences), tech = technical wording, level = core or adv."""
    return {"key": key, "section": section, "title": title, "fig": fig,
            "note": note, "tech": tech or note, "level": level}


def _source_column(feature, columns):
    """Map a model feature like 'Sex_male' back to its original column 'Sex'."""
    if feature in columns:
        return feature
    matches = [c for c in columns if feature.startswith(f"{c}_")]
    return max(matches, key=len) if matches else None


def top_driver_columns(cleaned, target, importance, n=3):
    cols = list(cleaned.columns)
    seen = []
    for feat in importance.index:
        src = _source_column(feat, cols)
        if src and src != target and src not in seen:
            seen.append(src)
        if len(seen) == n:
            break
    return seen


def _pretty(feature, cols):
    src = _source_column(feature, cols)
    if src and src != feature:
        return f"{src}: {feature[len(src) + 1:]}"
    return feature


# ---------- Section 1: data trust ----------
def _missing_chart(raw):
    miss = (raw.isna().mean() * 100).round(1)
    miss = miss[miss > 0].sort_values()
    if miss.empty:
        return None
    fig = px.bar(
        x=miss.values, y=miss.index, orientation="h",
        color=miss.values, color_continuous_scale=[GOOD, ACCENT, BAD],
        labels={"x": "% of rows that are empty", "y": "Column"},
    )
    fig.update_coloraxes(showscale=False)
    overall = raw.isna().mean().mean() * 100
    return _chart(
        "missing", S1, "Gaps in the data", fig,
        "Each bar is a column with empty cells. Long bars mean less reliable information.",
        f"Share of empty cells per column ({len(miss)} columns affected, {overall:.1f}% of all cells). "
        "Columns above 60% are dropped by the cleaner.",
    )


# ---------- Section 2: the target ----------
def _target_chart(cleaned, target, task):
    s = cleaned[target]
    if task == "classification":
        counts = s.astype(str).value_counts().head(12)
        pct = (counts / counts.sum() * 100).round(1)
        fig = px.bar(
            x=counts.index, y=counts.values, color=counts.index,
            color_discrete_sequence=COLORWAY, text=[f"{p}%" for p in pct],
            labels={"x": target, "y": "Number of rows"},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        note = "How many rows fall into each group of what we want to predict."
        if pct.min() < 20:
            note += " The groups are uneven, so a high accuracy alone can mislead."
        tech = (f"Class counts for '{target}' (top 12). Smallest class is {pct.min():.1f}% of rows; "
                "below 20%, prefer balanced accuracy or AUC over plain accuracy.")
    else:
        fig = px.histogram(cleaned, x=target, nbins=40)
        note = "The most common values of what we want to predict. Tall bars mean many rows have values in that range."
        tech = (f"Histogram of '{target}': mean {s.mean():.2f}, median {s.median():.2f}, "
                f"skew {s.skew():.2f}. Strong skew can justify a log transform.")
    return _chart("target", S2, f"Overview of {target}", fig, note, tech)


# ---------- Section 3: what matters ----------
def _importance_chart(model_out, target, cols):
    imp = model_out["importance"].head(10).iloc[::-1]
    labels = [_pretty(f, cols) for f in imp.index]
    fig = px.bar(
        x=imp.values, y=labels, orientation="h",
        color=imp.values, color_continuous_scale=SCALE,
        labels={"x": "How much it influences the prediction", "y": "Factor"},
    )
    fig.update_coloraxes(showscale=False)
    return _chart(
        "drivers", S3, "Top factors", fig,
        "The longer the bar, the more the model relies on that factor. Influence shows what goes together with the outcome, not what causes it.",
        f"{model_out['importance_method']} on {model_out['modeled_rows']:,} rows, top 10 features. "
        "One-hot columns are listed as column: value.",
    )


def _driver_chart(df, target, task, col):
    d = df[[col, target]].dropna().copy()
    x, t = d[col], d[target]
    continuous = pd.api.types.is_numeric_dtype(x) and x.nunique() > 10

    if not continuous:
        top = x.value_counts().head(10).index
        d = d[x.isin(top)].copy()
        d[col] = d[col].astype(str)
        if task == "classification" and pd.api.types.is_numeric_dtype(t) and t.nunique() == 2:
            positive = t.max()
            g = (d[target] == positive).groupby(d[col]).mean().mul(100).round(1)
            fig = px.bar(
                x=g.index, y=g.values, color=g.index, color_discrete_sequence=COLORWAY,
                text=g.values, labels={"x": col, "y": f"% with {target} = {positive}"},
            )
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(showlegend=False)
            note = (f"Each bar shows how often {target} = {positive} inside each '{col}' group. "
                    "Big differences between bars mean this factor matters.")
            tech = (f"Share of rows with {target} = {positive} per '{col}' group (top 10 groups). "
                    "Descriptive only; check group sizes before comparing.")
        elif task == "classification":
            ct = pd.crosstab(d[col], d[target].astype(str), normalize="index").mul(100).round(1)
            fig = px.bar(ct, barmode="stack", color_discrete_sequence=COLORWAY,
                         labels={"value": "% of rows", "variable": target})
            note = (f"Each bar is a '{col}' group split by '{target}'. "
                    "If the colours change a lot from bar to bar, this factor matters.")
            tech = f"Row-normalised class mix of '{target}' within each '{col}' group."
        else:
            g = d.groupby(col)[target].mean().round(2).sort_values()
            fig = px.bar(x=g.index, y=g.values, color=g.index, color_discrete_sequence=COLORWAY,
                         labels={"x": col, "y": f"Average {target}"})
            fig.update_layout(showlegend=False)
            note = f"The average {target} for each '{col}' group. Large gaps between bars mean this factor matters."
            tech = f"Mean {target} by '{col}' group. Gaps can reflect sample size as well as real differences."
    elif task == "classification":
        d["_t"] = t.astype(str)
        fig = px.box(d, x="_t", y=col, color="_t", color_discrete_sequence=COLORWAY, labels={"_t": target})
        fig.update_layout(showlegend=False)
        note = (f"Each box shows the typical range of '{col}' for each '{target}' group. "
                "Boxes at different heights mean this factor matters.")
        tech = (f"Distribution of '{col}' (median, quartiles, whiskers) per '{target}' class. "
                "Overlapping boxes imply weak separation.")
    else:
        bins = pd.qcut(x, q=10, duplicates="drop")
        g = d.groupby(bins, observed=True)[target].mean()
        mid = [float(iv.mid) for iv in g.index]
        fig = px.line(x=mid, y=g.values, markers=True, labels={"x": col, "y": f"Average {target}"})
        note = f"As '{col}' goes up, does {target} go up, down or stay flat? The line shows the average trend."
        tech = f"Mean {target} across deciles of '{col}'. A monotonic line suggests a stable relationship."
    return _chart(f"driver_{col}", S3, f"{col} and {target}", fig, note, tech)


# ---------- Section 4: model quality ----------
def _metric_name(model_out):
    if model_out.get("metric_label"):
        return model_out["metric_label"]
    return "Accuracy" if model_out["task"] == "classification" else "R² (share of variation explained)"


def _compare_chart(model_out):
    r = model_out["results"].copy()
    base_label, model_label = "Simple guess (baseline)", "Machine-learning model"
    r["kind"] = np.where(r["model"].str.startswith("Baseline"), base_label, model_label)
    metric = _metric_name(model_out)
    fig = px.bar(
        r, x="model", y="mean", color="kind", error_y="std",
        color_discrete_map={base_label: BAD, model_label: ACCENT},
        labels={"mean": f"{metric} (higher is better)", "model": "", "kind": ""},
    )
    base = float(r[r["kind"] == base_label]["mean"].iloc[0])
    best = float(r[r["model"] == model_out["best_model"]]["mean"].iloc[0])
    split = model_out.get("split", "shuffled")
    return _chart(
        "compare", S4, "Models vs a simple guess", fig,
        "The red bar is what you would get without any intelligence. A useful model must clearly beat it.",
        f"Mean {metric} over a {split} cross-validation, with 1 std error bars. "
        f"Best model {model_out['best_model']} {best:.3f} vs baseline {base:.3f}.",
    )


def _fit_chart(model_out):
    """Training score next to test score for every model. A large gap is
    overfitting; both low is underfitting. This is the over/underfit check
    made visible."""
    r = model_out["results"]
    if "train" not in r:
        return None
    r = r[~r["model"].str.startswith("Baseline")]
    if r.empty:
        return None
    fig = go.Figure()
    fig.add_bar(x=r["model"], y=r["train"], name="On data it learned from", marker_color=MUTED)
    fig.add_bar(x=r["model"], y=r["mean"], name="On new data", marker_color=ACCENT)
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.12, x=0))
    fig.update_yaxes(title_text=_metric_name(model_out))
    fc = model_out.get("fit_check") or {}
    return _chart(
        "fit", S4, "Is the model memorising or learning?", fig,
        fc.get("plain", "Grey is the score on training data, amber on new data. "
                        "A big gap means the model is memorising."),
        fc.get("tech", "Mean train vs test score per model across folds.")
        + " Gap above 0.15 is treated as overfitting; lift under 0.02 over the baseline as underfitting.",
    )


def _oof(model_out):
    """Out-of-fold predictions. Precomputed by the modeling step, which also
    covers time-ordered splits where cross_val_predict cannot be used."""
    if model_out.get("oof") is not None:
        o = model_out["oof"]
        return o["y"], o["pred"], o.get("proba")
    X, y = model_out["X"], model_out["y"]
    model = model_out["model_objs"][model_out["best_model"]]
    cv = model_out["cv"]
    if model_out["task"] == "classification":
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
        return y, proba.argmax(axis=1), proba
    return y, cross_val_predict(model, X, y, cv=cv), None


def _prediction_charts(model_out):
    out = []
    y, pred, proba = _oof(model_out)
    y = pd.Series(np.asarray(y))
    pred = np.asarray(pred)

    if model_out["task"] == "classification":
        classes = model_out.get("classes") or [str(c) for c in sorted(set(y))]
        if len(classes) <= 12:
            cm = confusion_matrix(y, pred, labels=list(range(len(classes))), normalize="true") * 100
            fig = px.imshow(
                cm.round(0), x=classes, y=classes, text_auto=".0f",
                color_continuous_scale=SCALE,
                labels=dict(x="What the model predicted", y="What actually happened", color="%"),
            )
            out.append(_chart(
                "confusion", S4, "Right and wrong predictions", fig,
                "Each row is what really happened and each column is what the model predicted. Bright cells on the diagonal are correct.",
                "Row-normalised confusion matrix from out-of-fold predictions; the diagonal is recall per class.",
                level="adv",
            ))
        if len(classes) == 2 and proba is not None and y.nunique() == 2:
            fpr, tpr, _ = roc_curve(y, proba[:, 1])
            score = auc(fpr, tpr)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Model (AUC {score:.2f})",
                                     line=dict(width=4, color=GOOD)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guessing",
                                     line=dict(dash="dash", color=MUTED)))
            fig.update_layout(xaxis_title="False alarms", yaxis_title="Correct detections")
            out.append(_chart(
                "roc", S4, "Separation quality", fig,
                f"The further the green curve bends towards the top left, the better. AUC {score:.2f} is the score (1.0 is perfect, 0.5 is guessing).",
                f"ROC curve from out-of-fold probabilities, AUC {score:.3f}. Unaffected by class imbalance.",
                level="adv",
            ))
    else:
        idx = np.random.RandomState(0).choice(len(y), min(len(y), 2000), replace=False)
        fig = px.scatter(x=y.values[idx], y=pred[idx], opacity=0.6,
                         labels={"x": "Actual value", "y": "Predicted value"})
        lo, hi = float(y.min()), float(y.max())
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                                 line=dict(dash="dash", color=ACCENT)))
        out.append(_chart(
            "actual_vs_pred", S4, "Predicted vs actual", fig,
            "Each dot is one row. The closer the dots are to the dashed line, the better the prediction.",
            "Out-of-fold predictions on up to 2,000 rows. Systematic curvature suggests missing non-linear structure.",
        ))
        if model_out.get("split") == "time-ordered":
            order = np.arange(len(y))
            fig = go.Figure()
            fig.add_scatter(x=order, y=y.values, mode="lines", name="Actual",
                            line=dict(color=MUTED, width=2))
            fig.add_scatter(x=order, y=pred, mode="lines", name="Predicted",
                            line=dict(color=ACCENT, width=2))
            fig.update_layout(legend=dict(orientation="h", y=1.12, x=0))
            fig.update_xaxes(title_text="Test periods, in time order")
            out.append(_chart(
                "forecast", S4, "Forecast against what happened", fig,
                "The amber line is what the model predicted before seeing each period; grey is what really happened.",
                "One-step-ahead predictions from each time-ordered test window, concatenated in order.",
            ))
    return out


def build_story_charts(raw, cleaned, target, model_out):
    charts = []
    task = model_out["task"] if model_out else detect_task(cleaned[target])

    try:
        c = _missing_chart(raw)
        if c:
            charts.append(c)
    except Exception:
        pass
    try:
        charts.append(_target_chart(cleaned, target, task))
    except Exception:
        pass

    if model_out:
        try:
            charts.append(_importance_chart(model_out, target, list(cleaned.columns)))
        except Exception:
            pass
        for i, src in enumerate(top_driver_columns(cleaned, target, model_out["importance"])):
            try:
                c = _driver_chart(cleaned, target, task, src)
                if i >= 2:
                    c["level"] = "adv"
                charts.append(c)
            except Exception:
                pass
        for build in (_compare_chart, _fit_chart):
            try:
                c = build(model_out)
                if c:
                    charts.append(c)
            except Exception:
                pass
        try:
            charts += _prediction_charts(model_out)
        except Exception:
            pass
    return charts
