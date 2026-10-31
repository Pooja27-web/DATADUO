"""tools/modeling.py — data-aware model training for DATADUO.

The old version trained every dataset the same way: shuffle, 5 folds,
accuracy or R². That gives misleading numbers in four common situations,
and this module handles each one explicitly:

1. Time-ordered data is never shuffled. The model is trained on earlier rows
   and tested on later ones, the way it would be used for real.
2. Imbalanced classes (few people leave, few visits convert) are reweighted
   and scored with balanced accuracy, so "always predict the majority" no
   longer looks like a good model.
3. Overfitting is measured, not assumed: training and test scores are both
   recorded and the gap is reported in plain words.
4. Columns that are near-copies of the target are removed before training,
   because a 99% score built on a leak is worthless.

Model complexity also scales with the amount of data, and a simpler model
is preferred when it scores about as well as a complex one.

The returned dict keeps every key the rest of the app already uses, and adds
"metric_label", "plan", "fit_check", "split" and "oof".
"""

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

MAX_ROWS = 30_000
MAX_LEVELS = 15          # categories kept per text column; the rest become "Other"
IMBALANCE = 0.20         # minority class below this share counts as imbalanced
LEAK_CORR = 0.97         # a feature this correlated with the target is a leak
SIMPLER_MARGIN = 0.01    # prefer the simpler model if it is within this of the best
_SIMPLICITY = {"Logistic Regression": 0, "Ridge Regression": 0, "Random Forest": 1, "XGBoost": 2}


# ─────────────────────────────── task & target ────────────────────────────

def detect_task(y: pd.Series) -> str:
    """Classification for categories / few unique values, else regression."""
    if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 10:
        return "classification"
    return "regression"


def _is_datetime_like(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype != object:
        return False
    sample = s.dropna().head(200)
    if sample.empty:
        return False
    try:
        return pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean() > 0.8
    except Exception:
        return False


def _to_dt(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    try:
        return pd.to_datetime(s, errors="coerce", format="mixed")
    except Exception:
        return pd.to_datetime(s, errors="coerce")


def _find_time_column(df, target):
    """The date column to order rows by, or None. It must cover most rows."""
    best, best_cov = None, 0.0
    for c in df.columns:
        if c == target or not _is_datetime_like(df[c]):
            continue
        cov = _to_dt(df[c]).notna().mean()
        if cov > best_cov:
            best, best_cov = c, cov
    return best if best_cov >= 0.9 else None


def _should_split_by_time(df, target, time_col):
    """Train-on-past / test-on-future is required when rows are a sequence
    over time (sensor feeds, prices, daily metrics). For a table of
    independent records that merely carry a date (HR files, survey rows),
    a shuffled split is still correct."""
    if time_col is None:
        return False
    try:
        from tools.domains import detect_domain
        domain = detect_domain(df)[0][0]
    except Exception:
        domain = None
    if domain in ("timeseries", "finance"):
        return True
    dt = _to_dt(df[time_col]).dropna()
    # mostly one row per timestamp, and few descriptive columns: a series
    return dt.nunique() / max(len(dt), 1) > 0.8 and df.shape[1] <= 8


# ─────────────────────────────── features ─────────────────────────────────

def _date_features(s, name):
    dt = _to_dt(s)
    out = pd.DataFrame(index=s.index)
    if dt.notna().mean() < 0.5:
        return out
    out[f"{name}_year"] = dt.dt.year
    out[f"{name}_month"] = dt.dt.month
    out[f"{name}_dayofweek"] = dt.dt.dayofweek
    if dt.dt.hour.nunique() > 1:
        out[f"{name}_hour"] = dt.dt.hour
    return out.astype(float)


def prepare_features(df: pd.DataFrame, target: str, max_categories: int = MAX_LEVELS):
    """Build X and y.

    ID-like and free-text columns are dropped. Dates become year / month /
    weekday features instead of being thrown away. Text columns with many
    values keep their most common levels plus 'Other' rather than being
    dropped outright."""
    y = df[target]
    X = df.drop(columns=[target])
    dropped, parts = [], []
    n = len(X)

    for col in list(X.columns):
        s = X[col]
        nunique = s.nunique(dropna=True)
        if nunique <= 1:
            dropped.append(col)
            continue
        if _is_datetime_like(s):
            parts.append(_date_features(s, col))
            continue
        if pd.api.types.is_numeric_dtype(s):
            # an all-unique integer that counts upward is a row number, not a signal
            if (nunique == n and n > 20 and pd.api.types.is_integer_dtype(s)
                    and (s.is_monotonic_increasing or str(col).lower().endswith("id"))):
                dropped.append(col)
                continue
            parts.append(s.astype(float).to_frame())
            continue
        # text column
        if nunique > 0.5 * n or nunique > 200:
            dropped.append(col)          # names, comments, IDs: not learnable
            continue
        s = s.astype(str)
        if nunique > max_categories:
            top = s.value_counts().nlargest(max_categories).index
            s = s.where(s.isin(top), "Other")
        parts.append(pd.get_dummies(s, prefix=col, drop_first=True).astype(float))

    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Series(pd.factorize(y)[0], index=y.index)
    return X, y, dropped


def _drop_leaks(X, y, task):
    """Remove features that are near-copies of the target."""
    leaks = []
    if X.empty:
        return X, leaks
    yv = pd.Series(np.asarray(y, dtype=float), index=X.index)
    if task == "classification" and yv.nunique() > 2:
        return X, leaks           # correlation is meaningless for unordered classes
    for col in X.columns:
        x = X[col]
        if x.nunique() < 2:
            continue
        r = x.corr(yv)
        if pd.notna(r) and abs(r) >= LEAK_CORR:
            leaks.append(col)
    return X.drop(columns=leaks), leaks


def _add_lags(X, y, order):
    """For time series, yesterday's value is usually the strongest predictor
    of today's. Lags are computed in time order and only look backwards."""
    ys = y.loc[order]
    lags = pd.DataFrame(index=order)
    lags["target_lag1"] = ys.shift(1)
    lags["target_lag2"] = ys.shift(2)
    lags["target_roll3"] = ys.shift(1).rolling(3, min_periods=1).mean()
    X = X.loc[order].join(lags)
    keep = X["target_lag1"].notna()
    return X[keep].fillna(0), y.loc[order][keep]


# ─────────────────────────────── models ───────────────────────────────────

class NaiveLastValue(BaseEstimator, RegressorMixin):
    """Predicts the previous period's value. For time series this is the
    honest baseline: a model that cannot beat 'same as last time' is useless."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(X["target_lag1"], dtype=float)


def _complexity(n_rows):
    """Smaller data gets shallower, more regularised trees, which is the
    main defence against overfitting a few hundred rows. Bigger data gets
    each forest tree trained on a subsample, which is much faster and
    barely changes accuracy."""
    if n_rows < 500:
        return dict(depth=3, leaf=5, trees=150, rf_trees=150, rf_depth=6, rf_sample=None)
    if n_rows < 5000:
        return dict(depth=4, leaf=3, trees=200, rf_trees=120, rf_depth=10, rf_sample=None)
    return dict(depth=6, leaf=2, trees=250, rf_trees=100, rf_depth=14, rf_sample=0.5)


def _get_models(task, n_rows, imbalanced=False, binary=False, pos_weight=1.0,
                log_target=False, time_series=False):
    c = _complexity(n_rows)
    if task == "classification":
        cw = "balanced" if imbalanced else None
        xgb_kw = dict(n_estimators=c["trees"], max_depth=c["depth"], learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                      min_child_weight=c["leaf"], eval_metric="logloss",
                      random_state=42, n_jobs=-1)
        if imbalanced and binary:
            xgb_kw["scale_pos_weight"] = pos_weight
        models = {
            "Baseline (most common class)": DummyClassifier(strategy="most_frequent"),
            "Logistic Regression": make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=2000, class_weight=cw)),
            "Random Forest": RandomForestClassifier(
                n_estimators=c["rf_trees"], max_depth=c["rf_depth"], min_samples_leaf=c["leaf"],
                max_samples=c["rf_sample"], class_weight=cw, random_state=42, n_jobs=-1),
            "XGBoost": XGBClassifier(**xgb_kw),
        }
        return models

    def wrap(m):
        return TransformedTargetRegressor(regressor=m, func=np.log1p, inverse_func=np.expm1) \
            if log_target else m

    models = {
        ("Baseline (previous value)" if time_series else "Baseline (mean)"):
            NaiveLastValue() if time_series else DummyRegressor(strategy="mean"),
        "Ridge Regression": wrap(make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        "Random Forest": wrap(RandomForestRegressor(
            n_estimators=c["rf_trees"], max_depth=c["rf_depth"], min_samples_leaf=c["leaf"],
            max_samples=c["rf_sample"], random_state=42, n_jobs=-1)),
        "XGBoost": wrap(XGBRegressor(
            n_estimators=c["trees"], max_depth=c["depth"], learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            min_child_weight=c["leaf"], random_state=42, n_jobs=-1)),
    }
    return models


def _unwrap(model):
    """Reach the fitted tree inside a log-target wrapper or pipeline."""
    if hasattr(model, "regressor_"):
        model = model.regressor_
    if hasattr(model, "steps"):
        model = model.steps[-1][1]
    return model


def explain_model(model, X: pd.DataFrame):
    """Feature importance: SHAP if it works, otherwise built-in importance."""
    model = _unwrap(model)
    try:
        import shap

        sv = shap.TreeExplainer(model).shap_values(X)
        if isinstance(sv, list):          # older SHAP: one array per class
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:                  # newer SHAP: (rows, features, classes)
            sv = sv[:, :, -1]
        importance = pd.Series(abs(sv).mean(axis=0), index=X.columns)
        method = "SHAP (mean absolute impact)"
    except Exception:
        importance = pd.Series(model.feature_importances_, index=X.columns)
        method = "built-in feature importance (SHAP unavailable)"
    return importance.sort_values(ascending=False), method


def _fit_check(results, best_name, baseline, metric_label):
    """Plain-language verdict on over- and underfitting from the gap between
    training and test scores."""
    row = results[results["model"] == best_name].iloc[0]
    test, train = float(row["mean"]), float(row["train"])
    gap = train - test
    lift = test - baseline
    if lift < 0:
        status = "no better than guessing"
        plain = ("The simple guess beats every model here, so the simple guess is the better "
                 "prediction. For a series that wanders randomly, 'same as last time' is often "
                 "as good as it gets.")
    elif lift < 0.02 and gap > 0.15:
        status = "no signal (memorising noise)"
        plain = ("The model learned the training rows by heart but does no better than guessing "
                 "on new ones. The columns here don't predict this outcome.")
    elif lift < 0.02:
        status = "underfitting"
        plain = ("The model barely beats a simple guess. The columns available don't carry much "
                 "information about this outcome, or the pattern needs data that isn't here.")
    elif gap > 0.15:
        status = "overfitting"
        plain = ("The model does much better on data it has seen than on new data. It is partly "
                 "memorising, so treat its score as optimistic.")
    elif gap > 0.07:
        status = "slight overfitting"
        plain = ("The model does a little better on data it has seen than on new data. That is "
                 "normal, and the test score is the one to trust.")
    else:
        status = "good fit"
        plain = ("The model scores about the same on new data as on data it learned from, so the "
                 "result should hold up.")
    tech = (f"{metric_label}: train {train:.3f}, test {test:.3f}, gap {gap:+.3f}; "
            f"lift over baseline {lift:+.3f}.")
    return {"status": status, "train": train, "test": test, "gap": gap,
            "lift": lift, "plain": plain, "tech": tech}


# ─────────────────────────────── main entry ───────────────────────────────

def auto_model(df: pd.DataFrame, target: str) -> dict:
    df = df[df[target].notna()]
    plan = []
    raw_target = df[target]
    if not pd.api.types.is_numeric_dtype(raw_target) and raw_target.nunique() > 0.5 * len(df):
        raise ValueError(f"'{target}' looks like an ID or free text, so there is nothing to predict.")

    task = detect_task(raw_target)
    time_col = _find_time_column(df, target)
    time_series = _should_split_by_time(df, target, time_col)

    X, y, dropped = prepare_features(df, target)

    classes = None
    if task == "classification":
        codes, uniques = pd.factorize(raw_target, sort=True)
        y = pd.Series(codes, index=df.index)
        classes = [str(u) for u in uniques]
    else:
        y = pd.to_numeric(raw_target, errors="coerce").astype(float)

    X, leaks = _drop_leaks(X, y, task)
    if leaks:
        dropped = dropped + leaks
        plan.append(f"Removed {', '.join(leaks)}: nearly identical to '{target}', so using "
                    f"{'it' if len(leaks) == 1 else 'them'} would be cheating rather than predicting.")
    if X.shape[1] == 0:
        raise ValueError("No usable input columns are left after removing IDs, free text and leaks.")

    # ordering and sampling
    total_rows = len(X)
    if time_series:
        order = _to_dt(df[time_col]).sort_values(kind="stable").index
        if task == "regression":
            X, y = _add_lags(X, y, order)
            plan.append("Added the previous values of the target as inputs, because the recent "
                        "past is usually the best guide to the next value.")
        else:
            X, y = X.loc[order], y.loc[order]
        if len(X) > MAX_ROWS:
            X, y = X.iloc[-MAX_ROWS:], y.iloc[-MAX_ROWS:]
        plan.append(f"Rows follow '{time_col}' in time, so the model is always trained on earlier "
                    "dates and tested on later ones. Shuffling would let it peek at the future.")
    elif len(X) > MAX_ROWS:
        idx = X.sample(MAX_ROWS, random_state=42).index
        X, y = X.loc[idx], y.loc[idx]

    n = len(X)
    if n < 20:
        raise ValueError("Fewer than 20 usable rows: too little data to train and test a model.")
    folds = 5 if n >= 100 else 3

    # imbalance and target shape
    imbalanced, binary, pos_weight, log_target = False, False, 1.0, False
    if task == "classification":
        share = y.value_counts(normalize=True)
        binary = len(share) == 2
        minority = float(share.min())
        if minority < IMBALANCE:
            imbalanced = True
            if binary:
                pos_weight = float(share.max() / share.min())
            plan.append(f"Only {minority:.0%} of rows are '{classes[int(share.idxmin())]}'. Rare cases "
                        "are given extra weight, and the score is balanced accuracy, so a model that "
                        "always predicts the common answer can't look good.")
        if (y.value_counts() < folds).any():
            folds = max(2, int(y.value_counts().min()))
    else:
        if y.min() >= 0 and y.skew() > 2:
            log_target = True
            plan.append(f"'{target}' is heavily skewed by a few very large values, so models learn "
                        "on a log scale and are scored on the original one.")

    if time_series:
        cv = TimeSeriesSplit(n_splits=min(folds, max(2, n // 30)))
        split = "time-ordered"
    elif task == "classification":
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        split = "stratified shuffled"
    else:
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        split = "shuffled"

    if task == "classification":
        scoring = "balanced_accuracy" if imbalanced else "accuracy"
        metric_label = "Balanced accuracy" if imbalanced else "Accuracy"
    else:
        scoring, metric_label = "r2", "R²"

    c = _complexity(n)
    plan.append(f"With {n:,} rows, tree models are limited to depth {c['depth']} "
                + ("and need several rows per leaf, to stop them memorising a small dataset."
                   if n < 5000 else "with row and column subsampling to keep them general."))

    models = _get_models(task, n, imbalanced, binary, pos_weight, log_target, time_series)

    # The folds are fixed once and every model trained during cross-validation
    # is kept. The charts and the feature importance reuse those models, so
    # nothing is trained twice.
    splits = list(cv.split(X, y))
    rows, fitted = [], {}
    for name, model in models.items():
        try:
            cvr = cross_validate(model, X, y, cv=splits, scoring=scoring,
                                 return_train_score=True, return_estimator=True)
            rows.append({"model": name, "mean": cvr["test_score"].mean(),
                         "std": cvr["test_score"].std(), "train": cvr["train_score"].mean()})
            fitted[name] = cvr["estimator"]
        except Exception:
            continue
    if not any(not r["model"].startswith("Baseline") for r in rows):
        raise ValueError("None of the models could be trained on this data.")
    results = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)

    real = results[~results["model"].str.startswith("Baseline")]
    top = real.iloc[0]
    close = real[real["mean"] >= top["mean"] - SIMPLER_MARGIN]
    best_name = min(close["model"], key=lambda m: _SIMPLICITY.get(m, 9))
    if best_name != top["model"]:
        plan.append(f"{best_name} scored within {SIMPLER_MARGIN:.0%} of {top['model']}, so the "
                    "simpler model was chosen: it is easier to explain and less likely to overfit.")

    base_rows = results[results["model"].str.startswith("Baseline")]
    baseline = float(base_rows["mean"].iloc[0]) if len(base_rows) else 0.0
    fit = _fit_check(results, best_name, baseline, metric_label)

    # out-of-fold predictions for the charts, from the models already trained
    # in each fold (works for time-ordered splits too, where some early rows
    # are never in a test fold)
    oof_idx, oof_pred, oof_proba = [], [], []
    for (tr, te), m in zip(splits, fitted[best_name]):
        oof_idx.extend(te)
        if task == "classification":
            proba = m.predict_proba(X.iloc[te])
            full = np.zeros((len(te), len(classes)))
            full[:, list(m.classes_)] = proba
            oof_proba.append(full)
            oof_pred.extend(full.argmax(axis=1))
        else:
            oof_pred.extend(m.predict(X.iloc[te]))
    oof = {"y": y.iloc[oof_idx].reset_index(drop=True), "pred": np.asarray(oof_pred),
           "proba": np.vstack(oof_proba) if oof_proba else None}

    trees = real[real["model"].isin(["Random Forest", "XGBoost"])]
    explain_name = trees.iloc[0]["model"]
    explain_obj = fitted[explain_name][-1]       # trained on the most data of any fold
    sample_X = X.sample(min(len(X), 2000), random_state=42)
    importance, method = explain_model(explain_obj, sample_X)

    fig = px.bar(
        importance.head(10).iloc[::-1], orientation="h",
        title=f"Top features ({explain_name}, {method})",
        labels={"value": "importance", "index": "feature"},
    )
    fig.update_layout(showlegend=False)

    return {
        "task": task,
        "metric": scoring,
        "metric_label": metric_label,
        "split": split,
        "plan": plan,
        "fit_check": fit,
        "dropped_columns": dropped,
        "features_used": list(X.columns),
        "results": results,
        "best_model": best_name,
        "explained_model": explain_name,
        "importance": importance,
        "importance_method": method,
        "figure": fig,
        "X": X,
        "y": y,
        "model_objs": models,
        "cv": cv,
        "oof": oof,
        "classes": classes,
        "modeled_rows": len(X),
        "total_rows": total_rows,
    }