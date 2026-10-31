"""Business analytics for sales-style data.

detect_columns() guesses which column plays which role (date, sales, profit...).
build_pages() turns the data into the 6 pages used by the dashboard and the 16:9 report.
Every chart carries a plain note ("note"), a technical note ("tech") and a level:
"core" charts are shown to everyone, "adv" charts only in the technical view.
Set the environment variable AA_DEBUG=1 to see chart errors.
"""
import os
import re
import traceback
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from tools.palette import ACCENT, BAD, COLORWAY, DIVERGING, GOOD, MUTED, OLIVE, SAND, SCALE

DEBUG = bool(os.getenv("AA_DEBUG"))

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
LAB = {"_country": "country", "_region": "region", "_state": "state", "_city": "city"}
PLURAL = {"_country": "countries", "_region": "regions", "_state": "states", "_city": "cities"}

ROLE_LABELS = {
    "date": "Order date",
    "sales": "Sales / revenue",
    "profit": "Profit",
    "cost": "Cost",
    "quantity": "Quantity",
    "discount": "Discount",
    "order_id": "Order ID",
    "customer": "Customer",
    "segment": "Customer segment",
    "category": "Product category",
    "product": "Product",
    "country": "Country",
    "region": "Region",
    "state": "State / province",
    "city": "City",
}

# Order matters: a column is given to the first role that claims it.
_RULES = [
    ("date", ["order_date", "invoice_date", "transaction_date", "sale_date", "date", "order_time", "timestamp"]),
    ("sales", ["sales", "revenue", "net_sales", "turnover", "total_amount", "amount", "total"]),
    ("profit", ["profit", "net_profit", "gross_profit"]),
    ("cost", ["cost", "cogs"]),
    ("quantity", ["quantity", "qty", "units_sold", "units"]),
    ("discount", ["discount", "discount_rate"]),
    ("order_id", ["order_id", "order_no", "order_number", "invoice_no", "invoice", "transaction_id"]),
    ("segment", ["segment", "customer_segment", "customer_type"]),
    ("customer", ["customer_name", "customer", "client_name", "client", "customer_id"]),
    ("category", ["category", "product_category", "sub_category"]),
    ("product", ["product_name", "product", "item_name", "item", "sku", "product_id"]),
    ("country", ["country", "nation"]),
    ("region", ["region", "zone", "territory", "area"]),
    ("state", ["state", "province"]),
    ("city", ["city", "town"]),
]
_NUMERIC = {"sales", "profit", "cost", "quantity", "discount"}
_ID_LIKE = {"customer", "order_id", "product"}

_US = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
_US_LOWER = {k.lower(): v for k, v in _US.items()}
_US_CODES = set(_US.values())


# ------------------------------------------------------------------ detection
def _norm(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _match(name, kw):
    return name == kw or kw in name.split("_") or ("_" in kw and kw in name)


def _dayfirst(s):
    """True for dd/mm/yyyy style text dates (some value has a first number above 12)."""
    ex = s.dropna().astype(str).str.extract(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-]\d{2,4}").dropna()
    if ex.empty:
        return False
    a, b = pd.to_numeric(ex[0]), pd.to_numeric(ex[1])
    return bool((a > 12).any() and not (b > 12).any())


def _to_date(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    dayfirst = _dayfirst(s)
    try:
        return pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=dayfirst)
    except Exception:
        return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)


def _ok(s, role):
    if role in _NUMERIC:
        if pd.api.types.is_datetime64_any_dtype(s):
            return False
        return pd.to_numeric(s, errors="coerce").notna().mean() > 0.8
    if role == "date":
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            return False
        sample = s.dropna().head(300)
        return len(sample) > 0 and _to_date(sample).notna().mean() > 0.8
    if role in _ID_LIKE:
        return True
    return not pd.api.types.is_numeric_dtype(s)


def detect_columns(df):
    """Return {role: column_name} for the roles we can recognise."""
    names = {c: _norm(c) for c in df.columns}
    used, found = set(), {}
    for role, kws in _RULES:
        for kw in kws:
            for c in df.columns:
                if c in used or not _match(names[c], kw):
                    continue
                try:
                    good = _ok(df[c], role)
                except Exception:
                    good = False
                if good:
                    found[role] = c
                    used.add(c)
                    break
            if role in found:
                break
    if "date" not in found:
        for c in df.columns:
            if c not in used and pd.api.types.is_datetime64_any_dtype(df[c]):
                found["date"] = c
                break
    return found


def prepare(df, mapping):
    """Copy the mapped columns into a tidy frame with standard names (_sales, _date...)."""
    d = pd.DataFrame(index=df.index)

    def col(role):
        c = mapping.get(role)
        return c if c in df.columns else None

    c = col("date")
    if c:
        d["_date"] = _to_date(df[c])
    for r in ("sales", "profit", "cost", "quantity", "discount"):
        c = col(r)
        if c:
            d["_" + r] = pd.to_numeric(df[c], errors="coerce")
    for r in ("order_id", "segment", "customer", "category", "product", "country", "region", "state", "city"):
        c = col(r)
        if c:
            s = df[c].astype(str).str.strip()
            d["_" + r] = s.where(~s.isin(["", "nan", "None", "NaN", "<NA>"]), np.nan)
    if "_profit" not in d and "_cost" in d and "_sales" in d:
        d["_profit"] = d["_sales"] - d["_cost"]
    return d


# ------------------------------------------------------------------- helpers
def _has(d, *cols):
    return all(c in d.columns for c in cols)


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    a = abs(v)
    for lim, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= lim:
            return f"{v / lim:.1f}{suf}"
    return f"{v:,.0f}" if a >= 100 else f"{v:,.2f}"


def _clip(s, n=28):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _gini(values):
    v = np.sort(np.asarray(values, dtype=float))
    if len(v) == 0 or v.sum() == 0:
        return 0.0
    i = np.arange(1, len(v) + 1)
    return float((2 * (i * v).sum()) / (len(v) * v.sum()) - (len(v) + 1) / len(v))


def _layout(fig, legend=False):
    fig.update_layout(margin=dict(l=8, r=8, t=10, b=8), showlegend=legend)
    if legend:
        fig.update_layout(legend=dict(orientation="h", y=1.12, x=0))
    return fig


def _chart(title, fig, note, tech=None, level="core"):
    return {"title": title, "fig": fig, "note": note, "tech": tech or note, "level": level}


def _grp(d, key, value="_sales"):
    return d.dropna(subset=[key]).groupby(key)[value].sum().sort_values(ascending=False)


def _top(d, key, n=10, ascending=False):
    g = _grp(d, key)
    return g.sort_values(ascending=True).head(n) if ascending else g.head(n)


def _monthly(d, col):
    t = d.dropna(subset=["_date", col])
    if t.empty:
        return None
    m = t.groupby(t["_date"].dt.to_period("M").dt.to_timestamp())[col].sum().sort_index()
    m.index = pd.DatetimeIndex(m.index).astype("datetime64[ns]")
    return m


def _hbar(g, name="Sales", color=None):
    g = g.iloc[::-1]
    labels = [_clip(i) for i in g.index]
    if color:
        bar = go.Bar(x=g.values, y=labels, orientation="h", marker_color=color)
    else:
        bar = go.Bar(x=g.values, y=labels, orientation="h", marker=dict(color=g.values, colorscale=SCALE))
    fig = go.Figure(bar)
    fig.update_traces(hovertemplate="%{y}: %{x:,.0f}<extra></extra>")
    fig.update_xaxes(title_text=name, tickformat="~s")
    return _layout(fig)


# ------------------------------------------------------ generic chart builders
def c_top(d, key, n, title, label):
    if not _has(d, key, "_sales") or d[key].nunique() < 5:
        return None
    g = _top(d, key, n=n)
    total = d["_sales"].sum()
    if g.empty or not total:
        return None
    share = g.head(3).sum() / total
    return _chart(
        title, _hbar(g),
        f"The top 3 {label} bring in {share:.0%} of all sales. Longer bars mean more sales.",
        f"Total sales per item, top {len(g)} of {d[key].nunique():,} shown. Top 3 share {share:.1%}.")


def c_share(d, key, title):
    if not _has(d, key, "_sales"):
        return None
    g = _top(d, key, n=7)
    total = d["_sales"].sum()
    if len(g) < 2 or not total:
        return None
    vals, names = [float(v) for v in g.values], [_clip(i) for i in g.index]
    rest = total - g.sum()
    if rest > 0.005 * total:
        vals.append(float(rest))
        names.append("Other")
    fig = go.Figure(go.Pie(labels=names, values=vals, hole=0.55, textinfo="percent"))
    fig.update_layout(legend=dict(font=dict(size=10)))
    pct = vals[0] / sum(vals)
    return _chart(
        title, _layout(fig, legend=True),
        f"'{names[0]}' is the biggest slice at {pct:.0%} of sales. Each slice is one group.",
        f"Share of total sales by group, top 7 plus Other. Largest group {pct:.1%} of {d[key].nunique():,} groups.")


def c_pareto(d, keys):
    key = next((k for k in keys if k in d.columns), None)
    if key is None or "_sales" not in d:
        return None
    g = _grp(d, key)
    g = g[g > 0]
    if len(g) < 5:
        return None
    cum = g.cumsum() / g.sum() * 100
    x = np.arange(1, len(g) + 1) / len(g) * 100
    k80 = int((cum.values >= 80).argmax()) + 1
    label = {"_product": "products", "_customer": "customers", "_category": "categories"}[key]
    fig = go.Figure(go.Scatter(x=x, y=cum.values, mode="lines", fill="tozeroy", line=dict(color=ACCENT, width=3)))
    fig.add_hline(y=80, line_dash="dash", line_color=MUTED)
    fig.update_xaxes(title_text=f"% of {label} (best first)", ticksuffix="%")
    fig.update_yaxes(title_text="Cumulative % of sales", ticksuffix="%")
    return _chart(
        f"How concentrated are sales? ({label})", _layout(fig),
        f"Only {k80 / len(g):.0%} of {label} ({k80:,}) produce 80% of sales. The steeper the curve, the more sales depend on a few.",
        f"Cumulative sales share over {label} ranked by sales. {k80:,} of {len(g):,} reach 80%; Gini coefficient {_gini(g.values):.2f}.")


# ------------------------------------------------------------ executive/sales
def c_exec_trend(d):
    if not _has(d, "_date", "_sales"):
        return None
    m = _monthly(d, "_sales")
    if m is None or len(m) < 2:
        return None
    fig = go.Figure()
    fig.add_bar(x=m.index, y=m.values, name="Sales", marker_color=ACCENT)
    if "_profit" in d:
        p = _monthly(d, "_profit")
        if p is not None:
            fig.add_scatter(x=p.index, y=p.values, name="Profit", mode="lines+markers",
                            line=dict(color=GOOD, width=3))
    fig.update_yaxes(tickformat="~s")
    chg = (m.iloc[-1] / m.iloc[-2] - 1) * 100 if m.iloc[-2] else 0
    chg_txt = "flat" if abs(chg) < 0.5 else f"{chg:+.0f}%"
    title = "Sales and profit by month" if "_profit" in d else "Sales by month"
    cv = m.std() / m.mean() if m.mean() else 0
    return _chart(
        title, _layout(fig, legend=True),
        f"Best month: {m.idxmax():%b %Y} ({_fmt(m.max())}). The latest month is {chg_txt} compared with the month before.",
        f"Monthly totals over {len(m)} months. Mean {_fmt(m.mean())}, coefficient of variation {cv:.0%}; last month {chg_txt} vs prior.")


def c_trend_ma(d):
    if not _has(d, "_date", "_sales"):
        return None
    m = _monthly(d, "_sales")
    if m is None or len(m) < 3:
        return None
    fig = go.Figure()
    fig.add_scatter(x=m.index, y=m.values, name="Monthly sales", mode="lines", fill="tozeroy",
                    line=dict(color=ACCENT, width=2))
    fig.add_scatter(x=m.index, y=m.rolling(3).mean().values, name="3-month average", mode="lines",
                    line=dict(color=SAND, width=3, dash="dot"))
    fig.update_yaxes(tickformat="~s")
    note = "The dotted line smooths out month-to-month noise so the real direction is easy to see."
    if len(m) >= 6:
        a, b = m.tail(3).mean(), m.iloc[-6:-3].mean()
        if b:
            note = f"The last 3 months average {_fmt(a)} a month, {a / b - 1:+.0%} compared with the 3 months before."
    slope = np.polyfit(np.arange(len(m)), m.values, 1)[0]
    rel = slope / m.mean() if m.mean() else 0
    return _chart("Sales trend", _layout(fig, legend=True), note,
                  f"Linear trend {rel:+.1%} of the mean per month over {len(m)} months; dotted line is a 3-month rolling mean.")


def c_growth(d):
    if not _has(d, "_date", "_sales"):
        return None
    m = _monthly(d, "_sales")
    if m is None or len(m) < 3:
        return None
    g = m.pct_change().mul(100).replace([np.inf, -np.inf], np.nan).dropna()
    if g.empty:
        return None
    fig = go.Figure(go.Bar(x=g.index, y=g.values, marker_color=[GOOD if v >= 0 else BAD for v in g.values]))
    fig.update_yaxes(ticksuffix="%", title_text="Change vs previous month")
    return _chart(
        "Month-over-month growth", _layout(fig),
        f"Biggest jump: {g.idxmax():%b %Y} ({g.max():+.0f}%). Biggest drop: {g.idxmin():%b %Y} ({g.min():+.0f}%).",
        f"Month-over-month percent change: mean {g.mean():+.1f}%, std {g.std():.1f}%, {int((g < 0).sum())} of {len(g)} months negative.")


def c_weekday(d):
    if not _has(d, "_date", "_sales"):
        return None
    t = d.dropna(subset=["_date", "_sales"])
    g = t.groupby(t["_date"].dt.dayofweek)["_sales"].sum().reindex(range(7)).fillna(0)
    if g.sum() <= 0 or (g > 0).sum() < 3:
        return None
    fig = go.Figure(go.Bar(x=DAYS, y=g.values, marker=dict(color=g.values, colorscale=SCALE)))
    fig.update_yaxes(tickformat="~s", title_text="Sales")
    best = int(g.idxmax())
    ratio = g.max() / g[g > 0].min()
    return _chart(
        "Which day of the week sells most?", _layout(fig),
        f"{DAYS[best]} is the strongest day, with {g.max() / g.sum():.0%} of all sales.",
        f"Sales by weekday. Strongest day is {ratio:.1f}x the weakest; check that order dates carry real days.")


def c_season(d):
    if not _has(d, "_date", "_sales"):
        return None
    t = d.dropna(subset=["_date", "_sales"])
    g = t.groupby([t["_date"].dt.year.rename("year"), t["_date"].dt.month.rename("month")])["_sales"].sum()
    g = g.reset_index()
    if g["month"].nunique() < 4:
        return None
    multi = g["year"].nunique() > 1
    if multi:
        fig = px.line(g, x="month", y="_sales", color=g["year"].astype(str), markers=True,
                      color_discrete_sequence=COLORWAY, labels={"color": "Year"})
    else:
        fig = px.bar(g, x="month", y="_sales", color_discrete_sequence=[ACCENT])
    fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)), ticktext=MONTHS, title_text="")
    fig.update_yaxes(tickformat="~s", title_text="Sales")
    mavg = g.groupby("month")["_sales"].mean()
    best = int(mavg.idxmax())
    return _chart(
        "Seasonality: which months are strongest?", _layout(fig, legend=multi),
        f"{MONTHS[best - 1]} is usually the strongest month. Each line is a year, so you can see if the pattern repeats.",
        f"Sales by calendar month, one line per year. Peak-to-trough ratio of the average month is {mavg.max() / mavg.min():.1f}x.")


def c_order_value(d):
    if "_sales" not in d:
        return None
    o = d.groupby("_order_id")["_sales"].sum() if "_order_id" in d else d["_sales"].dropna()
    o = o[o > 0]
    if len(o) < 30:
        return None
    cap = o.quantile(0.99)
    fig = go.Figure(go.Histogram(x=o[o <= cap].values, nbinsx=40, marker_color=OLIVE))
    fig.update_xaxes(title_text="Order value", tickformat="~s")
    fig.update_yaxes(title_text="Orders")
    return _chart(
        "How big is a typical order?", _layout(fig),
        f"Most orders are worth about {_fmt(o.median())}. A few very large orders lift the average to {_fmt(o.mean())}.",
        f"Order value distribution clipped at the 99th percentile. Median {_fmt(o.median())}, mean {_fmt(o.mean())}, skew {o.skew():.1f}.")


def c_cumulative(d):
    if not _has(d, "_date", "_sales"):
        return None
    m = _monthly(d, "_sales")
    if m is None or len(m) < 4:
        return None
    fig = go.Figure(go.Scatter(x=m.index, y=m.cumsum().values, mode="lines", line=dict(color=GOOD, width=3)))
    fig.update_yaxes(tickformat="~s", title_text="Total sales so far")
    return _chart(
        "Sales building up over time", _layout(fig),
        "Total sales added up month by month. A steeper line means faster growth.",
        "Cumulative sum of monthly sales; changes in slope mark acceleration or slowdown.")


# -------------------------------------------------------------------- product
def c_tree(d):
    if not _has(d, "_category", "_product", "_sales"):
        return None
    g = d.dropna(subset=["_category", "_product"]).groupby(["_category", "_product"])["_sales"].sum()
    g = g.reset_index()
    g = g[g["_sales"] > 0].sort_values("_sales", ascending=False).head(30)
    if len(g) < 3:
        return None
    fig = px.treemap(g, path=["_category", "_product"], values="_sales", color_discrete_sequence=COLORWAY)
    fig.update_layout(margin=dict(l=2, r=2, t=2, b=2))
    return _chart(
        "Product map: category and product", fig,
        "Bigger boxes mean more sales. Colours group products by category.",
        f"Treemap of the top {len(g)} products by sales, grouped by category. Shows composition, not profitability.")


def c_prod_scatter(d):
    if not _has(d, "_product", "_sales", "_profit"):
        return None
    g = (d.dropna(subset=["_product"]).groupby("_product")
         .agg(sales=("_sales", "sum"), profit=("_profit", "sum")).reset_index())
    g = g.sort_values("sales", ascending=False).head(150)
    if len(g) < 5:
        return None
    fig = px.scatter(g, x="sales", y="profit", hover_name="_product", color="profit",
                     color_continuous_scale=DIVERGING, color_continuous_midpoint=0)
    fig.update_coloraxes(showscale=False)
    fig.add_hline(y=0, line_dash="dash", line_color=MUTED)
    fig.update_xaxes(title_text="Sales", tickformat="~s")
    fig.update_yaxes(title_text="Profit", tickformat="~s")
    loss = int((g["profit"] < 0).sum())
    note = (f"Each dot is a product. {loss} of the top {len(g)} sell well but lose money (below the line)."
            if loss else "Each dot is a product. All top products make money (above the line).")
    r = g["sales"].corr(g["profit"])
    return _chart("Sales vs profit by product", _layout(fig), note,
                  f"Sales vs profit for the top {len(g)} products. Correlation r = {r:.2f}; points below zero lose money.")


def c_bottom(d):
    if not _has(d, "_product", "_sales"):
        return None
    g = _top(d, "_product", n=10, ascending=True)
    if len(g) < 3:
        return None
    return _chart(
        "Slowest-selling products", _hbar(g.sort_values(ascending=False), color=SAND),
        "These sell the least. They are candidates to promote, bundle or drop.",
        "Bottom 10 products by total sales. Check they have enough selling history before judging them.")


# ------------------------------------------------------------------- customer
def c_tiers(d):
    if "_segment" in d and "_sales" in d:
        c = c_share(d, "_segment", "Sales by customer segment")
        if c:
            return c
    if not _has(d, "_customer", "_sales"):
        return None
    g = _grp(d, "_customer")
    n, tot = len(g), g.sum()
    if n < 10 or not tot:
        return None
    a, b = max(1, int(n * 0.1)), max(2, int(n * 0.5))
    parts = [("Top 10%", g.iloc[:a].sum()), ("Next 40%", g.iloc[a:b].sum()), ("Bottom 50%", g.iloc[b:].sum())]
    fig = go.Figure(go.Bar(x=[p[0] for p in parts], y=[p[1] / tot * 100 for p in parts],
                           marker_color=[ACCENT, SAND, OLIVE],
                           text=[f"{p[1] / tot:.0%}" for p in parts], textposition="outside"))
    fig.update_yaxes(title_text="% of sales", ticksuffix="%")
    return _chart(
        "Customer value tiers", _layout(fig),
        f"The top 10% of customers bring in {parts[0][1] / tot:.0%} of all sales.",
        f"Customers ranked by sales and split into three groups. Top 10% share {parts[0][1] / tot:.1%} of {n:,} customers.")


def c_orders_hist(d):
    if "_customer" not in d:
        return None
    t = d.dropna(subset=["_customer"])
    o = t.groupby("_customer")["_order_id"].nunique() if "_order_id" in t else t.groupby("_customer").size()
    if len(o) < 5:
        return None
    c = o.clip(upper=10).value_counts().sort_index()
    labels = [str(int(i)) if i < 10 else "10+" for i in c.index]
    fig = go.Figure(go.Bar(x=labels, y=c.values, marker_color=OLIVE))
    fig.update_xaxes(title_text="Orders per customer")
    fig.update_yaxes(title_text="Customers")
    return _chart(
        "Do customers come back?", _layout(fig),
        f"{(o > 1).mean():.0%} of customers bought more than once. The average customer placed {o.mean():.1f} orders.",
        f"Orders per customer, capped at 10+. Median {o.median():.0f}, mean {o.mean():.1f}, repeat rate {(o > 1).mean():.1%}.")


def c_new_returning(d):
    if not _has(d, "_date", "_customer"):
        return None
    t = d.dropna(subset=["_date", "_customer"]).copy()
    t["_m"] = t["_date"].dt.to_period("M").dt.to_timestamp()
    t["_new"] = t["_m"] == t.groupby("_customer")["_m"].transform("min")
    g = t.groupby(["_m", "_new"])["_customer"].nunique().unstack(fill_value=0).sort_index()
    g.index = pd.DatetimeIndex(g.index).astype("datetime64[ns]")
    g = g.iloc[1:]  # the first month always looks "new" because history starts there
    if len(g) < 2:
        return None
    new = g[True] if True in g.columns else pd.Series(0, index=g.index)
    ret = g[False] if False in g.columns else pd.Series(0, index=g.index)
    fig = go.Figure()
    fig.add_bar(x=g.index, y=ret.values, name="Returning", marker_color=ACCENT)
    fig.add_bar(x=g.index, y=new.values, name="New", marker_color=GOOD)
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Active customers")
    share = ret.sum() / max(ret.sum() + new.sum(), 1)
    return _chart(
        "New vs returning customers", _layout(fig, legend=True),
        f"{share:.0%} of monthly buyers are returning customers. Green bars show new customers.",
        f"Unique active customers per month split by first-order month. Returning share {share:.1%}; first month dropped.")


# ----------------------------------------------------------------- geography
def c_geo_bar(d, key, title):
    if not _has(d, key, "_sales"):
        return None
    g = _top(d, key, n=10)
    total = d["_sales"].sum()
    if g.empty or not total:
        return None
    return _chart(
        title, _hbar(g),
        f"{_clip(g.index[0], 24)} leads with {g.iloc[0] / total:.0%} of all sales.",
        f"Sales by {LAB[key]}, top {len(g)} of {d[key].nunique():,}. Top 3 hold {g.head(3).sum() / total:.1%} of sales.")


def _style_geo(fig):
    fig.update_geos(bgcolor="rgba(0,0,0,0)", landcolor="#141414", showlakes=False, showframe=False,
                    coastlinecolor="#2A2A2A", countrycolor="#2A2A2A", subunitcolor="#2A2A2A")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False)
    return fig


@lru_cache(maxsize=1)
def _iso_map():
    g = px.data.gapminder()
    m = dict(zip(g["country"], g["iso_alpha"]))
    m.update({"USA": "USA", "US": "USA", "United States of America": "USA", "UK": "GBR",
              "Great Britain": "GBR", "Russian Federation": "RUS", "South Korea": "KOR",
              "UAE": "ARE", "United Arab Emirates": "ARE", "Czech Republic": "CZE", "Czechia": "CZE",
              "Vietnam": "VNM", "Viet Nam": "VNM", "Turkey": "TUR", "Türkiye": "TUR"})
    return m


def c_geo_map(d):
    if "_sales" not in d:
        return None
    tech = "Choropleth of total sales per area on a linear colour scale; the largest area can hide differences between small ones."
    if "_state" in d:
        g = d.dropna(subset=["_state"]).groupby("_state")["_sales"].sum().reset_index()
        g["code"] = g["_state"].map(
            lambda s: s.upper() if s.upper() in _US_CODES else _US_LOWER.get(s.lower()))
        if len(g) >= 3 and g["code"].notna().mean() >= 0.7:
            g = g.dropna(subset=["code"])
            fig = px.choropleth(g, locations="code", locationmode="USA-states", color="_sales",
                                scope="usa", hover_name="_state", color_continuous_scale=SCALE)
            top = g.sort_values("_sales", ascending=False).iloc[0]
            return _chart("Sales map (US states)", _style_geo(fig),
                          f"Brighter states sell more. {top['_state']} leads with {_fmt(top['_sales'])}.", tech)
    if "_country" in d and d["_country"].nunique() >= 3:
        g = d.dropna(subset=["_country"]).groupby("_country")["_sales"].sum().reset_index()
        g["iso"] = g["_country"].map(_iso_map())
        if g["iso"].notna().mean() >= 0.6:
            g = g.dropna(subset=["iso"])
            fig = px.choropleth(g, locations="iso", color="_sales", hover_name="_country",
                                color_continuous_scale=SCALE)
            top = g.sort_values("_sales", ascending=False).iloc[0]
            return _chart("Sales map (world)", _style_geo(fig),
                          f"Brighter countries sell more. {top['_country']} leads with {_fmt(top['_sales'])}.", tech)
    return None


def c_geo_tree(d):
    keys = [k for k in ("_country", "_region", "_state", "_city") if k in d.columns]
    path = [k for k in keys if d[k].nunique() > 1]
    if len(path) < 2 or "_sales" not in d:
        return None
    g = d.dropna(subset=path).groupby(path)["_sales"].sum().reset_index()
    g = g[g["_sales"] > 0].sort_values("_sales", ascending=False).head(40)
    if len(g) < 3:
        return None
    fig = px.treemap(g, path=path, values="_sales", color_discrete_sequence=COLORWAY)
    fig.update_layout(margin=dict(l=2, r=2, t=2, b=2))
    order = " > ".join(LAB[k] for k in path)
    return _chart("Where sales come from", fig,
                  f"Bigger boxes mean more sales. Boxes are grouped by {order}.",
                  f"Hierarchical totals ({order}), top {len(g)} leaf areas by sales.")


def c_geo_margin(d, key):
    if not _has(d, key, "_sales", "_profit"):
        return None
    g = d.dropna(subset=[key]).groupby(key).agg(s=("_sales", "sum"), p=("_profit", "sum"))
    g = g[g["s"] > 0]
    if len(g) < 2:
        return None
    g["m"] = g["p"] / g["s"] * 100
    g = g.sort_values("m", ascending=False).head(10)
    fig = go.Figure(go.Bar(x=[_clip(i, 14) for i in g.index], y=g["m"].values,
                           marker_color=[GOOD if v >= 0 else BAD for v in g["m"]]))
    fig.update_yaxes(ticksuffix="%", title_text="Profit margin")
    return _chart(
        f"Profit margin by {LAB[key]}", _layout(fig),
        f"Best margin: {_clip(g['m'].idxmax(), 20)} ({g['m'].max():.0f}%). Weakest: {_clip(g['m'].idxmin(), 20)} ({g['m'].min():.0f}%).",
        f"Profit divided by sales per {LAB[key]}, top {len(g)} by margin. Margins on small sales volumes are noisy.")


# --------------------------------------------------------------- profitability
def c_profit_group(d):
    key = next((k for k in ("_category", "_region", "_segment") if k in d.columns), None)
    if not key or not _has(d, "_sales", "_profit"):
        return None
    g = d.dropna(subset=[key]).groupby(key).agg(s=("_sales", "sum"), p=("_profit", "sum"))
    g = g[g["s"] > 0]
    if len(g) < 2:
        return None
    g["m"] = g["p"] / g["s"] * 100
    overall = g["p"].sum() / g["s"].sum() * 100
    g = g.sort_values("p", ascending=False).head(10)
    fig = go.Figure(go.Bar(x=[_clip(i, 14) for i in g.index], y=g["p"].values,
                           marker_color=[GOOD if v >= 0 else BAD for v in g["p"]],
                           text=[f"{m:.0f}%" for m in g["m"]], textposition="outside"))
    fig.update_yaxes(title_text="Profit", tickformat="~s")
    name = {"_category": "category", "_region": "region", "_segment": "segment"}[key]
    return _chart(
        f"Profit by {name}", _layout(fig),
        f"'{_clip(g['p'].idxmax(), 20)}' earns the most profit. '{_clip(g['m'].idxmin(), 20)}' has the thinnest margin. Labels show margin.",
        f"Total profit per {name}; labels are profit margin (profit / sales). Overall margin {overall:.1f}%.")


def c_discount(d):
    if not _has(d, "_discount", "_profit"):
        return None
    t = d.dropna(subset=["_discount", "_profit"])
    if t["_discount"].nunique() < 2:
        return None
    pct = t["_discount"].max() <= 1
    few = t["_discount"].nunique() <= 8
    band = t["_discount"] if few else pd.qcut(t["_discount"], 6, duplicates="drop")
    g = t.groupby(band, observed=True)["_profit"].mean()
    if few:
        labels = [f"{v:.0%}" if pct else f"{v:g}" for v in g.index]
    else:
        labels = [f"{iv.left:.0%}-{iv.right:.0%}" if pct else f"{iv.left:g}-{iv.right:g}" for iv in g.index]
    fig = go.Figure(go.Bar(x=labels, y=g.values, marker_color=[GOOD if v >= 0 else BAD for v in g.values]))
    fig.update_xaxes(title_text="Discount given")
    fig.update_yaxes(title_text="Average profit per line", tickformat="~s")
    neg = [lab for lab, v in zip(labels, g.values) if v < 0]
    note = (f"Profit turns negative from about {neg[0]} discount. Deeper discounts lose money."
            if neg else "Average profit stays positive at every discount level in this data.")
    r = t["_discount"].corr(t["_profit"])
    return _chart("Do discounts hurt profit?", _layout(fig), note,
                  f"Mean profit per order line by discount band. Correlation of discount with profit r = {r:.2f}.")


def c_loss(d):
    if not _has(d, "_product", "_profit"):
        return None
    g = d.dropna(subset=["_product"]).groupby("_product")["_profit"].sum()
    neg = g[g < 0].sort_values()
    if len(neg):
        return _chart(
            "Products that lose money", _hbar(neg.head(10).abs(), name="Loss", color=BAD),
            f"{len(neg)} products lose money overall, {_fmt(neg.sum())} in total.",
            f"Total profit of loss-making products only. {len(neg)} of {len(g):,} products have negative total profit.")
    low = g.sort_values().head(10)
    if len(low) < 3:
        return None
    return _chart(
        "Lowest-profit products", _hbar(low.sort_values(ascending=False), name="Profit", color=SAND),
        "No product loses money overall. These earn the least.",
        "Ten lowest total profits; none are negative.")


def c_profit_trend(d):
    if not _has(d, "_date", "_profit"):
        return None
    p = _monthly(d, "_profit")
    if p is None or len(p) < 2:
        return None
    fig = go.Figure(go.Bar(x=p.index, y=p.values, marker_color=[GOOD if v >= 0 else BAD for v in p.values]))
    fig.update_yaxes(tickformat="~s", title_text="Profit")
    loss = int((p < 0).sum())
    note = (f"{loss} of {len(p)} months made a loss. Best month: {p.idxmax():%b %Y} ({_fmt(p.max())})."
            if loss else f"Every month was profitable. Best month: {p.idxmax():%b %Y} ({_fmt(p.max())}).")
    return _chart("Profit by month", _layout(fig), note,
                  f"Monthly profit totals over {len(p)} months. Mean {_fmt(p.mean())}, std {_fmt(p.std())}.")


# ----------------------------------------------------------------------- KPIs
def _kpi_exec(d):
    s = d["_sales"].sum()
    out = [("Total sales", _fmt(s))]
    if "_profit" in d:
        p = d["_profit"].sum()
        out += [("Total profit", _fmt(p)), ("Profit margin", f"{p / s:.1%}" if s else "n/a")]
    orders = d["_order_id"].nunique() if "_order_id" in d else len(d)
    out.append(("Orders", f"{orders:,}"))
    if "_customer" in d:
        out.append(("Customers", f"{d['_customer'].nunique():,}"))
    if orders:
        out.append(("Avg order value", _fmt(s / orders)))
    return out


def _kpi_sales(d):
    out = [("Total sales", _fmt(d["_sales"].sum()))]
    m = _monthly(d, "_sales")
    if m is not None and len(m):
        out += [("Average per month", _fmt(m.mean())), ("Best month", f"{m.idxmax():%b %Y}")]
        if len(m) >= 6:
            a, b = m.tail(3).mean(), m.iloc[-6:-3].mean()
            if b:
                out.append(("Last 3 months", f"{a / b - 1:+.0%}"))
    return out


def _kpi_product(d):
    key = "_product" if "_product" in d else "_category"
    g = _grp(d, key)
    tot = d["_sales"].sum()
    out = [("Products" if key == "_product" else "Categories", f"{len(g):,}")]
    if key == "_product" and "_category" in d:
        out.append(("Categories", f"{d['_category'].nunique():,}"))
    out.append(("Best seller", _clip(g.index[0], 18)))
    if tot:
        out.append(("Top 3 share", f"{g.head(3).sum() / tot:.0%}"))
    return out


def _kpi_customer(d):
    if "_customer" not in d:
        return [("Segments", f"{d['_segment'].nunique():,}")]
    t = d.dropna(subset=["_customer"])
    n = t["_customer"].nunique()
    o = t.groupby("_customer")["_order_id"].nunique() if "_order_id" in t else t.groupby("_customer").size()
    g = _grp(d, "_customer")
    k = max(1, int(len(g) * 0.1))
    return [("Customers", f"{n:,}"), ("Avg sales / customer", _fmt(t["_sales"].sum() / n)),
            ("Repeat customers", f"{(o > 1).mean():.0%}"), ("Top 10% share", f"{g.head(k).sum() / g.sum():.0%}")]


def _kpi_geo(d, keys, coarse):
    out = [(PLURAL[k].capitalize(), f"{d[k].nunique():,}") for k in keys]
    g = _grp(d, coarse)
    out += [(f"Top {LAB[coarse]}", _clip(g.index[0], 18)), ("Its share", f"{g.iloc[0] / g.sum():.0%}")]
    return out


def _kpi_profit(d):
    s, p = d["_sales"].sum(), d["_profit"].sum()
    out = [("Total profit", _fmt(p)), ("Profit margin", f"{p / s:.1%}" if s else "n/a")]
    t = d.dropna(subset=["_profit"])
    out.append(("Loss-making lines", f"{(t['_profit'] < 0).mean():.0%}"))
    if "_category" in d:
        g = d.dropna(subset=["_category"]).groupby("_category")["_profit"].sum()
        out.append(("Most profitable", _clip(g.idxmax(), 18)))
    return out


# ---------------------------------------------------------------------- pages
def _page(pid, title, subtitle, d, kpi_fn, builders, need=(), need_any=(), missing=""):
    page = {"id": pid, "title": title, "subtitle": subtitle, "kpis": [], "charts": [], "message": ""}
    lacking = (need and not all(c in d.columns for c in need)) or \
              (need_any and not any(c in d.columns for c in need_any))
    if lacking:
        page["message"] = missing
        return page
    try:
        page["kpis"] = kpi_fn(d)
    except Exception:
        if DEBUG:
            traceback.print_exc()
    for build in builders:
        if len(page["charts"]) == 6:
            break
        try:
            c = build(d)
        except Exception:
            c = None
            if DEBUG:
                traceback.print_exc()
        if c:
            page["charts"].append(c)
    # the first four charts are the core story; anything after is for the technical view
    for i, c in enumerate(page["charts"]):
        c["level"] = "core" if i < 4 else "adv"
    if not page["charts"]:
        page["message"] = missing or "Not enough information in the data for this page."
    return page


def build_pages(df, mapping):
    """Return the 6 pages: a list of dicts with title, kpis, charts and an optional message."""
    d = prepare(df, mapping)
    if "_sales" not in d:
        raise ValueError("A sales / revenue column is needed.")
    keys = [k for k in ("_country", "_region", "_state", "_city")
            if k in d.columns and d[k].nunique() > 1]
    coarse = "_region" if "_region" in keys else (keys[0] if keys else None)
    fine = keys[-1] if keys else None

    geo = []
    if coarse:
        geo.append(lambda x: c_geo_bar(x, coarse, f"Sales by {LAB[coarse]}"))
    geo.append(c_geo_map)
    if fine and fine != coarse:
        geo.append(lambda x: c_geo_bar(x, fine, f"Top {PLURAL[fine]} by sales"))
    geo.append(c_geo_tree)
    if coarse:
        geo.append(lambda x: c_geo_margin(x, coarse))

    return [
        _page("exec", "Executive Overview", "How is the business doing overall?", d, _kpi_exec, [
            c_exec_trend,
            lambda x: c_top(x, "_product", 8, "Top products by sales", "products"),
            lambda x: c_top(x, "_category", 8, "Top categories by sales", "categories"),
            lambda x: c_share(x, "_category", "Sales by category"),
            lambda x: c_geo_bar(x, coarse, f"Sales by {LAB[coarse]}") if coarse else None,
            lambda x: c_share(x, "_segment", "Sales by customer segment"),
            lambda x: c_top(x, "_customer", 8, "Top customers by sales", "customers"),
        ]),
        _page("sales", "Sales Analysis", "How are sales moving over time?", d, _kpi_sales,
              [c_trend_ma, c_growth, c_weekday, c_season, c_order_value, c_cumulative], need=("_date",),
              missing="This page needs an order date column. Pick one in the column list."),
        _page("product", "Product Analysis", "Which products drive the results?", d, _kpi_product, [
            lambda x: c_top(x, "_product", 10, "Top 10 products", "products"),
            c_tree,
            lambda x: c_pareto(x, ["_product", "_category"]),
            c_prod_scatter,
            c_bottom,
            lambda x: c_share(x, "_category", "Sales by category"),
        ], need_any=("_product", "_category"),
              missing="This page needs a product or category column. Pick one in the column list."),
        _page("customer", "Customer Analysis", "Who buys from us, and do they come back?", d,
              _kpi_customer, [
                  lambda x: c_top(x, "_customer", 10, "Top 10 customers", "customers"),
                  c_tiers, c_orders_hist, c_new_returning,
                  lambda x: c_pareto(x, ["_customer"]),
              ], need_any=("_customer", "_segment"),
              missing="This page needs a customer column. Pick one in the column list."),
        _page("geo", "Geographical Analysis", "Where do we sell the most?", d,
              lambda x: _kpi_geo(x, keys, coarse), geo, need_any=tuple(keys),
              missing="This page needs a country, region, state or city column with more than one place."),
        _page("profit", "Profitability Analysis", "Where do we make or lose money?", d, _kpi_profit, [
            c_profit_group, c_discount, c_loss, c_profit_trend,
            lambda x: c_geo_margin(x, coarse) if coarse else None,
        ], need=("_profit",),
              missing="This page needs a Profit column (or a Cost column so profit can be worked out)."),
    ]