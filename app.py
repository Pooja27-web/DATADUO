from tools.theme import apply_theme
import copy
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from agent.brain import LAST_PROVIDER, ask, ask_stream, available_providers
from tools.background import background_data_uri
from tools.business import ROLE_LABELS, detect_columns
from tools.domains import build_dashboard, detect_domain
from tools.deck import build_deck_html
from tools.loaders import SUPPORTED_TYPES, get_options, load_table, load_url
from tools.pipeline import run_pipeline
from tools.profiler import profile_dataframe, profile_to_text
from tools.report import build_report_html
from tools.theme import caption, chart_title, guide, inject_css, masthead, register_plotly_theme

st.set_page_config(page_title="DATADUO", layout="wide")
register_plotly_theme()


@st.cache_resource(show_spinner=False)
def cached_bg():
    return background_data_uri()


inject_css(cached_bg())


# ---------- cached helpers ----------
@st.cache_resource(show_spinner=False)
def cached_options(name, data):
    return get_options(name, data)


@st.cache_resource(show_spinner=False)
def cached_load(name, data, option):
    return load_table(name, data, option)


@st.cache_resource(show_spinner=False)
def cached_url(url):
    return load_url(url)


@st.cache_resource(show_spinner=False)
def cached_pipeline(df, target):
    return run_pipeline(df, target)


@st.cache_resource(show_spinner=False)
def cached_report(run_id, ai_text, _res, dataset_name, audience):
    return build_report_html(
        title="Data Analysis Report", dataset_name=dataset_name, target=_res["target"],
        summary_points=_res["summary"], ai_summary=ai_text, insights=_res["insights"],
        charts=_res["story"], log=_res["log"], model_out=_res["model_out"], audience=audience,
    )


@st.cache_resource(show_spinner=False)
def cached_dashboard(df, domain, items):
    return build_dashboard(df, domain=domain, mapping=dict(items) if items else None)


@st.cache_resource(show_spinner=False)
def cached_deck(df, domain, items, dataset_name, audience):
    return build_deck_html(cached_dashboard(df, domain, items), dataset_name, audience,
                           bg_uri=cached_bg())


def build_context(res):
    ctx = "DATA PROFILE:\n" + profile_to_text(profile_dataframe(res["cleaned"]))[:12000]
    ctx += "\n\nCLEANING ACTIONS:\n" + "\n".join(res["log"])
    ctx += "\n\nSUMMARY:\n" + "\n".join(res["summary"])
    if res["insights"]:
        ctx += "\n\nINSIGHTS:\n" + "\n".join(res["insights"])
    mo_ = res["model_out"]
    if mo_:
        ctx += "\n\nMODEL RESULTS:\n" + mo_["results"].round(3).to_string(index=False)
        ctx += "\n\nTOP FEATURES:\n" + mo_["importance"].head(8).round(3).to_string()
    return ctx


def extra_note(key):
    """(plain, technical) note for the extra figures produced by tools.analysis."""
    if key == "correlation_heatmap":
        return ("Shows which numbers move together. Green means they rise together, red means one rises as the other falls.",
                "Pearson correlation matrix of numeric columns. |r| above 0.7 between inputs signals multicollinearity.")
    if key.startswith("dist_"):
        return ("Shows how common each value is, split by outcome. Where the two colours separate, this column helps tell outcomes apart.",
                "Overlaid histograms by target class. Visible separation suggests discriminative power; confirm with a KS test.")
    if key.startswith("rate_by_") or key.startswith("rate_top_"):
        return ("The outcome rate inside each group. Bars that differ a lot mean the group matters.",
                "Target rate by category level, groups under 10 rows hidden. Descriptive only, no significance test applied.")
    if key.startswith("trend_"):
        return ("How the outcome has moved over time.",
                "Monthly mean of the target. Watch for seasonality before trusting a trend as causal.")
    if key.startswith("violin_"):
        return ("The full shape of the outcome in each group, not just the average.",
                "Violin + box plot by category level, used when group sizes are large enough to show distribution shape.")
    if key == "target_distribution":
        return ("How the value we want to understand is spread out.",
                "Histogram of the target. Check skew and outliers before choosing a model.")
    if key.startswith("box_"):
        return ("The typical range of the outcome in each group. Boxes at different heights mean the group matters.",
                "Box plots of the target by category level (median, quartiles, whiskers). Used over a histogram when the distribution is skewed.")
    if key.startswith("scatter_"):
        return ("Each dot is a row. A clear slope means the two columns move together.",
                "Scatter of the target against a numeric column. Look for non-linearity and heteroscedasticity.")
    return ("An extra view of the data.", "Extra diagnostic view.")


def show_chart(c, key, technical, height=None):
    chart_title(c["title"])
    fig = copy.deepcopy(c["fig"])
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, key=key)
    caption((c.get("tech") or c["note"]) if technical else c["note"])


masthead(
    "DATADUO", "Automated analysis",
    "Analyse a table of data in one pass",
    "Load a CSV, Excel, JSON or SQLite file. The app cleans it, finds the patterns that matter, tests "
    "prediction models against a plain baseline and writes a report you can send. Choose Everyone or "
    "Technical in the sidebar to change how results are explained.",
)

# ---------- sidebar ----------
raw, dataset_name = None, ""
SAMPLES = {"Sample: Titanic": "data/titanic.csv", "Sample: Sales (fake)": "data/sample_sales.csv"}
with st.sidebar:
    st.markdown("### 1  Data")
    source = st.radio("Source", ["Upload files", "Link or Google Sheet"] + list(SAMPLES),
                      label_visibility="collapsed")
    try:
        if source == "Upload files":
            files = st.file_uploader("Drop files here", type=SUPPORTED_TYPES, accept_multiple_files=True,
                                     help="CSV, TSV, Excel, JSON, Parquet or SQLite")
            if files:
                names = [f.name for f in files]
                pick = st.selectbox("Which file?", names) if len(names) > 1 else names[0]
                data = files[names.index(pick)].getvalue()
                kind, options = cached_options(pick, data)
                option = st.selectbox(f"Which {kind}?", options) if options else None
                raw = cached_load(pick, data, option)
                dataset_name = pick + (f" ({option})" if option else "")
        elif source == "Link or Google Sheet":
            url = st.text_input("Paste a link", placeholder="https://docs.google.com/spreadsheets/d/...")
            st.caption("Google Sheets must be shared as 'Anyone with the link can view'.")
            if url:
                raw = cached_url(url)
                dataset_name = url
        else:
            path = Path(SAMPLES[source])
            if path.exists():
                raw = cached_load(path.name, path.read_bytes(), None)
                dataset_name = path.name
            else:
                st.warning(f"{path} not found. Run 'python make_sample_sales.py' first.")
    except Exception as e:
        st.error(f"Couldn't read that data: {e}")
        raw = None

    target, technical, provider = None, False, None
    if raw is not None and not raw.empty:
        st.markdown("### 2  Question")
        cols = list(raw.columns)
        guesses = [c for c in ("Survived", "Profit", "Sales") if c in cols]
        default = cols.index(guesses[0]) if guesses else len(cols) - 1
        target = st.selectbox("What do you want to understand or predict?", cols, index=default,
                              help="Pick the outcome column, for example price, churn or survived.")
        st.markdown("### 3  Audience")
        audience = st.radio("Explain results for", ["Everyone", "Technical"], label_visibility="collapsed",
                            help="Everyone: fewer charts, plain sentences. Technical: more charts, statistics and caveats.")
        technical = audience == "Technical"
        st.markdown("### 4  AI model")
        providers = available_providers()
        if providers:
            choice = st.selectbox("AI model", ["Auto (best available)"] + providers)
            provider = None if choice.startswith("Auto") else choice
        else:
            st.warning("No AI configured. Charts and reports still work. Add keys to .env for the chat.")

if raw is None or raw.empty or target is None:
    st.info("Add some data in the sidebar to begin. The two samples are a quick way to see how it works.")
    st.stop()

audience = "Technical" if technical else "Everyone"

# ---------- run the pipeline ----------
with st.spinner("Cleaning data, finding patterns and training models..."):
    try:
        res = cached_pipeline(raw, target)
    except Exception as e:
        st.error(f"Something went wrong while analysing this data: {e}")
        st.stop()

run_id = hashlib.md5(f"{dataset_name}|{target}|{raw.shape}".encode()).hexdigest()[:12]
mo = res["model_out"]
ai_text = st.session_state.get("ai_summary", {}).get(run_id, "")

tab_sum, tab_biz, tab_charts, tab_model, tab_data, tab_chat, tab_report = st.tabs(
    ["Executive Overview", "Business Intelligence", "Exploratory Analysis",
     "ML Studio", "Data Quality", "AI Analyst", "Reports"]
)

# ---------- Summary ----------
with tab_sum:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{res['rows_total']:,}")
    c2.metric("Columns", res["raw"].shape[1])
    c3.metric("Data health", f"{res['health']}/100")
    if mo:
        label = mo.get("metric_label", "Accuracy" if mo["task"] == "classification" else "R²")
        c4.metric(f"Best model ({label})", f"{res['score']:.0%}",
                  delta=f"{res['score'] - res['baseline']:+.0%} vs guessing")
    else:
        c4.metric("Best model", "n/a")

    st.subheader("In plain English")
    st.markdown("\n".join(f"- {s}" for s in res["summary"]))
    if ai_text:
        st.markdown("**AI-written summary**")
        st.write(ai_text)

    if technical:
        st.subheader("Technical summary")
        lines = []
        if mo:
            r = mo["results"]
            best_row = r[r["model"] == mo["best_model"]].iloc[0]
            base_row = r[r["model"].str.startswith("Baseline")].iloc[0]
            lines.append(f"Task: {mo['task']}. Metric: {mo['metric']}, 5-fold cross-validation.")
            lines.append(f"Best model {mo['best_model']}: {best_row['mean']:.3f} (std {best_row['std']:.3f}). "
                         f"Baseline: {base_row['mean']:.3f}.")
            lines.append(f"{len(mo['features_used'])} input features after one-hot encoding. "
                         f"{mo['modeled_rows']:,} of {mo['total_rows']:,} rows used for modeling.")
        total_cells = max(raw.shape[0] * raw.shape[1], 1)
        lines.append(f"Missing cells: {int(raw.isna().sum().sum()):,} ({raw.isna().sum().sum() / total_cells:.1%}). "
                     f"Duplicate rows: {int(raw.duplicated().sum()):,}.")
        st.markdown("\n".join(f"- {s}" for s in lines))

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Rewrite this summary with AI", key="ai_sum_btn"):
            prompt = ("Rewrite these findings as a short, friendly summary for someone with no technical "
                      "background. Plain text only (no markdown, no bullet symbols), under 120 words. "
                      "Use only these facts:\n" + "\n".join(res["summary"] + res["insights"][:6]))
            with st.spinner("Writing..."):
                try:
                    text = ask(prompt, system="You explain data findings simply and honestly.", provider=provider)
                    st.session_state.setdefault("ai_summary", {})[run_id] = text
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    with b2:
        st.download_button("Download the general report", key="dl_report_top",
                           data=cached_report(run_id, ai_text, res, dataset_name, audience),
                           file_name="analysis_report.html", mime="text/html")

    if res["insights"]:
        with st.expander("Key findings", expanded=not technical):
            st.markdown("\n".join(f"- {s}" for s in res["insights"]))
    if res["sampled"]:
        st.caption("Large dataset: a random sample of 100,000 rows was analysed to keep things fast.")

# ---------- Business dashboard ----------
with tab_biz:
    ranked = detect_domain(raw)
    dom_labels = [lab for _, lab, _ in ranked]
    dom_keys = [k for k, _, _ in ranked]
    top_conf = ranked[0][2] if ranked else 0.0

    c_pick, c_note = st.columns([2, 3])
    with c_pick:
        chosen_dom = st.selectbox(
            "Analyse this data as", dom_labels, index=0, key=f"domain_{dataset_name}",
            help="DATADUO guesses the kind of dataset from your column names. "
                 "Change this if the guess is wrong.")
    domain = dom_keys[dom_labels.index(chosen_dom)]
    with c_note:
        st.write("")
        if domain == dom_keys[0] and top_conf >= 0.5:
            st.caption(f"Detected automatically as **{dom_labels[0]}** from your column names.")
        elif domain == "generic":
            st.caption("General dashboard: works on any table, with no domain assumed.")
        else:
            st.caption("Showing the dashboard you picked rather than the detected one.")

    guide("How to read this page",
          ["Each page answers one question, shown under its title.",
           "The sentence under every chart says what the chart shows.",
           "Switch to Technical in the sidebar for more charts and the statistics behind them."]
          if not technical else
          ["Core charts come first. Extra diagnostics follow in the same grid.",
           "Notes under each chart name the statistic used and its main caveat.",
           "Column roles are inferred from names and types. Change the dataset kind above if a page looks wrong."])

    mapping = {}
    if domain == "sales":
        guessed = detect_columns(raw)
        with st.expander("How we understood your columns (change anything that looks wrong)",
                         expanded="sales" not in guessed):
            st.caption("Pick the right column for each role. Leave '(none)' if you don't have it.")
            options = ["(none)"] + list(raw.columns)
            grid = st.columns(3)
            for i, (role, label) in enumerate(ROLE_LABELS.items()):
                idx = options.index(guessed[role]) if role in guessed else 0
                choice = grid[i % 3].selectbox(label, options, index=idx,
                                               key=f"map_{role}_{dataset_name}")
                if choice != "(none)":
                    mapping[role] = choice

    items = tuple(sorted(mapping.items())) if mapping else None
    pages = None
    try:
        pages = cached_dashboard(raw, domain, items)
    except Exception as e:
        st.warning(f"The {chosen_dom} dashboard needs columns this data doesn't have ({e}). "
                   "Showing the general dashboard instead.")
        domain, items = "generic", None
        try:
            pages = cached_dashboard(raw, domain, items)
        except Exception as e2:
            st.error(f"Couldn't build a dashboard: {e2}")

    if pages:
        names = [p["title"] for p in pages]
        if hasattr(st, "segmented_control"):
            chosen_page = st.segmented_control("Page", names, default=names[0],
                                               key=f"biz_page_{domain}",
                                               label_visibility="collapsed") or names[0]
        else:
            chosen_page = st.radio("Page", names, horizontal=True, label_visibility="collapsed",
                                   key=f"biz_page_{domain}")
        pg = pages[names.index(chosen_page)]
        st.subheader(pg["title"])
        st.caption(pg["subtitle"])
        if pg["kpis"]:
            kcols = st.columns(len(pg["kpis"]))
            for kc, (k, v) in zip(kcols, pg["kpis"]):
                kc.metric(k, v)
        charts = [c for c in pg["charts"] if technical or c["level"] == "core"]
        if pg["message"] and not charts:
            st.info(pg["message"])
        cols2 = st.columns(2)
        for i, c in enumerate(charts):
            with cols2[i % 2]:
                show_chart(c, f"biz_{pg['id']}_{i}", technical, height=330)
        st.divider()
        st.download_button(
            f"Download the report (16:9, {audience.lower()} edition)", key="dl_deck",
            data=cached_deck(raw, domain, items, dataset_name, audience),
            file_name=f"{domain}_report_{audience.lower()}.html", mime="text/html")
        st.caption("Open the file in your browser. For a PDF: Ctrl + P, Save as PDF, "
                   "margins None, tick Background graphics.")

# ---------- Data charts ----------
with tab_charts:
    guide("How to read this page",
          ["These charts check the data before any model is trained: gaps first, then the outcome itself.",
           "The sentence under each chart says what it shows."] if not technical else
          ["Data quality and target distribution first, then extra diagnostics from the analysis step.",
           "Notes carry the statistic and the caveat. Extra figures are shown only in this view."])
    shown = [c for c in res["story"] if c["section"].startswith(("1.", "2.")) and (technical or c["level"] == "core")]
    if shown:
        cols2 = st.columns(2)
        for i, c in enumerate(shown):
            with cols2[i % 2]:
                show_chart(c, f"story_{c['key']}", technical)
    if technical and res["extra_figures"]:
        st.subheader("Extra diagnostics")
        cols3 = st.columns(2)
        for i, (k, fig) in enumerate(res["extra_figures"].items()):
            plain, tech = extra_note(k)
            with cols3[i % 2]:
                show_chart({"title": k.replace("_", " "), "fig": fig, "note": plain, "tech": tech},
                           f"adv_{k}", True)

# ---------- Model ----------
with tab_model:
    if mo:
        label = mo.get("metric_label", "Accuracy" if mo["task"] == "classification" else "R²")
        if technical:
            st.success(f"Best model: {mo['best_model']}. {label}: {res['score']:.1%} "
                       f"(baseline {res['baseline']:.1%}, {mo.get('split', 'shuffled')} validation).")
        elif mo["task"] == "classification" and mo.get("metric") == "balanced_accuracy":
            st.success(f"Counting every outcome equally, the model is right {res['score']:.0%} of the time. "
                       f"Guessing would score {res['baseline']:.0%}.")
        elif mo["task"] == "classification":
            st.success(f"The model is right {res['score']:.0%} of the time. Simply guessing would be right "
                       f"{res['baseline']:.0%} of the time.")
        else:
            st.success(f"The model explains {max(res['score'], 0):.0%} of the ups and downs in {target}.")

        fc = mo.get("fit_check")
        if fc:
            show = {"good fit": st.success, "slight overfitting": st.info}.get(fc["status"], st.warning)
            show(f"Fit check: {fc['status']}. {fc['plain']}")
            if technical:
                st.caption(fc["tech"])
        if mo.get("plan"):
            with st.expander("How DATADUO set this model up for your data", expanded=technical):
                st.markdown("\n".join(f"- {p}" for p in mo["plan"]))

        for c in [c for c in res["story"] if c["section"].startswith(("3.", "4.")) and (technical or c["level"] == "core")]:
            show_chart(c, f"model_{c['key']}", technical)

        if technical:
            st.subheader("Scores for every model")
            st.dataframe(mo["results"].round(3))
            if mo["modeled_rows"] < mo["total_rows"]:
                st.caption(f"Models were trained on a {mo['modeled_rows']:,}-row sample to stay fast.")
            if mo["dropped_columns"]:
                st.caption("Not used as inputs (IDs or free text): " + ", ".join(mo["dropped_columns"]))
        else:
            with st.expander("What do these words mean?"):
                st.markdown(
                    "- **Baseline**: the score from simply guessing. A model must beat it.\n"
                    "- **Accuracy**: the share of correct predictions.\n"
                    "- **R²**: how much of the ups and downs the model explains (0% none, 100% all).\n"
                    "- **Cross-validation**: testing on data the model hasn't seen, 5 times."
                )
    else:
        st.error(f"Modeling failed: {res['model_error']}")

# ---------- Data and cleaning ----------
with tab_data:
    st.subheader("What we fixed")
    if res["log"]:
        for line in res["log"]:
            st.write("-", line)
    else:
        st.write("Nothing needed cleaning.")
    st.subheader("Cleaned data")
    st.dataframe(res["cleaned"].head(50))
    st.download_button("Download cleaned data (CSV)", data=res["cleaned"].to_csv(index=False),
                       file_name="cleaned_data.csv", mime="text/csv", key="dl_clean")
    with st.expander("Original data (first 50 rows)"):
        st.dataframe(res["raw"].head(50))
    with st.expander("Data profile (what the AI reads)", expanded=technical):
        st.code(profile_to_text(profile_dataframe(res["raw"])))

# ---------- Ask AI ----------
with tab_chat:
    st.caption("The AI only sees summaries of your data, never your raw rows. "
               "Try: 'What should I focus on?' or 'Explain the top factor simply.'")
    if st.session_state.get("chat_run") != run_id:
        st.session_state.chat = []
        st.session_state.chat_run = run_id
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Ask anything about your data...")
    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        history = "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.chat[-7:-1])
        prompt = (f"{build_context(res)}\n\nTarget column: {target}\n\n"
                  f"Conversation so far:\n{history}\n\nQuestion: {q}")
        style = ("Use precise technical language where helpful." if technical
                 else "Explain in simple everyday language and avoid jargon.")
        system = ("You are a careful data analyst. Answer using ONLY the facts provided. "
                  "If the facts don't support an answer, say so. Mention caveats such as correlation "
                  "not being causation. " + style)
        with st.chat_message("assistant"):
            try:
                reply = st.write_stream(ask_stream(prompt, system=system, provider=provider))
                st.caption(f"Answered by {LAST_PROVIDER['name']}")
            except Exception as e:
                reply = f"Error: {e}"
                st.error(reply)
        st.session_state.chat.append({"role": "assistant", "content": reply})

# ---------- Report ----------
with tab_report:
    st.subheader("Your reports")
    st.markdown(
        f"Both reports follow the audience chosen in the sidebar (now: {audience}). "
        "Each is a single file that opens in any browser. For a PDF, press Ctrl + P and choose Save as PDF."
    )
    st.markdown("**General report.** Plain-English summary, charts with notes, cleaning log and glossary.")
    st.download_button("Download general report", key="dl_report_tab",
                       data=cached_report(run_id, ai_text, res, dataset_name, audience),
                       file_name="analysis_report.html", mime="text/html")
    st.markdown("**Business report.** Six 16:9 pages for sales data. It is built in the Business dashboard tab.")