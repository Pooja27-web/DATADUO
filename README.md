# DATADUO — Auto Analyst & Data Scientist

**Upload any table of data. DATADUO works out what kind of data it is, cleans it, builds the right dashboard, trains and checks prediction models, and explains everything in plain English, with no code required.**

> 🔗 **Live demo:** https://dataduo.streamlit.app/
>** 🎓 BCA Final Year Project · Alliance University · Poojashree H **

---

## The problem

Most people who have data can't analyse it. Turning a raw spreadsheet into answers takes statistics, machine learning and visualisation skills, and even then it's easy to fool yourself: a model that looks 90% accurate can be no better than guessing.

## The solution

DATADUO is an automated analyst that does the whole job in one pass, and is honest about what the data can and can't tell you:

```
Upload → Detect data type → Clean → Domain dashboard → Train & validate models → Explain → Report
```

---

## Key features

### 🧭 Automatic domain detection
DATADUO reads the columns and decides what kind of dataset it is, then builds a dashboard designed for that kind of data:

| Detected as | Dashboard pages |
|---|---|
| **Sales & Commerce** | Executive overview, sales trends, products, customers, geography, profitability |
| **Web & Product Analytics** | Traffic, acquisition, engagement, audience, conversion, content |
| **People & HR** | Workforce, compensation, attrition, performance, hiring & tenure |
| **Finance & Transactions** | Money flow, balances, categories, unusual activity, or market price & risk |
| **Time Series & Operations** | Trend, seasonality, anomalies, volatility, relationships |
| **General (anything else)** | Overview, distributions, categories, relationships, data quality |

Every dataset gets a full dashboard, and pages that would be empty are left out. The guess can be overridden from a dropdown.

### 🤖 Honest machine learning
Instead of one fixed recipe, the model setup adapts to the data and guards against the usual ways a model can mislead:

- **Time-aware validation.** Data that follows time is trained on the past and tested on the future, never shuffled. Models must beat a "same as last time" baseline.
- **Imbalance handling.** Rare outcomes (e.g. 11% employee attrition) are reweighted and scored with balanced accuracy, so "always predict the majority" can't look good.
- **Overfitting check.** Training and test scores are compared, and a plain-English verdict is given: *good fit*, *overfitting*, *underfitting*, or *no better than guessing*.
- **Leak detection.** Columns that are near-copies of the target are removed automatically.
- **Skewed targets** are modelled on a log scale and reported in real units.
- **Size-aware complexity.** Small datasets get simpler, more regularised models, and a simpler model is preferred when it scores about the same.

Models compared: Logistic / Ridge Regression, Random Forest and XGBoost, all measured against a baseline, with **SHAP** feature importance.

### 📊 Adaptive exploratory analysis
Chart types are chosen per column: box plots for skewed data, top-8 + "Other" for many categories, trend lines for dates. They are ranked by effect size, so the most important patterns come first.

### 💬 AI Analyst
Ask questions about your data in plain English. It is powered by Google Gemini with automatic failover across up to six LLM providers, and answers stream as they're written. **The AI only sees summaries of your data, never the raw rows.**

### 👥 Two audiences
Switch between **Everyone** (plain sentences, core charts) and **Technical** (statistics, caveats, full model scores).

### 📄 Downloadable reports
- A general analysis report as a single HTML file, ready to print to PDF
- A 16:9 dashboard report for the detected domain

### 📥 Flexible input
CSV, TSV, Excel (multiple sheets), JSON, Parquet, SQLite (multiple tables), public URLs and Google Sheets links.

---

## Tech stack

| Layer | Tools |
|---|---|
| App & UI | Streamlit, custom CSS theme |
| Data | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost, SHAP |
| Visualisation | Plotly |
| AI | Google Gemini via the OpenAI-compatible API, multi-provider failover |
| Reports | Self-contained HTML |

---

## Project structure

```
DATADUO/
├── app.py                  # Streamlit app: sidebar, 7 tabs, layout
├── agent/
│   └── brain.py            # LLM calls, provider failover, streaming
├── tools/
│   ├── loaders.py          # File, URL and Google Sheets ingestion
│   ├── cleaning.py         # Duplicates, missing values, cleaning log
│   ├── profiler.py         # Dataset profile (what the AI reads)
│   ├── analysis.py         # Adaptive exploratory charts
│   ├── domains.py          # Domain detection + 6 domain dashboards
│   ├── business.py         # Sales & commerce dashboard
│   ├── modeling.py         # Model training, validation, fit check
│   ├── charts.py           # Model and story charts
│   ├── pipeline.py         # Runs clean → analyse → model in order
│   ├── report.py / deck.py # Downloadable HTML reports
│   └── theme.py / palette.py / background.py   # Visual design
├── data/                   # Sample datasets (Titanic, sales)
└── site/index.html         # Landing page
```

---

## Run it locally

**Requirements:** Python 3.10 or newer.

```bash
git clone https://github.com/Pooja27-web/DATADUO.git
cd DATADUO
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
python make_sample_sales.py    # creates the sample sales data
python -m streamlit run app.py
```

### AI setup (optional)
Charts, dashboards, models and reports work without any API key. To enable the AI Analyst, create a `.env` file in the project folder:

```
LLM_API_KEY=your-gemini-api-key
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-flash-latest
```

Get a free key from [Google AI Studio](https://aistudio.google.com). Extra fallback providers can be added as `LLM2_API_KEY`, `LLM2_BASE_URL`, `LLM2_MODEL`, and so on, up to `LLM6_...`.

> ⚠️ Never commit your `.env` file. It is already listed in `.gitignore`.

---

## How it works

1. **Load.** The file is read and column types are recognised.
2. **Clean.** Duplicates are removed and gaps are filled or flagged. Every change is logged.
3. **Detect.** Column names and types are scored against each domain, and the best match is chosen.
4. **Analyse.** Charts are chosen per column and ranked by how strongly each relates to the outcome.
5. **Model.** The validation method, metric and model complexity are chosen to fit the data. Leaks are removed, models are compared with a baseline, and overfitting is checked.
6. **Explain.** Findings are written in plain English, with an optional AI-written summary and chat.

---

## Roadmap

- [ ] Prediction on new data: enter values or upload new rows and get predictions
- [ ] AI code agent: the AI writes and runs its own pandas analysis in a sandbox (`agent/planner.py`, `agent/executor.py`)
- [ ] Domain detection from column *values*, not only names
- [ ] Automated tests and continuous integration
- [ ] Docker image

---

## Author

**Poojashree ** · BCA Final Year · Alliance University
Linkedin : https://linkedin.com/in/poojashree-harish-kumar
EMail : Poojashreeharishkumar@outlook.com

*Patterns found by DATADUO describe the data. They are not proof of cause and effect.*
