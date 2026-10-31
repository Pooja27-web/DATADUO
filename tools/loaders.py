"""tools/loaders.py — file and URL ingestion for DATADUO."""

import io
import os
import sqlite3
import tempfile

import pandas as pd

SUPPORTED_TYPES = ["csv", "tsv", "xlsx", "xls", "json", "parquet", "db", "sqlite"]


def get_options(name: str, data: bytes):
    """For files that can contain multiple tables (Excel sheets, SQLite
    tables), return (kind, [names]). For everything else, (None, []) so
    the app skips the sub-selection UI entirely."""
    low = name.lower()

    if low.endswith((".xlsx", ".xls")):
        try:
            xl = pd.ExcelFile(io.BytesIO(data))
            sheets = xl.sheet_names
            return ("sheet", sheets) if len(sheets) > 1 else ("sheet", [])
        except Exception:
            return ("sheet", [])

    if low.endswith((".db", ".sqlite")):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            conn = sqlite3.connect(tmp_path)
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
            conn.close()
            names = tables["name"].tolist()
            return ("table", names) if len(names) > 1 else ("table", [])
        except Exception:
            return ("table", [])
        finally:
            if tmp_path:
                os.unlink(tmp_path)

    return (None, [])


def load_table(name: str, data: bytes, option=None) -> pd.DataFrame:
    """Load file bytes into a DataFrame. 'option' is the chosen sheet name
    (Excel) or table name (SQLite) when there was more than one; None picks
    the first / default automatically."""
    low = name.lower()
    buf = io.BytesIO(data)

    if low.endswith(".csv"):
        return _read_csv(buf)
    if low.endswith(".tsv"):
        return pd.read_csv(buf, sep="\t")
    if low.endswith((".xlsx", ".xls")):
        return pd.read_excel(buf, sheet_name=option if option else 0)
    if low.endswith(".json"):
        return _read_json(buf)
    if low.endswith(".parquet"):
        return pd.read_parquet(buf)
    if low.endswith((".db", ".sqlite")):
        return _read_sqlite(data, table=option)

    raise ValueError(f"Unsupported file type: {name}")


def load_url(url: str) -> pd.DataFrame:
    """Load a CSV/TSV/JSON/Parquet from a public URL.
    Also handles Google Sheets share links (converts to CSV export URL).
    Raises ValueError on failure.
    """
    url = url.strip()

    if "docs.google.com/spreadsheets" in url:
        url = _gsheet_to_csv_url(url)

    low = url.lower().split("?")[0]

    try:
        if low.endswith(".tsv"):
            return pd.read_csv(url, sep="\t")
        if low.endswith(".json"):
            return _read_json_url(url)
        if low.endswith(".parquet"):
            return pd.read_parquet(url)
        return _read_csv_url(url)
    except Exception as exc:
        raise ValueError(f"Could not load URL: {exc}") from exc


# ── helpers ───────────────────────────────────────────────────────────────

def _read_csv(buf: io.BytesIO) -> pd.DataFrame:
    buf.seek(0)
    sample = buf.read(4096).decode("utf-8", errors="replace")
    buf.seek(0)
    sep = ","
    if sample.count(";") > sample.count(","):
        sep = ";"
    elif sample.count("|") > sample.count(","):
        sep = "|"
    return pd.read_csv(buf, sep=sep)


def _read_csv_url(url: str) -> pd.DataFrame:
    import urllib.request
    with urllib.request.urlopen(url, timeout=15) as resp:
        content = resp.read()
    return _read_csv(io.BytesIO(content))


def _read_json(buf: io.BytesIO) -> pd.DataFrame:
    buf.seek(0)
    for orient in ("records", "split", "index", "columns", "values", "table"):
        try:
            buf.seek(0)
            return pd.read_json(buf, orient=orient)
        except Exception:
            continue
    buf.seek(0)
    return pd.read_json(buf, lines=True)


def _read_json_url(url: str) -> pd.DataFrame:
    import urllib.request
    with urllib.request.urlopen(url, timeout=15) as resp:
        content = resp.read()
    return _read_json(io.BytesIO(content))


def _read_sqlite(raw: bytes, table: str = None) -> pd.DataFrame:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        conn = sqlite3.connect(tmp_path)
        if not table:
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
            if tables.empty:
                raise ValueError("No tables found in SQLite database.")
            table = tables.iloc[0]["name"]
        df = pd.read_sql(f"SELECT * FROM [{table}]", conn)
        conn.close()
    finally:
        if tmp_path:
            os.unlink(tmp_path)
    return df


def _gsheet_to_csv_url(url: str) -> str:
    import re
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if not match:
        raise ValueError("Cannot parse Google Sheets URL.")
    sheet_id = match.group(1)
    gid_match = re.search(r"gid=(\d+)", url)
    gid = gid_match.group(1) if gid_match else "0"
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"