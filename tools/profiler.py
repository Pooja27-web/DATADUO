import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> dict:
    columns = []
    for col in df.columns:
        s = df[col]
        info = {
            "name": col,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique": int(s.nunique()),
        }
        if pd.api.types.is_bool_dtype(s):
            info["kind"] = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            info["kind"] = "datetime"
            info["min"] = str(s.min())
            info["max"] = str(s.max())
        elif pd.api.types.is_numeric_dtype(s):
            info["kind"] = "numeric"
            clean = s.dropna()
            if len(clean) > 0:
                q1, q3 = clean.quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = clean[(clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)]
                info.update({
                    "mean": round(float(clean.mean()), 3),
                    "std": round(float(clean.std()), 3) if len(clean) > 1 else 0.0,
                    "min": float(clean.min()),
                    "median": float(clean.median()),
                    "max": float(clean.max()),
                    "outliers": int(len(outliers)),
                })
        else:
            info["kind"] = "categorical"
            info["top_values"] = s.value_counts().head(5).to_dict()
        columns.append(info)
    return {
        "rows": int(df.shape[0]),
        "columns_count": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": columns,
    }


def profile_to_text(profile: dict) -> str:
    lines = [
        f"Rows: {profile['rows']}, Columns: {profile['columns_count']}, "
        f"Duplicate rows: {profile['duplicate_rows']}",
        "",
    ]
    for c in profile["columns"]:
        line = f"- {c['name']} ({c['kind']}, {c['dtype']}): {c['unique']} unique"
        if c["missing"]:
            line += f", {c['missing']} missing ({c['missing_pct']}%)"
        if c["kind"] == "numeric" and "mean" in c:
            line += (f", mean={c['mean']}, min={c['min']}, "
                     f"max={c['max']}, outliers={c['outliers']}")
        if c["kind"] == "categorical":
            line += f", top={c['top_values']}"
        lines.append(line)
    return "\n".join(lines)
