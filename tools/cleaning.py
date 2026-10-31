import pandas as pd


def clean_dataframe(df: pd.DataFrame, drop_missing_above: float = 60.0):
    """
    Clean a DataFrame automatically.
    Returns (cleaned_df, log) where log lists every action taken.
    """
    df = df.copy()
    log = []

    # 1. Remove duplicate rows
    dupes = int(df.duplicated().sum())
    if dupes:
        df = df.drop_duplicates()
        log.append(f"Removed {dupes} duplicate rows")

    # 2. Handle missing values column by column
    for col in list(df.columns):
        missing = int(df[col].isna().sum())
        if missing == 0:
            continue

        pct = missing / len(df) * 100

        # Too many missing: keep a "was it known?" flag, then drop the column
        if pct > drop_missing_above:
            df[f"{col}_known"] = df[col].notna().astype(int)
            df = df.drop(columns=[col])
            log.append(
                f"Dropped '{col}' ({pct:.1f}% missing) "
                f"and added flag '{col}_known'"
            )
        # Numeric: fill with median
        elif pd.api.types.is_numeric_dtype(df[col]):
            value = df[col].median()
            df[col] = df[col].fillna(value)
            log.append(f"Filled {missing} missing in '{col}' with median ({value})")
        # Categorical: fill with most common value
        else:
            value = df[col].mode()[0]
            df[col] = df[col].fillna(value)
            log.append(f"Filled {missing} missing in '{col}' with most common value ('{value}')")

    return df, log
