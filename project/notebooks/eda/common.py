import pandas as pd
import pyarrow.parquet as pq

def read_parquet_clean(path, columns=None):
    path = str(path)
    real = pq.read_schema(path).names
    if columns is None:
        df = pd.read_parquet(path)
    else:
        resolved = []
        for col in columns:
            if col in real:
                resolved.append(col)
            elif f"toValidUTF8({col})" in real:
                resolved.append(f"toValidUTF8({col})")
            else:
                raise KeyError(f"Колонка {col!r} не найдена в {path}. Доступны: {real}")
        df = pd.read_parquet(path, columns=resolved)
    df.columns = [
        c[len("toValidUTF8("):-1] if c.startswith("toValidUTF8(") and c.endswith(")") else c
        for c in df.columns
    ]
    return df

def normalize_id(series):
    return series.astype("string").str.strip()

def require_columns(df, required, name):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name}: отсутствуют обязательные колонки: {missing}")
