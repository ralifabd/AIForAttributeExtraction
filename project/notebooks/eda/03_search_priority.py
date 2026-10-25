import pandas as pd
import pyarrow.parquet as pq
from config import PATH_ATTRIBUTES, PATH_SEARCH, OUTPUT_DIR, SERVICE_ATTRIBUTES
from common import read_parquet_clean, require_columns

def main():
    if not PATH_SEARCH.exists():
        print(f"\n[03] SEARCH: файл не найден, пропускаю: {PATH_SEARCH}")
        return

    raw_columns = pq.read_schema(str(PATH_SEARCH)).names
    clean_columns = {
        c[len("toValidUTF8("):-1] if c.startswith("toValidUTF8(") and c.endswith(")") else c
        for c in raw_columns
    }
    optional = ["query_text", "sku_position", "sku_price", "sku_subject_id"]
    columns = ["sku_id"] + [c for c in optional if c in clean_columns]
    traffic = read_parquet_clean(PATH_SEARCH, columns=columns)
    require_columns(traffic, ["sku_id"], "query_clicks")

    shows = traffic.groupby("sku_id").size().rename("shows").reset_index()
    total_shows = int(shows["shows"].sum())

    attrs = pd.read_parquet(PATH_ATTRIBUTES, columns=["sku_id", "attr", "value"])
    attrs["value"] = attrs["value"].astype("string").str.strip()
    attrs = attrs[attrs["value"].notna() & attrs["value"].ne("")]
    attrs = attrs[["sku_id", "attr"]].drop_duplicates()

    weighted = attrs.merge(shows, on="sku_id", how="inner", validate="many_to_one")
    priority = weighted.groupby("attr", observed=True).agg(
        sku_with_shows=("sku_id", "nunique"),
        impressions_with_value=("shows", "sum"),
    ).reset_index()
    priority["impression_coverage"] = priority["impressions_with_value"] / max(total_shows, 1)
    priority["is_service_candidate"] = priority["attr"].astype(str).str.lower().isin(SERVICE_ATTRIBUTES)
    priority = priority.sort_values("impressions_with_value", ascending=False)
    priority.to_parquet(OUTPUT_DIR / "attribute_priority.parquet", index=False)

    print("\n[03] SEARCH PRIORITY")
    print(f"Строк лога: {total_shows:,}; SKU с показами: {shows['sku_id'].nunique():,}")
    print(priority[["attr", "impressions_with_value", "impression_coverage", "is_service_candidate"]].head(30).to_string(index=False))
    print("impression_coverage — доля строк лога, относящихся к SKU с заполненным атрибутом.")
    print("Это не доказывает влияние атрибута на CTR или конверсию.")
    print("Сохранено: attribute_priority.parquet")

if __name__ == "__main__":
    main()