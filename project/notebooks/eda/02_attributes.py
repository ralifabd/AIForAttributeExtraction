import pandas as pd
from config import PATH_ATTRIBUTES, PATH_SKUS, OUTPUT_DIR, SERVICE_ATTRIBUTES
from common import require_columns

def main():
    attrs = pd.read_parquet(PATH_ATTRIBUTES)
    skus = pd.read_parquet(PATH_SKUS)
    require_columns(attrs, ["sku_id", "categoryId", "attr", "value", "unit", "is_search"], "attrs")
    require_columns(skus, ["sku_id"], "skus_meta")

    attrs = attrs.copy()
    attrs["attr"] = attrs["attr"].astype("string").str.strip()
    attrs["value"] = attrs["value"].astype("string").str.strip()
    valid_value = attrs["value"].notna() & attrs["value"].ne("")
    total_skus = max(skus["sku_id"].nunique(), 1)

    rows = attrs.groupby("attr", observed=True).size().rename("row_count")
    sku_count = attrs.groupby("attr", observed=True)["sku_id"].nunique().rename("sku_count")
    filled = attrs.loc[valid_value]
    filled_count = filled.groupby("attr", observed=True)["sku_id"].nunique().rename("sku_with_value")
    cardinality = filled.groupby("attr", observed=True)["value"].nunique().rename("distinct_values")
    unit_count = attrs.loc[attrs["unit"].notna()].groupby("attr", observed=True)["unit"].nunique().rename("distinct_units")
    search_share = attrs.groupby("attr", observed=True)["is_search"].mean().rename("search_row_share")

    profile = pd.concat([rows, sku_count, filled_count, cardinality, unit_count, search_share], axis=1).fillna(0).reset_index()
    profile["sku_coverage"] = profile["sku_count"] / total_skus
    profile["value_coverage"] = profile["sku_with_value"] / total_skus
    profile["is_service_candidate"] = profile["attr"].astype(str).str.lower().isin(SERVICE_ATTRIBUTES)
    profile = profile.sort_values(["sku_with_value", "row_count"], ascending=False)
    profile.to_parquet(OUTPUT_DIR / "attribute_profile.parquet", index=False)

    multi = (filled.groupby(["sku_id", "attr"], observed=True)["value"].nunique()
             .rename("distinct_values").reset_index())
    multi = multi[multi["distinct_values"] > 1]
    multi.to_parquet(OUTPUT_DIR / "attribute_issues.parquet", index=False)

    print("\n[02] ATTRIBUTES")
    print(f"Строк атрибутов: {len(attrs):,}; уникальных атрибутов: {attrs['attr'].nunique():,}")
    print(f"Атрибутов с частотой <10: {(profile['row_count'] < 10).sum():,}")
    print(f"SKU-атрибутов с несколькими значениями: {len(multi):,}")
    print(f"Атрибутов с несколькими единицами: {(profile['distinct_units'] > 1).sum():,}")
    print("\nТоп-20 по числу SKU с заполненным значением:")
    print(profile[["attr", "sku_with_value", "value_coverage", "distinct_values", "distinct_units"]].head(20).to_string(index=False))
    print("Сохранено: attribute_profile.parquet, attribute_issues.parquet")

if __name__ == "__main__":
    main()
