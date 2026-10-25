import pandas as pd
from config import PATH_CATEGORIES, PATH_SKUS, PATH_TEXT, OUTPUT_DIR
from common import normalize_id, require_columns


categories = pd.read_parquet(PATH_CATEGORIES)

def build_category_paths(categories):
    require_columns(categories, ["category_id", "parent_id", "category_name"], "cat_df")
    if categories["category_id"].duplicated().any():
        raise ValueError("В дереве есть дубли category_id")

    names = dict(zip(categories["category_id"], categories["category_name"]))
    parents = dict(zip(categories["category_id"], categories["parent_id"]))
    valid = set(categories["category_id"].dropna())
    paths, depths = [], []

    for cid in categories["category_id"]:
        cur, seen, parts = cid, set(), []
        while pd.notna(cur):
            if cur in seen:
                raise ValueError(f"Цикл в дереве категорий около {cid}")
            seen.add(cur)
            if cur not in valid:
                parts = []
                break
            name = names.get(cur)
            if pd.notna(name) and str(name).strip():
                parts.append(str(name).strip())
            cur = parents.get(cur, pd.NA)
        paths.append(" / ".join(reversed(parts)) if parts else None)
        depths.append(len(parts) - 1 if parts else -1)

    out = categories.copy()
    out["category_path"] = paths
    out["depth"] = depths
    return out

def main():
    categories = pd.read_parquet(PATH_CATEGORIES)
    skus = pd.read_parquet(PATH_SKUS)
    require_columns(skus, ["sku_id", "categoryId", "name", "vendor", "model"], "skus_meta")
    if skus["sku_id"].duplicated().any():
        raise ValueError("В skus_meta дублируется sku_id")

    tree = build_category_paths(categories)
    tree.to_parquet(OUTPUT_DIR / "category_tree.parquet", index=False)

    skus = skus.copy()
    skus["sku_key"] = normalize_id(skus["sku_id"])
    duplicate_text_ids = 0

    if PATH_TEXT.exists():
        text = pd.read_parquet(PATH_TEXT)
        require_columns(text, ["sku_id"], "sku_desc")
        text = text.copy()
        text["sku_key"] = normalize_id(text["sku_id"])
        for col in ("title", "description"):
            if col not in text.columns:
                text[col] = pd.NA
        duplicate_text_ids = int(text["sku_key"].duplicated().sum())

        text = text.groupby("sku_key", as_index=False, dropna=False).agg({
            "title": lambda s: next((str(x) for x in s if pd.notna(x) and str(x).strip()), None),
            "description": lambda s: next((str(x) for x in s if pd.notna(x) and str(x).strip()), None),
        })
        master = skus.merge(text, on="sku_key", how="left", validate="one_to_one")
    else:
        master = skus.copy()
        master["title"], master["description"] = pd.NA, pd.NA

    tree_cols = tree[["category_id", "category_name", "parent_id", "category_path", "depth"]]
    master = master.merge(tree_cols, left_on="categoryId", right_on="category_id",
                          how="left", validate="many_to_one")
    master["has_title"] = master["title"].fillna("").astype(str).str.strip().ne("")
    master["has_description"] = master["description"].fillna("").astype(str).str.strip().ne("")
    master["has_text"] = master["has_title"] | master["has_description"]
    master.to_parquet(OUTPUT_DIR / "catalog_master.parquet", index=False)

    print("\n[01] CATALOG")
    print(f"SKU: {len(master):,}; уникальных sku_id: {master['sku_id'].nunique():,}")
    print(f"SKU с текстом: {master['has_text'].sum():,} ({master['has_text'].mean():.1%})")
    print(f"SKU без category_path: {master['category_path'].isna().sum():,}")
    print(f"Дубли sku_id в sku_desc до агрегации: {duplicate_text_ids:,}")
    print("Сохранено: catalog_master.parquet, category_tree.parquet")

if __name__ == "__main__":
    main()
