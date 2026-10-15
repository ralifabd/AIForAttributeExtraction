# ============================================================
# eda_skus.py
# EDA по skus.pkl (YML-каталог М.Видео)
# ============================================================
import os
import re
import pickle
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ---------- глобальные настройки ----------
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)

RED       = "#D32F2F"
DARK_RED  = "#8B0000"
BLACK     = "#0D0D0D"
GRAY      = "#4D4D4D"
WHITE     = "#F5F5F5"

plt.rcParams.update({
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "axes.edgecolor": BLACK,
    "axes.labelcolor": BLACK,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
    "text.color": BLACK,
    "axes.titlecolor": BLACK,
    "grid.color": GRAY,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "figure.figsize": (10, 5),
})

PLOTS = "plots"
os.makedirs(PLOTS, exist_ok=True)

# ---------- пути (подстрой под себя) ----------
PATH_SKUS    = "../data/skus.pkl"
PATH_CATALOG = "../data/sku_desc.parquet"     # title + description
PATH_SEARCH  = "../data/query_clicks.parquest"

# ============================================================
# ХЕЛПЕРЫ
# ============================================================
import pyarrow.parquet as pq

def read_parquet_clean(path, columns=None):
    """Читает parquet и снимает обёртку toValidUTF8(...) с имён колонок.
    Позволяет запрашивать колонки без обёртки."""
    real_cols = pq.read_schema(path).names

    if columns is not None:
        resolved = []
        for c in columns:
            if c in real_cols:
                resolved.append(c)
            elif f"toValidUTF8({c})" in real_cols:
                resolved.append(f"toValidUTF8({c})")
            else:
                raise KeyError(f"Колонка {c!r} не найдена в {path}.\n"
                               f"Доступные: {real_cols}")
        df = pd.read_parquet(path, columns=resolved)
    else:
        df = pd.read_parquet(path)

    # снять toValidUTF8(...) со всех имён
    df.columns = [
        c[len("toValidUTF8("):-1]
        if c.startswith("toValidUTF8(") and c.endswith(")")
        else c
        for c in df.columns
    ]
    return df

def save(fig_name):
    path = os.path.join(PLOTS, fig_name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[saved] {path}")


def bar_vertical(series, title, xlabel, ylabel="Количество",
                 rotation=45, fname="bar.png", top_n=None):
    s = series.head(top_n) if top_n else series
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(s.index.astype(str), s.values,
                  color=RED, edgecolor=BLACK, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.bar_label(bars, fmt="%d", padding=3, color=BLACK, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=rotation, ha="right")
    save(fname)


def bar_horizontal(series, title, xlabel, fname="barh.png", top_n=None,
                   label_fmt="%d"):
    s = series.head(top_n) if top_n else series
    s = s.sort_values()
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(s))))
    bars = ax.barh(s.index.astype(str), s.values,
                   color=RED, edgecolor=BLACK, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.bar_label(bars, fmt=label_fmt, padding=3, color=BLACK, fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    save(fname)


def hist_series(series, title, xlabel, fname, bins=50, logx=False):
    fig, ax = plt.subplots(figsize=(10, 5))
    data = series.dropna()
    ax.hist(data, bins=bins, color=RED, edgecolor=BLACK, alpha=0.9)
    if logx:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Частота")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    save(fname)


# ============================================================
# 1. ЗАГРУЗКА И ПАРСИНГ YML
# ============================================================
print("=" * 80)
print("1. ПАРСИНГ YML-КАТАЛОГА")
print("=" * 80)

with open(PATH_SKUS, "rb") as f:
    raw = pickle.load(f)

shop = raw["yml_catalog"]["shop"]
print("Дата выгрузки:", raw["yml_catalog"].get("@date"))
print("Ключи shop:", list(shop.keys()))

# ---------- categories ----------
cats = shop["categories"]["category"]
if isinstance(cats, dict):
    cats = [cats]

cat_df = pd.DataFrame([
    {
        "category_id":   int(c["@id"]),
        "parent_id":     int(c["@parentId"]) if "@parentId" in c else pd.NA,
        "category_name": c.get("#text"),
    }
    for c in cats
])
cat_df["parent_id"] = cat_df["parent_id"].astype("Int64")
print("cat_df:", cat_df.shape)

# ---------- offers ----------
offers = shop["offers"]["offer"]
if isinstance(offers, dict):
    offers = [offers]
print("offers:", len(offers))
# ---------- устойчивый парсер param ----------
KV_PATTERN = re.compile(r"^\s*(?P<name>[^:—\-]{2,60}?)\s*[:—\-]\s*(?P<value>.+?)\s*$")

def parse_param(p, sid, cat):
    """Умеет dict / str 'Ключ: значение' / None. Невалидное → bad_rows."""
    if isinstance(p, dict):
        name = p.get("@name")
        if name is None:
            bad_rows.append({"sku_id": sid, "reason": "dict без @name",
                             "raw": repr(p)[:200]})
            return []
        value = p.get("#text")
        return [{
            "sku_id":     sid,
            "categoryId": cat,
            "attr":       str(name),
            "value":      None if value is None else str(value),
            "unit":       p.get("@unit"),
            "has_unit":   "@unit" in p,
            "is_search":  str(name).endswith("_search"),
        }]

    if isinstance(p, str):
        m = KV_PATTERN.match(p)
        if m:
            name  = m.group("name").strip()
            value = m.group("value").strip()
            return [{
                "sku_id":     sid,
                "categoryId": cat,
                "attr":       name,
                "value":      value,
                "unit":       None,
                "has_unit":   False,
                "is_search":  name.endswith("_search"),
            }]
        bad_rows.append({"sku_id": sid, "reason": "str без разделителя",
                         "raw": p[:200]})
        return []

    bad_rows.append({"sku_id": sid, "reason": f"param[] is {type(p).__name__}",
                     "raw": repr(p)[:200]})
    return []


meta_rows, attr_rows, bad_rows = [], [], []

for o in offers:
    sid = int(o["@id"])
    cat = int(o["categories"]["categoryId"]) if o.get("categories") else pd.NA

    params = o.get("param", [])
    # param может быть чем угодно — приводим к списку
    if params is None:
        params = []
    elif isinstance(params, (dict, str)):
        params = [params]
    elif not isinstance(params, list):
        bad_rows.append({"sku_id": sid,
                         "reason": f"param is {type(params).__name__}",
                         "raw": repr(params)[:200]})
        params = []

    meta_rows.append({
        "sku_id":     sid,
        "available":  str(o.get("@available", "")).lower() == "true",
        "vendor":     o.get("vendor"),
        "model":      o.get("model"),
        "name":       o.get("name"),
        "categoryId": cat,
        "n_params":   len(params),
    })

    for p in params:
        attr_rows.extend(parse_param(p, sid, cat))

print("bad_rows:", len(bad_rows))

skus_meta = pd.DataFrame(meta_rows)
skus_meta["categoryId"] = skus_meta["categoryId"].astype("Int64")
skus_meta["n_params"]   = skus_meta["n_params"].astype("int32")

attrs = pd.DataFrame(attr_rows)
attrs["categoryId"] = attrs["categoryId"].astype("Int64")
attrs["has_unit"]   = attrs["has_unit"].astype(bool)
attrs["is_search"]  = attrs["is_search"].astype(bool)
attrs["attr"]       = attrs["attr"].astype("category")

print("skus_meta:", skus_meta.shape)
print("attrs:    ", attrs.shape)

# ---------- сохраняем ----------
cat_df.to_parquet("cat_df.parquet")
skus_meta.to_parquet("skus_meta.parquet")
attrs.to_parquet("attrs.parquet")
print("[saved] cat_df.parquet / skus_meta.parquet / attrs.parquet")


# ============================================================
# 2. EDA: КАТЕГОРИИ (cat_df)
# ============================================================
# print("\n" + "=" * 80)
# print("2. EDA: КАТЕГОРИИ")
# print("=" * 80)
#
# # 2.1 глубина дерева
# parent_map = dict(zip(cat_df["category_id"], cat_df["parent_id"]))
#
# def depth(cid, seen=None):
#     seen = seen or set()
#     d, cur = 0, cid
#     while pd.notna(parent_map.get(cur)) and cur not in seen:
#         seen.add(cur); cur = parent_map[cur]; d += 1
#         if d > 20:
#             return -1
#     return d
#
# cat_df["depth"] = cat_df["category_id"].apply(depth)
# print("Глубина дерева категорий:")
# print(cat_df["depth"].value_counts().sort_index())
#
# # 2.2 битые parent_id
# valid_ids = set(cat_df["category_id"])
# broken = cat_df[~cat_df["parent_id"].isna() & ~cat_df["parent_id"].isin(valid_ids)]
# print(f"\nБитых parent_id: {len(broken)}")
#
# # 2.3 сколько SKU в каждой категории
# used = skus_meta["categoryId"].dropna().astype(int).value_counts()
# cat_df["n_skus"] = cat_df["category_id"].map(used).fillna(0).astype(int)
#
# print("\nРаспределение SKU по категориям:")
# print(cat_df["n_skus"].describe(percentiles=[.5, .9, .99]))
# print("Категорий без SKU:", (cat_df["n_skus"] == 0).sum())
# print("Категорий с 1 SKU:", (cat_df["n_skus"] == 1).sum())
# print("Категорий с <5 SKU:", (cat_df["n_skus"] < 5).sum())
#
# # 2.4 SKU в неизвестных категориях
# orphan_ids = set(skus_meta["categoryId"].dropna().astype(int)) - valid_ids
# n_orphan = skus_meta["categoryId"].isin(orphan_ids).sum()
# print(f"SKU в неизвестных категориях: {n_orphan:,}")
#
# # 2.5 графики
# hist_series(cat_df["depth"], "Глубина дерева категорий",
#             "Уровень", "01_cat_depth.png", bins=10)
#
# hist_series(cat_df["n_skus"].replace(0, np.nan).dropna(),
#             "Число SKU в категории (log-scale)",
#             "SKU", "02_cat_skus_hist.png", bins=60, logx=True)
#
# top_cat = cat_df.sort_values("n_skus", ascending=False).head(20) \
#                 .set_index("category_name")["n_skus"]
# bar_horizontal(top_cat, "Топ-20 категорий по числу SKU",
#                "SKU", "03_cat_top20.png")
#
# # 2.6 дубли имён
# dup_names = cat_df["category_name"].str.strip().str.lower().value_counts()
# dup_names = dup_names[dup_names > 1]
# print(f"\nДублирующихся имён категорий: {len(dup_names)}")
# if len(dup_names) > 0:
#     bar_horizontal(dup_names.head(20),
#                    "Топ-20 дублирующихся имён категорий",
#                    "Количество", "04_cat_dup_names.png")
#

# ============================================================
# 3. EDA: SKU (skus_meta)
# ============================================================
print("\n" + "=" * 80)
print("3. EDA: SKU")
print("=" * 80)

# # 3.1 пропуски
# miss = skus_meta.isna().sum()
# miss_pct = (miss / len(skus_meta) * 100).round(2)
# print("Пропуски:")
# for c in skus_meta.columns:
#     print(f"  {c:12s}: {miss[c]:>10,} ({miss_pct[c]:>6.2f}%)")
#
# # 3.2 дубли
# print("\nДубли sku_id:", skus_meta["sku_id"].duplicated().sum())
#
# # 3.3 n_params
# print("\nn_params:")
# print(skus_meta["n_params"].describe(percentiles=[.05, .25, .5, .75, .95, .99]))
# print("SKU без param:   ", (skus_meta["n_params"] == 0).sum())
# print("SKU с <3 attr:   ", (skus_meta["n_params"] < 3).sum())
# print("SKU с >100 attr: ", (skus_meta["n_params"] > 100).sum())
#
# hist_series(skus_meta["n_params"], "Число атрибутов на SKU",
#             "Атрибутов", "05_sku_nparams.png", bins=60)
#
# # 3.4 доступность
# print("\nДоступность:")
# print(skus_meta["available"].value_counts(normalize=True).round(4))
# print(skus_meta.groupby("available")["n_params"]
#       .agg(["mean", "median", "count"]).round(2))
#
# avail = skus_meta["available"].value_counts()
# bar_vertical(avail, "Доступность SKU", "available", "SKU",
#              rotation=0, fname="06_sku_available.png")
#
# # 3.5 бренды
# print("\nУникальных vendor:", skus_meta["vendor"].nunique())
# skus_meta["vendor_norm"] = skus_meta["vendor"].str.strip().str.lower()
# print("После нормализации:", skus_meta["vendor_norm"].nunique())
#
# top_vendors = skus_meta["vendor"].value_counts().head(25)
# bar_horizontal(top_vendors, "Топ-25 брендов по числу SKU",
#                "SKU", "07_sku_top_vendors.png")
#
# vc = skus_meta["vendor_norm"].value_counts()
# print("Брендов с 1 SKU:", (vc == 1).sum(),
#       f"({vc[vc == 1].sum():,} SKU)")
#
# # 3.6 согласованность name / vendor / model
# def contains(name, token):
#     if pd.isna(name) or pd.isna(token):
#         return False
#     return str(token).lower() in str(name).lower()
#
# skus_meta["vendor_in_name"] = skus_meta.apply(
#     lambda r: contains(r["name"], r["vendor"]), axis=1)
# skus_meta["model_in_name"] = skus_meta.apply(
#     lambda r: contains(r["name"], r["model"]), axis=1)
#
# print("\nvendor в name:", round(skus_meta["vendor_in_name"].mean(), 4))
# print("model  в name:", round(skus_meta["model_in_name"].mean(), 4))
#
# consistency = pd.Series({
#     "vendor в name": skus_meta["vendor_in_name"].sum(),
#     "model в name":  skus_meta["model_in_name"].sum(),
# })
# bar_vertical(consistency, "Согласованность name с vendor / model",
#              "Признак", "SKU", rotation=0, fname="08_sku_consistency.png")
#
#
# # ============================================================
# # 4. EDA: АТРИБУТЫ (attrs)
# # ============================================================
# print("\n" + "=" * 80)
# print("4. EDA: АТРИБУТЫ")
# print("=" * 80)
#
# total_skus = len(skus_meta)
#
# # 4.1 длинный хвост
# print("Всего пар (sku, attr):", f"{len(attrs):,}")
# print("Уникальных attr:      ", attrs["attr"].nunique())
#
# attr_freq = attrs["attr"].value_counts()
# print("\nattr с частотой ≥10000:", (attr_freq >= 10000).sum())
# print("attr с частотой <10:   ", (attr_freq <  10).sum())
#
# bar_horizontal(attr_freq.head(30),
#                "Топ-30 атрибутов по частоте",
#                "Число пар (sku, attr)", "09_attr_top30.png")
#
# # 4.2 _search
# search_share = attrs["is_search"].value_counts(normalize=True)
# print("\nДоля _search:", search_share.round(4).to_dict())
#
# base = set(attrs.loc[~attrs["is_search"], "attr"].astype(str))
# srch = set(attrs.loc[ attrs["is_search"], "attr"]
#            .astype(str).str.replace("_search", "", regex=False))
# print("Только без _search:", len(base - srch))
# print("Только с _search:  ", len(srch - base))
# print("В обоих:           ", len(base & srch))
#
# # 4.3 coverage
# coverage = (attrs.groupby("attr", observed=True)["sku_id"].nunique()
#             / total_skus).sort_values(ascending=False)
# print("\nTop-10 по coverage:")
# print((coverage.head(10) * 100).round(2))
#
# bar_horizontal(coverage.head(25) * 100,
#                "Топ-25 атрибутов по coverage, %",
#                "% SKU", "10_attr_coverage.png", label_fmt="%.1f%%")
#
# # 4.4 кардинальность
# card = (attrs.groupby("attr", observed=True)["value"].nunique()
#         .sort_values(ascending=False))
# print("\nTop-15 по числу уникальных значений:")
# print(card.head(15))
#
# bar_horizontal(card.head(25),
#                "Топ-25 атрибутов по числу уникальных значений",
#                "Уникальных значений", "11_attr_cardinality.png")
#
# # 4.5 enum-score
# def top10_share(g):
#     return g["value"].value_counts(normalize=True).head(10).sum()
#
# enum = (attrs.groupby("attr", observed=True)
#         .apply(top10_share, include_groups=False)
#         .sort_values(ascending=False))
#
# print("\nКандидаты в словари (top-10 ≥ 0.8):",
#       (enum >= 0.8).sum())
# print("Свободный текст (top-10 < 0.3):   ", (enum < 0.3).sum())
#
# enum_plot = enum.head(30) * 100
# bar_horizontal(enum_plot,
#                "Топ-30 атрибутов по концентрации значений (top-10 share, %)",
#                "% значений в топ-10", "12_attr_enum.png", label_fmt="%.1f%%")
#
# # 4.6 единицы измерения
# unit_share = attrs["has_unit"].mean()
# print(f"\nДоля записей с @unit: {unit_share:.4f}")
# units = attrs["unit"].value_counts(dropna=False)
# print(units.head(20))
#
# if units.dropna().shape[0] > 0:
#     bar_vertical(units.dropna().head(20),
#                  "Топ-20 единиц измерения (@unit)",
#                  "unit", "Записей",
#                  fname="13_attr_units.png", rotation=45)
#
# # один attr — разные unit
# units_per_attr = (attrs.dropna(subset=["unit"])
#                   .groupby("attr", observed=True)["unit"].nunique())
# multi = units_per_attr[units_per_attr > 1].sort_values(ascending=False)
# print("\nАтрибутов с >1 разными @unit:", len(multi))
# if len(multi) > 0:
#     print(multi.head(15))
#
# # 4.7 дубли одного attr в одном SKU
# dup = attrs.groupby(["sku_id", "attr"], observed=True).size()
# n_dup = (dup > 1).sum()
# print(f"\nПар (sku, attr) с >1 значением: {n_dup:,} "
#       f"({n_dup / len(dup) * 100:.2f}%)")
#
# # 4.8 согласованность attr vs attr_search
# pairs = attrs.copy()
# pairs["attr_base"] = (pairs["attr"].astype(str)
#                       .str.replace("_search", "", regex=False))
#
# wide = (pairs.pivot_table(index=["sku_id", "attr_base"],
#                           columns="is_search",
#                           values="value",
#                           aggfunc="first")
#         .dropna())
# mismatch = wide[wide[False].astype(str) != wide[True].astype(str)]
# print(f"\nРасхождений attr vs attr_search: "
#       f"{len(mismatch):,} из {len(wide):,} "
#       f"({len(mismatch) / max(len(wide), 1) * 100:.2f}%)")
#
# # 4.9 профиль топ-категорий
# top5_cats = skus_meta["categoryId"].value_counts().head(5).index.tolist()
# fig, axes = plt.subplots(1, 5, figsize=(22, 6))
# for ax, cat in zip(axes, top5_cats):
#     sub = attrs[attrs["categoryId"] == cat]
#     top = sub["attr"].value_counts().head(10).sort_values()
#     cat_name = cat_df.loc[cat_df["category_id"] == cat, "category_name"]
#     cat_name = cat_name.iloc[0] if len(cat_name) else str(cat)
#     bars = ax.barh(top.index.astype(str), top.values,
#                    color=RED, edgecolor=BLACK, linewidth=1.0)
#     ax.set_title(f"{cat_name}\n({sub['sku_id'].nunique():,} SKU)", fontsize=10)
#     ax.bar_label(bars, fmt="%d", padding=2, color=BLACK, fontsize=8)
#     ax.grid(axis="x", linestyle="--", alpha=0.3)
# save("14_attr_top_cats.png")
#
# # 4.10 атрибутов на категорию
# cat_attr_counts = attrs.groupby("categoryId", observed=True)["attr"].nunique()
# print("\nАтрибутов на категорию:")
# print(cat_attr_counts.describe(percentiles=[.5, .9, .99]))
# print("Категорий с <3 атрибутами:", (cat_attr_counts < 3).sum())
#
# hist_series(cat_attr_counts, "Число уникальных атрибутов в категории",
#             "Атрибутов", "15_cat_attr_counts.png", bins=50)
#
#
# # ============================================================
# # 5. СВОДНАЯ ТАБЛИЦА МЕТРИК
# # ============================================================
# print("\n" + "=" * 80)
# print("5. СВОДНАЯ ТАБЛИЦА")
# print("=" * 80)
#
# summary = pd.DataFrame([
#     ("Дата выгрузки",                raw["yml_catalog"].get("@date"), "", ""),
#     ("Всего категорий",              len(cat_df), "", ""),
#     ("Корневых категорий",           cat_df["parent_id"].isna().sum(), "", ""),
#     ("Максимальная глубина дерева",  cat_df["depth"].max(), "", ""),
#     ("Битых parent_id",              len(broken), "", ""),
#     ("Категорий без SKU",            (cat_df["n_skus"] == 0).sum(),
#      f"{(cat_df['n_skus'] == 0).mean() * 100:.1f}%", ""),
#     ("Категорий с <5 SKU",           (cat_df["n_skus"] < 5).sum(),
#      f"{(cat_df['n_skus'] < 5).mean() * 100:.1f}%", "длинный хвост"),
#     ("Всего SKU",                    len(skus_meta), "", ""),
#     ("Дубли sku_id",                 skus_meta["sku_id"].duplicated().sum(), "", ""),
#     ("SKU без param",                (skus_meta["n_params"] == 0).sum(),
#      f"{(skus_meta['n_params'] == 0).mean() * 100:.1f}%", ""),
#     ("SKU с <3 атрибутами",          (skus_meta["n_params"] < 3).sum(),
#      f"{(skus_meta['n_params'] < 3).mean() * 100:.1f}%", ""),
#     ("Медиана атрибутов на SKU",     int(skus_meta["n_params"].median()), "", ""),
#     ("Доступных SKU",                skus_meta["available"].sum(),
#      f"{skus_meta['available'].mean() * 100:.1f}%", ""),
#     ("Уникальных vendor",            skus_meta["vendor"].nunique(), "", ""),
#     ("Брендов с 1 SKU",              (vc == 1).sum(), "", "длинный хвост"),
#     ("vendor в name",                round(skus_meta["vendor_in_name"].mean(), 3), "", ""),
#     ("model в name",                 round(skus_meta["model_in_name"].mean(), 3), "", ""),
#     ("Всего (sku, attr)",            len(attrs), "", ""),
#     ("Уникальных attr",              attrs["attr"].nunique(), "", ""),
#     ("attr с частотой <10",          (attr_freq < 10).sum(),
#      f"{(attr_freq < 10).mean() * 100:.1f}%", "длинный хвост"),
#     ("Доля _search",                 round(attrs["is_search"].mean(), 3), "", ""),
#     ("Доля с @unit",                 round(attrs["has_unit"].mean(), 3), "", ""),
#     ("Атрибутов с >1 @unit",         len(multi), "", ""),
#     ("Пар (sku, attr) с >1 значением", n_dup,
#      f"{n_dup / len(dup) * 100:.2f}%", "дубли в SKU"),
#     ("Расхождений attr vs _search",  len(mismatch),
#      f"{len(mismatch) / max(len(wide), 1) * 100:.2f}%", ""),
#     ("Кандидатов в словари (≥0.8)",  (enum >= 0.8).sum(), "", ""),
#     ("Свободный текст (<0.3)",       (enum < 0.3).sum(), "", "нужен ML/LLM"),
# ], columns=["metric", "value", "share", "comment"])
#
# print(summary.to_string(index=False))
# summary.to_csv("summary_metrics.csv", index=False)
# print("\n[saved] summary_metrics.csv")
#
#
# # ============================================================
# # 6. JOIN С catalog.parquet (если есть)
# # ============================================================
# if os.path.exists(PATH_CATALOG):
#     print("\n" + "=" * 80)
#     print("6. JOIN С catalog.parquet")
#     print("=" * 80)
#
#     catalog = pd.read_parquet(PATH_CATALOG)
#     catalog["sku_id"] = catalog["sku_id"].astype("int64")
#     merged = skus_meta.merge(catalog, on="sku_id", how="inner")
#     print("Совпало SKU:", len(merged), "из", len(skus_meta))
#
#     # «слепая зона»: атрибут есть в param, но не в тексте
#     catalog_text = catalog.set_index("sku_id")["title"].fillna("") + " " + \
#                    catalog.set_index("sku_id")["description"].fillna("")
#
#     # 1. ограничь атрибуты — только топ-100 из _search
#     top_attrs = (attrs[attrs["is_search"]]
#                  .groupby("attr", observed=True).size()
#                  .nlargest(100).index)
#     gold_attr = attrs[attrs["attr"].isin(top_attrs) & attrs["value"].notna()].copy()
#     print(f"Строк после фильтра: {len(gold_attr):,}")  # ~500k вместо 20M
#
#     # 2. собери текст в один Series
#     catalog_idx = catalog.set_index("sku_id")
#     catalog_text = (catalog_idx["title"].fillna("") + " " +
#                     catalog_idx["description"].fillna("")).str.lower()
#
#     # 3. join, а не apply
#     gold_attr = gold_attr.merge(
#         catalog_text.rename("text"),
#         left_on="sku_id", right_index=True, how="left"
#     )
#     gold_attr["text"] = gold_attr["text"].fillna("")
#     gold_attr["in_text"] = [
#         v.lower() in t for v, t in zip(gold_attr["value"], gold_attr["text"])
#     ]
#
#     blind = gold_attr.groupby("attr", observed=True)["in_text"].mean() \
#                      .sort_values()
#     print("\nАтрибуты с наименьшей встречаемостью в тексте (слепая зона):")
#     print((blind.head(20) * 100).round(1))
#
#     bar_horizontal(blind.head(25) * 100,
#                    "«Слепая зона»: атрибуты, редко встречающиеся в тексте, %",
#                    "% найденных в title+description",
#                    "16_blind_zone.png", label_fmt="%.1f%%")
#

# ============================================================
# 7. SEARCH LOG (query_clicks.parquet)
# ============================================================
PATH_SEARCH = "../data/query_clicks.parquet"

if os.path.exists(PATH_SEARCH):
    print("\n" + "=" * 80)
    print("7. SEARCH LOG")
    print("=" * 80)

    # 7.0 — что реально лежит в файле
    print("Колонки в файле:", pq.read_schema(PATH_SEARCH).names)

    traffic = read_parquet_clean(
        PATH_SEARCH,
        columns=["sku_id", "query_text", "sku_position",
                 "sku_price", "sku_subject_id"]
    )
    traffic["sku_id"] = traffic["sku_id"].astype("int64")
    traffic["query_text"] = traffic["query_text"].str.strip().str.lower()

    print(f"\nПоказов: {len(traffic):,}")
    print(f"Уникальных SKU: {traffic['sku_id'].nunique():,}")
    print(f"Уникальных запросов: {traffic['query_text'].nunique():,}")

    # ---------- 7.1 трафик по SKU ----------
    shows_per_sku = traffic.groupby("sku_id").size().rename("shows")
    print("\nПоказов на SKU:")
    print(shows_per_sku.describe(percentiles=[.5, .9, .99]))

    hist_series(shows_per_sku, "Показов на SKU (log-scale)",
                "Показов", "18_traffic_per_sku.png", bins=60, logx=True)

    # ---------- 7.2 топ запросов ----------
    top_q = traffic["query_text"].value_counts().head(25)
    print("\nТоп-25 запросов:")
    print(top_q)
    bar_horizontal(top_q, "Топ-25 поисковых запросов",
                   "Показов", "19_top_queries.png")

    # ---------- 7.3 топ категорий по трафику ----------
    top_cat_traffic = (traffic.groupby("sku_subject_id").size()
                       .sort_values(ascending=False).head(25))
    print("\nТоп-25 категорий по трафику:")
    print(top_cat_traffic)
    bar_horizontal(top_cat_traffic, "Топ-25 категорий по трафику",
                   "Показов", "20_top_categories_traffic.png")

    # ---------- 7.4 traffic-weighted coverage по атрибутам ----------
    # ---------- 7.4 traffic-weighted coverage по атрибутам ----------
    # Правильный расчёт: для каждого атрибута берём МНОЖЕСТВО SKU,
    # у которых он заполнен, и суммируем показы этих SKU.
    # Каждый SKU считается один раз на атрибут — без дублирования.

    total_shows = shows_per_sku.sum()

    # Множество SKU для каждого атрибута (только с непустым значением)
    attr_to_skus = (
        attrs[attrs["value"].notna()]
        .groupby("attr", observed=True)["sku_id"]
        .apply(set)
    )

    # Показы только по SKU, которые есть в search-логе
    shows_index = shows_per_sku.index

    tw = {}
    for attr, skus in attr_to_skus.items():
        # пересекаем с SKU, у которых есть показы
        skus_with_shows = shows_index.intersection(skus)
        tw[attr] = shows_per_sku.loc[skus_with_shows].sum() / total_shows

    tw = pd.Series(tw).sort_values(ascending=False)

    print("\nТоп-30 атрибутов по traffic-weighted coverage:")
    print((tw.head(30) * 100).round(2))

    # Проверка: все значения должны быть в [0, 1]
    assert tw.max() <= 1.0, f"BUG: tw.max() = {tw.max()}"
    assert tw.min() >= 0.0, f"BUG: tw.min() = {tw.min()}"
    print(f"\nПроверка: tw в диапазоне [{tw.min():.3f}, {tw.max():.3f}] — OK")

    # ---------- сервисные vs продуктовые ----------
    SERVICE_ATTRS = {
        "Подборки", "Топ коллекция", "Страна_search", "Продавец",
        "Гарантия предоставляется_search", "Гарантия_search",
        "Потратить бонусы", "Акции", "Состояние_search",
        "Срок службы_search",
    }

    top30 = tw.head(30)
    service_attrs_in_top30 = top30[top30.index.isin(SERVICE_ATTRS)]
    product_attrs_in_top30 = top30[~top30.index.isin(SERVICE_ATTRS)]

    print(f"\nСервисных в топ-30:   {len(service_attrs_in_top30)}")
    print(f"Продуктовых в топ-30: {len(product_attrs_in_top30)}")
    print(f"\nМедиана покрытия сервисных:   {service_attrs_in_top30.median():.1%}")
    print(f"Медиана покрытия продуктовых: {product_attrs_in_top30.median():.1%}")

    # ---------- график с раскраской сервисные/продуктовые ----------
    colors = [
        "#9E9E9E" if attr in SERVICE_ATTRS else RED
        for attr in top30.index
    ]

    fig, ax = plt.subplots(figsize=(11, 9))
    bars = ax.barh(top30.index[::-1], (top30.values * 100)[::-1],
                   color=colors[::-1], edgecolor=BLACK, linewidth=1.0)
    ax.set_xlabel("% показов каталога")
    ax.set_title(
        f"Топ-30 атрибутов по traffic-weighted coverage\n"
        f"серые — сервисные ({len(service_attrs_in_top30)}), "
        f"красные — продуктовые ({len(product_attrs_in_top30)})",
        fontsize=12, fontweight="bold"
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3, color=BLACK, fontsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    save("21_traffic_weights.png")

    # ---------- 7.5 цена ----------
    print("\nЦена:")
    print(traffic["sku_price"].describe(percentiles=[.05, .5, .95, .99]))
    print("Нулевых цен:", (traffic["sku_price"] == 0).sum())

    # ---------- 7.6 длина запроса ----------
    traffic["q_words"] = traffic["query_text"].str.split().str.len()
    print("\nСлов в запросе:")
    print(traffic["q_words"].describe())
    hist_series(traffic["q_words"], "Слов в поисковом запросе",
                "Слов", "22_query_len.png", bins=20)


print("\n" + "=" * 80)
print("ГОТОВО. Все графики в:", PLOTS)
print("Все таблицы: cat_df.parquet, skus_meta.parquet, attrs.parquet,")
print("             summary_metrics.csv")
print("=" * 80)


