import pandas as pd
import numpy as np
import re
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Чтобы видеть длинные строки
pd.set_option("display.max_colwidth", 300)
pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)

# ============================================================
# ЕДИНЫЙ КРАСНО-ЧЁРНЫЙ СТИЛЬ
# ============================================================
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

# Раскомментируй, если хочешь тёмную тему:
# plt.style.use("dark_background")
# WHITE, BLACK = BLACK, WHITE   # поменять местами

import os
os.makedirs("plots", exist_ok=True)

# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================
catalog = pd.read_parquet("../data/sku_desc.parquet")
print("Загружено:", catalog.shape)

print("=" * 80)
print("1. ОБЩАЯ ИНФОРМАЦИЯ")
print("=" * 80)

print(f"Всего карточек: {len(catalog):,}")
print(f"Колонки: {catalog.columns.tolist()}")
print(f"Типы:\n{catalog.dtypes}")
print(f"Память: {catalog.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# Пропуски
print("\n--- Пропуски ---")
missing = catalog.isna().sum()
missing_pct = (missing / len(catalog) * 100).round(2)
for col in catalog.columns:
    print(f"{col:15s}: {missing[col]:>10,} ({missing_pct[col]:>6.2f}%)")

# Дубликаты
print(f"\nДубликатов по sku_id: {catalog['sku_id'].duplicated().sum():,}")
print(f"Уникальных sku_id:    {catalog['sku_id'].nunique():,}")
print(f"Дубликатов по title:  {catalog['title'].duplicated().sum():,}")
print(f"Дубликатов по description: {catalog['description'].duplicated().sum():,}")


# ============================================================
# ГРАФИК 1.1: ПРОПУСКИ ПО КОЛОНКАМ
# ============================================================
missing_plot = missing_pct[missing_pct > 0].sort_values(ascending=False)
if len(missing_plot) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(missing_plot.index, missing_plot.values,
                  color=RED, edgecolor=BLACK, linewidth=1.2)
    ax.set_title("Пропуски по колонкам, %")
    ax.set_ylabel("% пропусков")
    ax.bar_label(bars, fmt="%.2f%%", padding=3, color=BLACK)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("plots/01_missing.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("Пропусков нет — график не строится.")


# ============================================================
# ГРАФИК 1.2: ДУБЛИКАТЫ И ОБЩИЕ МЕТРИКИ
# ============================================================
dup_metrics = pd.Series({
    "Всего карточек":        len(catalog),
    "Уникальных sku_id":     catalog["sku_id"].nunique(),
    "Дубликатов sku_id":     catalog["sku_id"].duplicated().sum(),
    "Дубликатов title":      catalog["title"].duplicated().sum(),
    "Дубликатов description": catalog["description"].duplicated().sum(),
})

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(dup_metrics.index, dup_metrics.values,
              color=RED, edgecolor=BLACK, linewidth=1.2)
ax.set_title("Общие метрики и дубликаты")
ax.set_ylabel("Количество")
ax.bar_label(bars, fmt="%d", padding=3, color=BLACK)
ax.grid(axis="y", linestyle="--", alpha=0.3)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("plots/02_duplicates.png", dpi=150, bbox_inches="tight")
plt.show()


print("\n" + "=" * 80)
print("2. ДЛИНЫ ТЕКСТОВ")
print("=" * 80)

for col in ["title", "description"]:
    lengths = catalog[col].dropna().str.len()
    print(f"\n--- {col} ---")
    print(f"Пустых (NaN):    {catalog[col].isna().sum():,}")
    print(f"Пустых строк:    {(catalog[col] == '').sum():,}")
    print(f"min:             {lengths.min():,}")
    print(f"25%:             {lengths.quantile(0.25):,.0f}")
    print(f"median:          {lengths.median():,.0f}")
    print(f"75%:             {lengths.quantile(0.75):,.0f}")
    print(f"95%:             {lengths.quantile(0.95):,.0f}")
    print(f"99%:             {lengths.quantile(0.99):,.0f}")
    print(f"max:             {lengths.max():,}")
    print(f"mean:            {lengths.mean():,.0f}")

    print(f"Короче 20 символов:  {(lengths < 20).sum():,} ({(lengths < 20).mean()*100:.2f}%)")
    print(f"Длиннее 5000:        {(lengths > 5000).sum():,} ({(lengths > 5000).mean()*100:.2f}%)")


# ============================================================
# ГРАФИК 2.1: ГИСТОГРАММЫ ДЛИН
# ============================================================
length_stats = {}
for col in ["title", "description"]:
    length_stats[col] = catalog[col].dropna().str.len()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col in zip(axes, ["title", "description"]):
    lengths = length_stats[col]
    upper = lengths.quantile(0.99)
    ax.hist(lengths[lengths <= upper], bins=50,
            color=RED, edgecolor=BLACK, alpha=0.9)
    ax.set_title(f"Длина: {col} (до 99-го перцентиля)")
    ax.set_xlabel("Символов")
    ax.set_ylabel("Частота")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("plots/03_length_hist.png", dpi=150, bbox_inches="tight")
plt.show()


# ============================================================
# ГРАФИК 2.2: BOXPLOT ДЛИН
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
bp = ax.boxplot(
    [length_stats["title"], length_stats["description"]],
    labels=["title", "description"],
    patch_artist=True,
    boxprops=dict(facecolor=RED, edgecolor=BLACK, linewidth=1.2),
    medianprops=dict(color=BLACK, linewidth=2),
    whiskerprops=dict(color=BLACK, linewidth=1.2),
    capprops=dict(color=BLACK, linewidth=1.2),
    flierprops=dict(marker="o", markerfacecolor=RED,
                    markeredgecolor=BLACK, alpha=0.5)
)
ax.set_title("Boxplot длин текстов")
ax.set_ylabel("Символов")
ax.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("plots/04_length_box.png", dpi=150, bbox_inches="tight")
plt.show()


print("\n" + "=" * 80)
print("3. HTML-МУСОР И СПЕЦСИМВОЛЫ")
print("=" * 80)

desc = catalog["description"].fillna("")

html_mask = desc.str.contains(r"<[^>]+>", regex=True, na=False)
print(f"С HTML-тегами:      {html_mask.sum():,} ({html_mask.mean()*100:.2f}%)")

for tag in ["<br>", "<b>", "<i>", "<ul>", "<li>", "<p>", "<div>", "&nbsp;"]:
    cnt = desc.str.contains(re.escape(tag), na=False).sum()
    if cnt > 0:
        print(f"  {tag:10s}: {cnt:>10,}")

print(f"С \\n:               {desc.str.contains(chr(10), regex=False).sum():,}")
print(f"С \\r\\n:             {desc.str.contains(chr(13), regex=False).sum():,}")

emoji_mask = desc.str.contains(r"[\U0001F300-\U0001FAFF]", regex=True, na=False)
print(f"С эмодзи:           {emoji_mask.sum():,}")

cyr_mask = desc.str.contains(r"[а-яА-Я]", regex=True, na=False)
lat_mask = desc.str.contains(r"[a-zA-Z]", regex=True, na=False)
print(f"С кириллицей:       {cyr_mask.sum():,} ({cyr_mask.mean()*100:.2f}%)")
print(f"С латиницей:        {lat_mask.sum():,} ({lat_mask.mean()*100:.2f}%)")


# ============================================================
# ГРАФИК 3: СПЕЦСИМВОЛЫ И РАЗМЕТКА
# ============================================================
special_counts = {}
special_counts["HTML-теги"] = html_mask.sum()

for tag in ["<br>", "<b>", "<i>", "<ul>", "<li>", "<p>", "<div>", "&nbsp;"]:
    cnt = desc.str.contains(re.escape(tag), na=False).sum()
    if cnt > 0:
        special_counts[tag] = cnt

special_counts["\\n"]        = desc.str.contains(chr(10), regex=False).sum()
special_counts["\\r\\n"]     = desc.str.contains(chr(13), regex=False).sum()
special_counts["Эмодзи"]     = emoji_mask.sum()
special_counts["Кириллица"]  = cyr_mask.sum()
special_counts["Латиница"]   = lat_mask.sum()

special_series = pd.Series(special_counts).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(special_series.index, special_series.values,
               color=RED, edgecolor=BLACK, linewidth=1.2)
ax.set_title("Спецсимволы и разметка в description")
ax.set_xlabel("Количество карточек")
ax.bar_label(bars, fmt="%d", padding=3, color=BLACK)
ax.grid(axis="x", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("plots/05_special.png", dpi=150, bbox_inches="tight")
plt.show()


print("\n" + "=" * 80)
print("4. АНАЛИЗ TITLE")
print("=" * 80)

titles = catalog["title"].fillna("")

first_word = titles.str.split().str[0]
print("Топ-20 первых слов в title:")
print(first_word.value_counts().head(20))

last_word = titles.str.split().str[-1]
print("\nТоп-20 последних слов:")
print(last_word.value_counts().head(20))


# ============================================================
# ГРАФИК 4: ТОП ПЕРВЫХ / ПОСЛЕДНИХ СЛОВ
# ============================================================
top_first = first_word.value_counts().head(20)
top_last  = last_word.value_counts().head(20)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, data, title in zip(
    axes,
    [top_first, top_last],
    ["Топ-20 первых слов в title", "Топ-20 последних слов в title"]
):
    data_sorted = data.sort_values()
    bars = ax.barh(data_sorted.index.astype(str), data_sorted.values,
                   color=RED, edgecolor=BLACK, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Количество")
    ax.bar_label(bars, fmt="%d", padding=3, color=BLACK, fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("plots/06_title_words.png", dpi=150, bbox_inches="tight")
plt.show()


print("\n" + "=" * 80)
print("5. ЧИСЛОВЫЕ ПАТТЕРНЫ В DESCRIPTION")
print("=" * 80)

units = ["мм", "см", "м", "кг", "г", "л", "мл",
         "Вт", "В", "ГБ", "ТБ", "МБ", "дюйм", "°C"]

unit_counts = {}
for u in units:
    pattern = rf"\d+\s*{re.escape(u)}\b"
    cnt = desc.str.contains(pattern, regex=True, na=False).sum()
    if cnt > 0:
        print(f"{u:8s}: {cnt:>10,} ({cnt/len(desc)*100:.2f}%)")
        unit_counts[u] = cnt


# ============================================================
# ГРАФИК 5: ЕДИНИЦЫ ИЗМЕРЕНИЯ
# ============================================================
if unit_counts:
    unit_series = pd.Series(unit_counts).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(unit_series.index, unit_series.values,
                  color=RED, edgecolor=BLACK, linewidth=1.2)
    ax.set_title("Частота единиц измерения в description")
    ax.set_ylabel("Количество карточек")
    ax.bar_label(bars, fmt="%d", padding=3, color=BLACK)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("plots/07_units.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("Единицы измерения не найдены — график не строится.")

print("\nВсе графики сохранены в папку: plots/")