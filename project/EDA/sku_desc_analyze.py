import pandas as pd
import numpy as np
import re
from collections import Counter

# Чтобы видеть длинные строки
pd.set_option("display.max_colwidth", 300)
pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)

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

    # Слишком короткие / длинные
    print(f"Короче 20 символов:  {(lengths < 20).sum():,} ({(lengths < 20).mean()*100:.2f}%)")
    print(f"Длиннее 5000:        {(lengths > 5000).sum():,} ({(lengths > 5000).mean()*100:.2f}%)")


print("\n" + "=" * 80)
print("3. HTML-МУСОР И СПЕЦСИМВОЛЫ")
print("=" * 80)

desc = catalog["description"].fillna("")

# HTML-теги
html_mask = desc.str.contains(r"<[^>]+>", regex=True, na=False)
print(f"С HTML-тегами:      {html_mask.sum():,} ({html_mask.mean()*100:.2f}%)")

# Конкретные теги
for tag in ["<br>", "<b>", "<i>", "<ul>", "<li>", "<p>", "<div>", "&nbsp;"]:
    cnt = desc.str.contains(re.escape(tag), na=False).sum()
    if cnt > 0:
        print(f"  {tag:10s}: {cnt:>10,}")

# Переносы строк
print(f"С \\n:               {desc.str.contains(chr(10), regex=False).sum():,}")
print(f"С \\r\\n:             {desc.str.contains(chr(13), regex=False).sum():,}")

# Эмодзи / необычные символы
emoji_mask = desc.str.contains(r"[\U0001F300-\U0001FAFF]", regex=True, na=False)
print(f"С эмодзи:           {emoji_mask.sum():,}")

# Латиница vs кириллица (определить язык)
cyr_mask = desc.str.contains(r"[а-яА-Я]", regex=True, na=False)
lat_mask = desc.str.contains(r"[a-zA-Z]", regex=True, na=False)
print(f"С кириллицей:       {cyr_mask.sum():,} ({cyr_mask.mean()*100:.2f}%)")
print(f"С латиницей:        {lat_mask.sum():,} ({lat_mask.mean()*100:.2f}%)")


print("\n" + "=" * 80)
print("4. АНАЛИЗ TITLE")
print("=" * 80)

titles = catalog["title"].fillna("")

# Первое слово — часто бренд или тип товара
first_word = titles.str.split().str[0]
print("Топ-20 первых слов в title:")
print(first_word.value_counts().head(20))

# Последнее слово
last_word = titles.str.split().str[-1]
print("\nТоп-20 последних слов:")
print(last_word.value_counts().head(20))


print("\n" + "=" * 80)
print("5. ЧИСЛОВЫЕ ПАТТЕРНЫ В DESCRIPTION")
print("=" * 80)

# Частые единицы измерения
units = ["мм", "см", "м", "кг", "г", "л", "мл", "Вт", "В", "ГБ", "ТБ", "МБ", "дюйм", "°C"]
for u in units:
    pattern = rf"\d+\s*{re.escape(u)}\b"
    cnt = desc.str.contains(pattern, regex=True, na=False).sum()
    if cnt > 0:
        print(f"{u:8s}: {cnt:>10,} ({cnt/len(desc)*100:.2f}%)")