import pandas as pd

#Чтение parguest
df = pd.read_parquet("../data/sku_desc.parquet")

# 1. Общая информация: типы, сколько non-null, память
print("=== INFO ===")
df.info()

# 2. Первые 5 строк без обрезки — покажет все колонки полностью
print("\n=== HEAD (полностью) ===")
with pd.option_context("display.max_columns", None,
                       "display.width", None,
                       "display.max_colwidth", 200):   # ширина ячейки
    print(df.head())

# 3. Случайные примеры — часто полезнее, чем первые строки
print("\n=== SAMPLE (5 случайных) ===")
with pd.option_context("display.max_columns", None,
                       "display.width", None,
                       "display.max_colwidth", 200):
    print(df.sample(5, random_state=42))

# 4. Типы + названия колонок
print("\n=== DTYPES ===")
print(df.dtypes)

print("\n=== COLUMNS ===")
print(df.columns.tolist())

# 5. Размер таблицы
print("\n=== SHAPE ===")
print(df.shape)