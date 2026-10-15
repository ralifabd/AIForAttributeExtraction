import pickle
import pandas as pd

# Чтение pkl

# with open("data/skus.pkl", "rb") as f:
#     obj = pickle.load(f)
#
# print("Тип:", type(obj))
#
# if hasattr(obj, "shape"):
#     print("Форма:", obj.shape)
#     print("Колонки:", obj.columns.tolist())
#     print(obj.head())
# elif isinstance(obj, dict):
#     print("Ключи:", list(obj.keys())[:20])
#     print("Пример:", {k: obj[k] for k in list(obj)[:3]})
# else:
#     print("Значение:", obj)


# Чтение parguest
# df = pd.read_parquet("data/sku_desc.parquet")
# print(df.head())       # первые 5 строк
# print(df.shape)        # (строк, столбцов)
# print(df.dtypes)       # типы колонок
# print(df.columns)      # названия колонок

# query_parguest
# df = pd.read_parquet("data/query_clicks.parquet")
# print(df.head())       # первые 5 строк
# print(df.shape)        # (строк, столбцов)
# print(df.dtypes)       # типы колонок
# print(df.columns)      # названия колонок


df = pd.read_parquet("data/query_clicks.parquet")

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