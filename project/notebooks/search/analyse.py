import pandas as pd

df = pd.read_parquet("sku_desc.parquet")

print(df.shape)                      # Количество строк и столбцов
print(df.head())                     # Первые 5 строк
df.info()                           # Столбцы и типы данных
print(df.isna().sum())               # Пропуски по столбцам
print(df.describe(include="all"))    # Сводная статистика

# При необходимости сохранить как CSV
#df.to_csv("data.csv", index=False, encoding="utf-8-sig")