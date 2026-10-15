
import pandas as pd
import pickle
# Чтение pkl

import pandas as pd

raw = pd.read_pickle("../data/skus.pkl")
shop = raw["yml_catalog"]["shop"]

print("Ключи shop:", list(shop.keys()))
print("Дата выгрузки:", raw["yml_catalog"].get("@date"))

offers = shop["offers"]["offer"]
if isinstance(offers, dict):
    offers = [offers]
print("Всего offers:", len(offers))
print("Ключи первого offer:", list(offers[0].keys()))
print("Пример offer:", {k: offers[0][k] for k in list(offers[0].keys())[:10]})

cats = shop["categories"]["category"]
if isinstance(cats, dict):
    cats = [cats]
print("Всего категорий:", len(cats))
print("Пример категории:", cats[0])