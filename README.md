## Здесь будет основное решение и инструкция запуска!

Скрипт `normalization.py` преобразует данные о товарах из файла `sku_desc.parquet` в JSON-файл `normalized.json`.

1. Перенесите файл `sku_desc.parquet` в корневую папку.

2. Установка зависимостей:

```bash
pip install pandas pyarrow
```

3. Запуск
```bash
python normalization.py
```



Пример:

```json
[
  {
    "@id": 400742273,
    "name": "Сушка для фруктов и овощей Centek CT-1662",
    "description": "Сушилка для овощей. Характеристики товара: Мощность: 650 Вт. Объем: 16 л.",
    "param": [
      {
        "id": 1,
        "@name": "Мощность",
        "#text": "650 Вт"
      },
      {
        "id": 2,
        "@name": "Объем",
        "#text": "16 л"
      }
    ]
  }
]
```