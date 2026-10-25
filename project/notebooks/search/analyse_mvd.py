#обходит все товары, собирает поля и характеристики, считает уникальные значения и заполненность по категориям
import csv
import json
import pickle
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


# УКАЖИТЕ ИМЯ ВАШЕГО PKL-ФАЙЛА:
BASE = Path(__file__).resolve().parent
SOURCE = BASE / "skus.pkl"

OUT = BASE / "catalog_analysis"
DB = OUT / "catalog.sqlite"


def as_list(value):
    """Одиночный элемент превращает в список."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def text(value):
    """Приводит значение к строке без потери вложенных структур."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def get_category_id(value):
    if isinstance(value, dict):
        value = value.get("#text")
    return text(value)


def main():
    if not SOURCE.is_file():
        raise SystemExit(f"Не найден исходный файл: {SOURCE}")

    if DB.exists():
        raise SystemExit(
            f"База уже существует: {DB}\n"
            "Чтобы выполнить анализ заново, переименуйте папку "
            "catalog_analysis и повторите запуск."
        )

    # Загружайте Pickle только из доверенного источника.
    print("Загрузка PKL...", flush=True)
    with SOURCE.open("rb") as f:
        data = pickle.load(f)

    catalog = data["yml_catalog"]
    shop = catalog["shop"]
    offers = as_list(shop.get("offers", {}).get("offer"))
    categories = as_list(shop.get("categories", {}).get("category"))

    if not offers:
        raise SystemExit("В каталоге не найдены товары.")

    print("Дата каталога:", catalog.get("@date"), flush=True)
    print(f"Товаров: {len(offers):,}", flush=True)
    print(f"Категорий: {len(categories):,}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)

    try:
        con.executescript("""
        PRAGMA temp_store = FILE;
        PRAGMA cache_size = -65536;

        CREATE TABLE categories (
            category_id TEXT PRIMARY KEY,
            parent_id TEXT,
            category_name TEXT
        );

        CREATE TABLE products (
            product_no INTEGER PRIMARY KEY,
            sku TEXT,
            available TEXT,
            vendor TEXT,
            model TEXT,
            name TEXT
        );

        CREATE TABLE product_categories (
            product_no INTEGER,
            category_id TEXT,
            PRIMARY KEY (product_no, category_id)
        );

        CREATE TABLE params (
            product_no INTEGER,
            parameter TEXT,
            unit TEXT,
            value TEXT
        );
        """)

        con.executemany(
            "INSERT INTO categories VALUES (?, ?, ?)",
            (
                (
                    text(c.get("@id")),
                    text(c.get("@parentId")),
                    text(c.get("#text")),
                )
                for c in categories
            ),
        )

        field_count = Counter()
        field_types = defaultdict(Counter)
        field_nonempty = Counter()
        repeated_params = 0

        for number, offer in enumerate(offers, start=1):
            for field, value in offer.items():
                field_count[field] += 1
                field_types[field][type(value).__name__] += 1

                if value is not None and value not in ("", [], {}):
                    field_nonempty[field] += 1

            con.execute(
                "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)",
                (
                    number,
                    text(offer.get("@id")),
                    text(offer.get("@available")),
                    text(offer.get("vendor")),
                    text(offer.get("model")),
                    text(offer.get("name")),
                ),
            )

            category_block = offer.get("categories") or {}
            ids = {
                get_category_id(v)
                for v in as_list(category_block.get("categoryId"))
            }
            ids.discard("")

            con.executemany(
                "INSERT INTO product_categories VALUES (?, ?)",
                ((number, cid) for cid in ids),
            )

            raw_entries = [
                (
                    text(p.get("@name")),
                    text(p.get("@unit")),
                    text(p.get("#text")),
                )
                for p in as_list(offer.get("param"))
                if isinstance(p, dict)
            ]

            # Разные значения одного параметра сохраняются.
            # Удаляются только полностью одинаковые записи
            # внутри одного товара.
            entries = set(raw_entries)
            repeated_params += len(raw_entries) - len(entries)

            con.executemany(
                "INSERT INTO params VALUES (?, ?, ?, ?)",
                (
                    (number, name, unit, value)
                    for name, unit, value in entries
                ),
            )

            if number % 10000 == 0:
                con.commit()
                print(
                    f"Обработано {number:,} / {len(offers):,}",
                    flush=True,
                )

        con.commit()

        print("Построение индексов...", flush=True)
        con.executescript("""
        CREATE INDEX params_product ON params(product_no);
        CREATE INDEX category_products
            ON product_categories(category_id, product_no);
        """)

        print("Подсчёт значений по категориям...", flush=True)
        con.executescript("""
        CREATE TABLE category_values AS
        SELECT
            pc.category_id,
            p.parameter,
            p.unit,
            p.value,
            COUNT(*) AS product_count
        FROM params AS p
        JOIN product_categories AS pc USING (product_no)
        GROUP BY
            pc.category_id, p.parameter, p.unit, p.value;
        """)
        con.commit()

        def export(filename, query):
            print(f"Создание {filename}...", flush=True)
            cursor = con.execute(query)

            with (OUT / filename).open(
                "w", encoding="utf-8-sig", newline=""
            ) as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(
                    [column[0] for column in cursor.description]
                )
                writer.writerows(cursor)

        print("Создание fields.csv...", flush=True)
        with (OUT / "fields.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                "field",
                "products_with_field",
                "presence_pct",
                "products_with_nonempty_field",
                "python_types",
            ])

            for field, count in field_count.most_common():
                writer.writerow([
                    field,
                    count,
                    round(100 * count / len(offers), 2),
                    field_nonempty[field],
                    json.dumps(
                        dict(field_types[field]),
                        ensure_ascii=False,
                    ),
                ])

        export("categories.csv", """
        WITH all_ids AS (
            SELECT category_id FROM categories
            UNION
            SELECT category_id FROM product_categories
        )
        SELECT
            a.category_id,
            c.parent_id,
            c.category_name,
            COUNT(pc.product_no) AS product_count,
            CASE WHEN c.category_id IS NULL
                 THEN 0 ELSE 1 END AS in_category_dictionary
        FROM all_ids AS a
        LEFT JOIN categories AS c USING (category_id)
        LEFT JOIN product_categories AS pc USING (category_id)
        GROUP BY a.category_id, c.parent_id, c.category_name
        ORDER BY product_count DESC, a.category_id
        """)

        export("parameters.csv", """
        SELECT
            parameter,
            unit,
            COUNT(DISTINCT product_no) AS products_with_parameter,
            COUNT(DISTINCT value) AS unique_values,
            COUNT(DISTINCT CASE
                WHEN TRIM(value) <> '' THEN product_no
            END) AS products_with_nonempty_value,
            ROUND(
                100.0 * COUNT(DISTINCT CASE
                    WHEN TRIM(value) <> '' THEN product_no
                END) / (SELECT COUNT(*) FROM products),
                2
            ) AS filled_pct_all_products
        FROM params
        GROUP BY parameter, unit
        ORDER BY products_with_parameter DESC, parameter, unit
        """)

        export("parameter_values.csv", """
        SELECT
            parameter,
            unit,
            value,
            COUNT(*) AS product_count
        FROM params
        GROUP BY parameter, unit, value
        ORDER BY parameter, unit, product_count DESC, value
        """)

        export("category_parameters.csv", """
        WITH coverage AS (
            SELECT
                pc.category_id,
                p.parameter,
                p.unit,
                COUNT(DISTINCT p.product_no)
                    AS products_with_parameter,
                COUNT(DISTINCT CASE
                    WHEN TRIM(p.value) <> '' THEN p.product_no
                END) AS products_with_nonempty_value
            FROM params AS p
            JOIN product_categories AS pc USING (product_no)
            GROUP BY pc.category_id, p.parameter, p.unit
        ),
        totals AS (
            SELECT
                category_id,
                COUNT(*) AS category_products
            FROM product_categories
            GROUP BY category_id
        ),
        value_counts AS (
            SELECT
                category_id,
                parameter,
                unit,
                COUNT(*) AS unique_values
            FROM category_values
            GROUP BY category_id, parameter, unit
        )
        SELECT
            s.category_id,
            c.category_name,
            s.parameter,
            s.unit,
            t.category_products,
            s.products_with_parameter,
            s.products_with_nonempty_value,
            ROUND(
                100.0 * s.products_with_nonempty_value
                / t.category_products,
                2
            ) AS filled_pct,
            v.unique_values
        FROM coverage AS s
        JOIN totals AS t USING (category_id)
        JOIN value_counts AS v
            USING (category_id, parameter, unit)
        LEFT JOIN categories AS c USING (category_id)
        ORDER BY s.category_id, s.parameter, s.unit
        """)

        export("category_values.csv", """
        SELECT
            v.category_id,
            c.category_name,
            v.parameter,
            v.unit,
            v.value,
            v.product_count
        FROM category_values AS v
        LEFT JOIN categories AS c USING (category_id)
        ORDER BY
            v.category_id,
            v.parameter,
            v.unit,
            v.product_count DESC,
            v.value
        """)

        # Уникальные значения основных полей товара.
        # Характеристики из param находятся в отдельных отчётах.
        queries = []
        for field in ("available", "vendor", "model", "name"):
            queries.append(f"""
                SELECT
                    '{field}' AS field,
                    "{field}" AS value,
                    COUNT(*) AS product_count
                FROM products
                GROUP BY "{field}"
            """)

        export(
            "product_field_values.csv",
            " UNION ALL ".join(queries)
            + " ORDER BY field, product_count DESC, value",
        )

        export("duplicate_sku.csv", """
        SELECT sku, COUNT(*) AS row_count
        FROM products
        GROUP BY sku
        HAVING COUNT(*) > 1
        ORDER BY row_count DESC, sku
        """)

        export("products_without_category.csv", """
        SELECT p.sku, p.name
        FROM products AS p
        WHERE NOT EXISTS (
            SELECT 1
            FROM product_categories AS pc
            WHERE pc.product_no = p.product_no
        )
        """)

        print("\nГотово!", flush=True)
        print("Результаты:", OUT)
        print(
            "Полностью одинаковых повторов характеристик "
            "внутри товаров:",
            repeated_params,
        )

    finally:
        con.close()


if __name__ == "__main__":
    main()