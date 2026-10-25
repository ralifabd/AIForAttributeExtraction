import json
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

PATH_ATTRIBUTES = DATA_DIR / "attrs.parquet"
PATH_ATTRIBUTE_SCHEMA = DATA_DIR / "ideal_attribute_schema.json"
OUTPUT_DIR = DATA_DIR / "search_dictionaries"
DB_PATH = OUTPUT_DIR / "search_values_tmp.sqlite"

BATCH_SIZE = 100_000


def main():
    print("Загружаю названия категорий...", flush=True)

    with open(PATH_ATTRIBUTE_SCHEMA, encoding="utf-8") as f:
        schema = json.load(f)

    category_names = {
        str(c.get("category_id", c.get("categoryId"))): c.get(
            "category", c.get("category_name", "Без названия")
        )
        for c in schema.get("categories", [])
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Открываю attrs.parquet...", flush=True)
    parquet_file = pq.ParquetFile(PATH_ATTRIBUTES)

    conn = sqlite3.connect(DB_PATH)

    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS search_values (
                category_id TEXT NOT NULL,
                attribute_name TEXT NOT NULL,
                value TEXT NOT NULL,
                UNIQUE(category_id, attribute_name, value)
            )
        """)
        conn.execute("DELETE FROM search_values")

        insert_sql = """
            INSERT OR IGNORE INTO search_values
                (category_id, attribute_name, value)
            VALUES (?, ?, ?)
        """

        rows_seen = 0
        rows_added = 0

        # Читаем Parquet порциями, а не целиком.
        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=["categoryId", "attr", "value", "is_search"],
        ):
            df = batch.to_pandas()

            df = df.loc[
                df["is_search"].fillna(False)
                & df["categoryId"].notna()
                & df["attr"].notna()
                & df["value"].notna(),
                ["categoryId", "attr", "value"],
            ]

            df = df.loc[df["value"].astype(str).str.strip().ne("")]

            records = (
                (
                    str(category_id),
                    str(attr),
                    str(value),
                )
                for category_id, attr, value in df.itertuples(
                    index=False, name=None
                )
            )

            chunk_records = list(records)

            conn.executemany(insert_sql, chunk_records)
            conn.commit()

            rows_seen += batch.num_rows
            rows_added += len(chunk_records)

            print(
                f"Прочитано строк: {rows_seen:,}; "
                f"строк с поисковыми атрибутами: {rows_added:,}",
                flush=True,
            )

            del df, chunk_records, batch

        print("Формирую JSON...", flush=True)

        category_ids = conn.execute("""
            SELECT DISTINCT category_id
            FROM search_values
            ORDER BY category_id
        """).fetchall()

        all_data = {}

        for (category_id,) in category_ids:
            attributes = {}

            names = conn.execute("""
                SELECT DISTINCT attribute_name
                FROM search_values
                WHERE category_id = ?
                ORDER BY attribute_name
            """, (category_id,)).fetchall()

            for (name,) in names:
                values = conn.execute("""
                    SELECT value
                    FROM search_values
                    WHERE category_id = ?
                      AND attribute_name = ?
                    ORDER BY value
                """, (category_id, name)).fetchall()

                attributes[name] = [row[0] for row in values]

            category_data = {
                "category_id": category_id,
                "category": category_names.get(
                    category_id, f"Категория {category_id}"
                ),
                "attributes": attributes,
            }

            all_data[category_id] = category_data

            with open(
                OUTPUT_DIR / f"{category_id}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    category_data,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        with open(
            OUTPUT_DIR / "all_search_dictionaries.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                all_data,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Категорий обработано: {len(all_data)}")
        print(f"Результаты сохранены в: {OUTPUT_DIR}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()