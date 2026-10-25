# Урезанная таблица товаров + векторная таблица для поиска категории.
#
# Берёт catalog.sqlite  и создаёт НОВЫЙ файл rag.sqlite.
# catalog.sqlite не изменяется.
#
# В rag.sqlite появятся таблицы:
#   index_items  - урезанная таблица: без повторов названий внутри категории,
#                  не больше --cap названий на категорию, + по одной строке
#                  на каждую категорию ("карточка": название категории и раздел)
#   item_vectors - вектор для каждой строки index_items (Qwen3-Embedding-0.6B)
#   spend        - сколько токенов потрачено по дням
#
# Порядок запуска (PowerShell, из папки со скриптом):
#   py -3.12 -m pip install requests numpy
#   $env:OPENAI_API_KEY="ваш_ключ"
#   py -3.12 make_vectors.py index --db catalog_analysis/catalog.sqlite
#   py -3.12 make_vectors.py embed --limit 1000      <- пробный запуск
#   py -3.12 make_vectors.py embed                   <- всё остальное
#   py -3.12 make_vectors.py stats                   <- что получилось
#
# embed можно запускать сколько угодно раз: он векторизует только то,
# чего ещё нет в item_vectors. При обрыве или дневном лимите просто
# запустите его снова (в т.ч. на следующий день).

import argparse
import os
import random
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

API_BASE = "https://hub.gigaschool.ru/api/v1/llm/v1"
API_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def norm(s):
    """Нижний регистр, ё -> е, лишние пробелы убираем."""
    s = str(s or "").lower().replace("ё", "е")
    return re.sub(r"\s+", " ", s).strip()


def make_text(name, vendor=""):
    """name + vendor. Бренд добавляем, только если его ещё нет в названии."""
    n, v = norm(name), norm(vendor)
    if v and v not in n:
        n = f"{n} {v}"
    return n


# ---------------------------------------------------------------- 1. урезанная таблица

def category_paths(con):
    """category_id -> 'Корень > Раздел > Категория' (одинаковые соседние имена склеиваем)."""
    rows = con.execute(
        "SELECT category_id, parent_id, category_name FROM categories").fetchall()
    parent = {r[0]: r[1] for r in rows}
    name = {r[0]: r[2] for r in rows}
    paths = {}
    for cid in name:
        chain, cur, guard = [], cid, 0
        while cur and cur in name and guard < 20:
            nm = name[cur] or f"[без названия {cur}]"
            if not chain or chain[-1] != nm:
                chain.append(nm)
            cur, guard = parent.get(cur) or "", guard + 1
        paths[cid] = " > ".join(reversed(chain))
    return paths, name, parent


def cmd_index(args):
    src, out = Path(args.db), Path(args.out)
    if not src.is_file():
        raise SystemExit(f"Не найден файл: {src.resolve()}")
    con_out = sqlite3.connect(out)
    if con_out.execute("SELECT name FROM sqlite_master "
                       "WHERE name='index_items'").fetchone():
        raise SystemExit(
            f"В {out} уже есть таблица index_items.\n"
            "Если хотите пересобрать: удалите файл rag.sqlite "
            "(векторы тоже придётся считать заново).")

    print(f"Читаю {src} ...", flush=True)
    con = sqlite3.connect(src)
    paths, cat_name, cat_parent = category_paths(con)
    rows = con.execute("""
        SELECT p.sku, p.name, p.vendor, pc.category_id
        FROM products AS p
        JOIN product_categories AS pc USING (product_no)
    """).fetchall()
    con.close()
    print(f"Товаров в каталоге: {len(rows):,}", flush=True)

    # 1) убираем повторы: одинаковый текст внутри одной категории -> одна строка
    by_cat = defaultdict(dict)      # category_id -> {text: (sku, name, vendor, count)}
    for sku, name, vendor, cid in rows:
        text = make_text(name, vendor)
        if not text:
            continue
        seen = by_cat[cid].get(text)
        if seen:
            by_cat[cid][text] = (seen[0], seen[1], seen[2], seen[3] + 1)
        else:
            by_cat[cid][text] = (sku, name, vendor, 1)
    unique = sum(len(v) for v in by_cat.values())
    print(f"Уникальных названий (в своей категории): {unique:,}", flush=True)

    # 2) не больше cap названий на категорию (случайная выборка, но воспроизводимая)
    rnd = random.Random(args.seed)
    items = []
    for cid in sorted(by_cat):
        entries = sorted(by_cat[cid].items())
        if len(entries) > args.cap:
            entries = rnd.sample(entries, args.cap)
        for text, (sku, name, vendor, cnt) in entries:
            items.append(("product", text, sku, name, vendor, cnt, cid,
                          paths.get(cid, "")))

    # 3) карточка категории: помогает маленьким категориям
    if not args.no_category_cards:
        for cid in sorted(by_cat):
            leaf = cat_name.get(cid) or ""
            if not leaf:
                continue
            par = cat_name.get(cat_parent.get(cid) or "", "") or ""
            text = norm(leaf if not par or par == leaf else f"{leaf} — {par}")
            items.append(("category", text, "", f"[категория] {leaf}", "", 0, cid,
                          paths.get(cid, "")))

    con_out.executescript("""
        CREATE TABLE index_items (
            item_id       INTEGER PRIMARY KEY,
            kind          TEXT,     -- product / category
            text          TEXT,     -- что отправляем в векторизацию
            sku           TEXT,     -- пример артикула с таким названием
            name          TEXT,
            vendor        TEXT,
            same_name_cnt INTEGER,  -- сколько товаров в категории с таким же названием
            category_id   TEXT,
            category_path TEXT
        );
        CREATE INDEX index_items_cat ON index_items(category_id);
        CREATE TABLE IF NOT EXISTS item_vectors (
            item_id INTEGER PRIMARY KEY,
            dim     INTEGER,
            vector  BLOB          -- float16, dim чисел
        );
        CREATE TABLE IF NOT EXISTS spend (day TEXT PRIMARY KEY, tokens INTEGER);
    """)
    con_out.executemany(
        "INSERT INTO index_items (kind, text, sku, name, vendor, same_name_cnt, "
        "category_id, category_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", items)
    con_out.commit()

    n_prod = sum(1 for i in items if i[0] == "product")
    n_cat = len(items) - n_prod
    chars = sum(len(i[1]) for i in items)
    print("\n=== Урезанная таблица index_items готова ===")
    print(f"Строк-товаров:   {n_prod:,}  (было товаров {len(rows):,}, "
          f"уникальных {unique:,}, лимит {args.cap} на категорию)")
    print(f"Строк-категорий: {n_cat:,}")
    print(f"Категорий:       {len(by_cat):,}")
    print(f"Оценка для векторизации: ~{chars / 3 / 1e6:.1f} млн токенов "
          f"≈ {chars / 3 / 1e6 * args.price:.0f} ₽ (при {args.price} ₽ за 1 млн)")
    print(f"Файл: {out.resolve()}")
    con_out.close()


# ---------------------------------------------------------------- 2. векторы

def spent_today(con):
    day = time.strftime("%Y-%m-%d")
    r = con.execute("SELECT tokens FROM spend WHERE day = ?", (day,)).fetchone()
    return r[0] if r else 0


def add_spend(con, tokens):
    day = time.strftime("%Y-%m-%d")
    con.execute("INSERT INTO spend (day, tokens) VALUES (?, ?) "
                "ON CONFLICT(day) DO UPDATE SET tokens = tokens + excluded.tokens",
                (day, int(tokens)))


def request_embeddings(texts, args, key):
    import requests
    url = f"{args.api_base.rstrip('/')}/embeddings"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": args.api_model, "input": texts}
    err = ""
    for attempt in range(6):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=120)
        except requests.RequestException as e:
            err = str(e)
        else:
            if r.status_code == 200:
                data = r.json()
                rows = sorted(data["data"], key=lambda x: x["index"])
                vecs = np.array([row["embedding"] for row in rows], dtype=np.float32)
                usage = data.get("usage") or {}
                tokens = usage.get("prompt_tokens") or usage.get("total_tokens")
                if tokens is None:          # сервис не прислал счётчик: оценка
                    tokens = sum(len(t) for t in texts) // 3
                return vecs, tokens
            if r.status_code in (401, 403):
                raise SystemExit(f"Ошибка доступа {r.status_code}: проверьте ключ.\n"
                                 f"{r.text[:500]}")
            if r.status_code not in (408, 409, 429) and r.status_code < 500:
                raise SystemExit(f"API вернул ошибку {r.status_code}:\n{r.text[:1000]}")
            err = f"HTTP {r.status_code}: {r.text[:200]}"
        wait = min(60, 2 ** attempt)
        print(f"  ошибка ({err}), повтор через {wait} с", flush=True)
        time.sleep(wait)
    raise SystemExit("API не отвечает после нескольких попыток. "
                     "Запустите ещё раз позже: готовое уже сохранено.")


def cmd_embed(args):
    key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Не задан API-ключ. В PowerShell выполните:\n"
                         '  $env:OPENAI_API_KEY="ваш_ключ"')
    con = sqlite3.connect(args.out)
    if not con.execute("SELECT name FROM sqlite_master "
                       "WHERE name='index_items'").fetchone():
        raise SystemExit("Сначала создайте урезанную таблицу: make_vectors.py index")

    todo = con.execute("""
        SELECT i.item_id, i.text FROM index_items AS i
        LEFT JOIN item_vectors AS v USING (item_id)
        WHERE v.item_id IS NULL
        ORDER BY i.item_id
    """).fetchall()
    done_before = con.execute("SELECT COUNT(*) FROM item_vectors").fetchone()[0]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("Все строки уже векторизованы.")
        return

    price = args.price / 1e6
    print(f"Уже есть векторов: {done_before:,}. Сейчас векторизуем: {len(todo):,}")
    print(f"Сегодня уже потрачено: {spent_today(con) * price:.2f} ₽ "
          f"из {args.daily_budget:.0f} ₽", flush=True)

    t0, done, tokens_run = time.time(), 0, 0
    for start in range(0, len(todo), args.batch):
        if spent_today(con) * price >= args.daily_budget:
            print(f"\nДневной лимит {args.daily_budget:.0f} ₽ достигнут. Готовое сохранено.\n"
                  "Запустите эту же команду завтра, она продолжит с места остановки.")
            break
        part = todo[start:start + args.batch]
        vecs, tokens = request_embeddings([t for _, t in part], args, key)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        con.executemany(
            "INSERT INTO item_vectors (item_id, dim, vector) VALUES (?, ?, ?)",
            [(item_id, v.shape[0], v.astype(np.float16).tobytes())
             for (item_id, _), v in zip(part, vecs)])
        add_spend(con, tokens)
        con.commit()                       # сохраняем после каждой пачки
        done += len(part)
        tokens_run += tokens
        if (start // args.batch) % 20 == 0 or done == len(todo):
            speed = done / max(time.time() - t0, 1e-6)
            left = (len(todo) - done) / max(speed, 1e-6)
            print(f"  {done:,} / {len(todo):,}  |  в этом запуске {tokens_run:,} токенов "
                  f"= {tokens_run * price:.2f} ₽  |  осталось ~{left / 60:.0f} мин",
                  flush=True)

    total = con.execute("SELECT COUNT(*) FROM item_vectors").fetchone()[0]
    all_rows = con.execute("SELECT COUNT(*) FROM index_items").fetchone()[0]
    print(f"\nВ этом запуске: {done:,} строк, {tokens_run:,} токенов = "
          f"{tokens_run * price:.2f} ₽")
    if done:
        print(f"В среднем {tokens_run / done:.1f} токена на строку. "
              f"Прогноз на всю таблицу: ~{tokens_run / done * all_rows / 1e6:.1f} млн токенов "
              f"≈ {tokens_run / done * all_rows * price:.0f} ₽")
    print(f"Векторизовано: {total:,} из {all_rows:,}")
    con.close()


# ---------------------------------------------------------------- 3. статистика

def cmd_stats(args):
    con = sqlite3.connect(args.out)
    q = lambda sql: con.execute(sql).fetchall()
    print("Строк в index_items:", q("SELECT kind, COUNT(*) FROM index_items GROUP BY kind"))
    print("Категорий:", q("SELECT COUNT(DISTINCT category_id) FROM index_items")[0][0])
    print("Векторизовано:", q("SELECT COUNT(*) FROM item_vectors")[0][0])
    dim = q("SELECT dim FROM item_vectors LIMIT 1")
    if dim:
        print("Размер вектора:", dim[0][0])
    print("Расход по дням (токены):", q("SELECT day, tokens FROM spend ORDER BY day"))
    print("\nКатегории с наибольшим числом строк:")
    for row in q("""SELECT category_path, COUNT(*) FROM index_items
                    WHERE kind='product' GROUP BY category_id
                    ORDER BY 2 DESC LIMIT 5"""):
        print("  ", row)
    print("\nПримеры строк:")
    for row in q("""SELECT item_id, kind, text, category_path FROM index_items
                    ORDER BY RANDOM() LIMIT 5"""):
        print("  ", row)
    con.close()


# ---------------------------------------------------------------- запуск

def main():
    ap = argparse.ArgumentParser(description="Урезанная таблица и векторы для RAG")
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("index", help="создать урезанную таблицу index_items")
    i.add_argument("--db", default="catalog_analysis/catalog.sqlite")
    i.add_argument("--out", default="rag.sqlite")
    i.add_argument("--cap", type=int, default=200, help="макс. названий на категорию")
    i.add_argument("--no-category-cards", action="store_true")
    i.add_argument("--seed", type=int, default=42)
    i.add_argument("--price", type=float, default=0.85, help="₽ за 1 млн токенов")

    e = sub.add_parser("embed", help="векторизовать index_items через API")
    e.add_argument("--out", default="rag.sqlite")
    e.add_argument("--limit", type=int, default=0, help="пробный запуск на N строках")
    e.add_argument("--api-base", default=API_BASE)
    e.add_argument("--api-model", default=API_MODEL)
    e.add_argument("--api-key", default=None)
    e.add_argument("--batch", type=int, default=64, help="строк в одном запросе")
    e.add_argument("--price", type=float, default=0.85, help="₽ за 1 млн токенов")
    e.add_argument("--daily-budget", type=float, default=95.0)

    s = sub.add_parser("stats", help="показать, что лежит в rag.sqlite")
    s.add_argument("--out", default="rag.sqlite")

    args = ap.parse_args()
    {"index": cmd_index, "embed": cmd_embed, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    main()
