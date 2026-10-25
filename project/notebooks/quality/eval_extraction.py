# Оценка качества извлечения атрибутов на синтетическом эталоне из нашего каталога.
#
# Идея: у наших товаров характеристики уже заполнены правильно.
#   1. Берём товар из catalog.sqlite и его правильные значения атрибутов схемы.
#   2. Часть атрибутов показываем, часть прячем (--hide, по умолчанию 30%).
#   3. LLM пишет по ПОКАЗАННЫМ характеристикам описание "как у продавца".
#   4. Это описание прогоняем через ваш extract_attributes().
#   5. Сравниваем извлечённое с правильным -> precision, recall, доля выдумок.
#
# Запуск из папки notebooks (там, где main.py):
#   py -3.12 -m quality.eval_extraction --n 10          <- сначала на 10 товарах
#   py -3.12 -m quality.eval_extraction --n 100
#
# Результат (папка reports/eval_extraction в корне проекта):
#   summary_attributes.csv  - метрики по каждому атрибуту
#   summary_categories.csv  - метрики по каждой категории
#   details.csv             - каждая пара "товар - атрибут": правильное, извлечённое, итог
#   descriptions.json       - сгенерированные описания (кэш: повторный запуск их не
#                             генерирует заново и не тратит запросы)

import argparse
import csv
import json
import random
import re
import sqlite3
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rapidfuzz import fuzz

from config import DATA_DIR, PROJECT_DIR
from extract_ideal_attributes import (
    CATEGORIES,
    ask_llm,
    extract_attributes,
    get_attribute_name,
    get_category_id,
)

# ---------------------------------------------------------------- сравнение значений

UNITS = {  # единица -> (величина, множитель к базовой)
    "мм": ("len", 1), "см": ("len", 10), "м": ("len", 1000),
    "г": ("mass", 1), "кг": ("mass", 1000),
    "мл": ("vol", 1), "л": ("vol", 1000),
    "вт": ("pow", 1), "квт": ("pow", 1000),
    "мб": ("mem", 1), "гб": ("mem", 1024), "тб": ("mem", 1024 * 1024),
    "mb": ("mem", 1), "gb": ("mem", 1024), "tb": ("mem", 1024 * 1024),
    "мач": ("cap", 1), "mah": ("cap", 1),
    "гц": ("freq", 1), "кгц": ("freq", 1000),
}
YES = {"да", "есть", "yes", "true", "поддерживается", "имеется", "присутствует", "+"}
NO = {"нет", "no", "false", "отсутствует", "не поддерживается", "-"}


def norm_text(s):
    s = str(s or "").lower().replace("ё", "е")
    return re.sub(r"\s+", " ", s).strip(" .;,")


def parse_number(value, unit=""):
    """'1,7 л' -> (1700, 'vol'); '2200' + unit 'Вт' -> (2200, 'pow')."""
    text = norm_text(f"{value} {unit or ''}")
    m = re.search(r"(-?\d+(?:[.,]\d+)?)\s*([a-zа-я]+)?", text)
    if not m:
        return None
    num = float(m.group(1).replace(",", "."))
    u = (m.group(2) or norm_text(unit) or "").strip()
    if u in UNITS:
        kind, k = UNITS[u]
        return num * k, kind
    return num, None


def values_match(true_value, true_unit, got_value, got_unit):
    """Мягкое сравнение: числа с переводом единиц, да/нет, текст по похожести."""
    t, g = norm_text(true_value), norm_text(got_value)
    if not t or not g:
        return False
    if t == g:
        return True
    # да / нет
    if t in YES | NO or g in YES | NO:
        return (t in YES and g in YES) or (t in NO and g in NO)
    # числа
    tn, gn = parse_number(true_value, true_unit), parse_number(got_value, got_unit)
    if tn and gn:
        (tv, tk), (gv, gk) = tn, gn
        if tk and gk and tk != gk:
            return False
        if tv == 0:
            return gv == 0
        return abs(tv - gv) / abs(tv) <= 0.02
    # текст: "черный" vs "черный матовый", "сталь, пластик" vs "пластик, сталь"
    return fuzz.token_set_ratio(t, g) >= 85


# ---------------------------------------------------------------- эталон из каталога

def build_truth(con, product_no, attr_to_param):
    """Правильные значения атрибутов схемы для товара."""
    rows = con.execute(
        "SELECT parameter, unit, value FROM params WHERE product_no = ?",
        (product_no,)).fetchall()
    by_param = defaultdict(list)
    for p, u, v in rows:
        if v and str(v).strip():
            by_param[p].append((str(v).strip(), (u or "").strip()))
    truth = {}
    for attr, param in attr_to_param.items():
        vals = by_param.get(param)
        if vals:
            truth[attr] = {"value": "; ".join(v for v, _ in vals),
                           "unit": vals[0][1], "param": param}
    return truth


def map_attributes(con, category, product_nos):
    """Атрибут схемы -> название характеристики в каталоге.
    Пробуем source_fields, затем "Имя_search" (точные значения), затем "Имя"."""
    if not product_nos:
        return {}
    marks = ",".join("?" * len(product_nos))
    present = {p for (p,) in con.execute(
        f"SELECT DISTINCT parameter FROM params WHERE product_no IN ({marks})",
        product_nos)}
    present_cf = {p.casefold(): p for p in present}
    mapping = {}
    for a in category.get("attributes", []):
        if a.get("priority") != "core":
            continue
        name = get_attribute_name(a)
        if not name:
            continue
        options = [s for s in (a.get("source_fields") or []) if isinstance(s, str)]
        options += [f"{name}_search", name]
        for opt in options:
            hit = present_cf.get(opt.casefold())
            if hit:
                mapping[name] = hit
                break
    return mapping


def collect_samples(con, n, per_cat, min_attrs, seed):
    rnd = random.Random(seed)
    cats = [c for c in CATEGORIES
            if any(a.get("priority") == "core" for a in c.get("attributes", []))]
    rnd.shuffle(cats)
    samples = []
    for cat in cats:
        if len(samples) >= n:
            break
        cid = str(get_category_id(cat))
        all_pnos = [r[0] for r in con.execute(
            "SELECT product_no FROM product_categories WHERE category_id = ? "
            "ORDER BY product_no", (cid,))]
        # одинаковая выборка при каждом запуске (зависит только от --seed)
        pnos = rnd.sample(all_pnos, min(40, len(all_pnos)))
        if not pnos:
            continue
        mapping = map_attributes(con, cat, pnos)
        if len(mapping) < min_attrs:
            continue
        taken = 0
        for pno in pnos:
            truth = build_truth(con, pno, mapping)
            if len(truth) < min_attrs:
                continue
            name = con.execute("SELECT name, sku FROM products WHERE product_no = ?",
                               (pno,)).fetchone()
            samples.append({"product_no": pno, "sku": name[1], "title": name[0],
                            "category": cat, "category_id": cid, "truth": truth})
            taken += 1
            if taken >= per_cat or len(samples) >= n:
                break
    return samples


# ---------------------------------------------------------------- генерация описания

GEN_PROMPT = """
Ты пишешь описание товара для карточки интернет-магазина от лица продавца.

Тебе дано название товара и список характеристик. Напиши связное описание
на русском языке (3-6 предложений), естественным языком, как пишут продавцы.

Правила:
1. Упомяни ВСЕ переданные характеристики, но своими словами, не списком.
2. Можно менять форму записи: "1,7 л" вместо "1700 мл", "чёрный" вместо "черный".
3. НЕ добавляй никаких других характеристик и чисел, которых нет в списке.
4. Не используй слово "характеристики" и не делай таблиц.

Верни только JSON: {"description": "текст описания"}
"""


def make_description(sample, shown):
    payload = {
        "title": sample["title"],
        "characteristics": [
            {"name": a, "value": sample["truth"][a]["value"],
             "unit": sample["truth"][a]["unit"] or None}
            for a in shown],
    }
    return ask_llm(GEN_PROMPT, payload).get("description", "")


# ---------------------------------------------------------------- один товар

def evaluate_one(sample, hide_share, rnd_seed, cache):
    rnd = random.Random(f"{rnd_seed}-{sample['product_no']}")
    attrs = sorted(sample["truth"])
    n_hide = int(round(len(attrs) * hide_share))
    hidden = set(rnd.sample(attrs, n_hide)) if n_hide else set()
    shown = [a for a in attrs if a not in hidden]

    key = str(sample["product_no"])
    if key in cache:
        description = cache[key]["description"]
        shown = cache[key]["shown"]
        hidden = set(cache[key]["hidden"])
    else:
        description = make_description(sample, shown)
        cache[key] = {"description": description, "shown": shown,
                      "hidden": sorted(hidden), "title": sample["title"]}

    source = {"sku_id": sample["sku"] or key, "title": sample["title"],
              "description": description}
    extracted = extract_attributes(source, sample["category"])
    got = {p["@name"]: (p.get("#text", ""), p.get("@unit", "")) for p in extracted}

    rows = []
    for a in attrs:
        t = sample["truth"][a]
        g = got.get(a)
        if a in hidden:
            status = "выдумка (атрибут спрятан)" if g else "верно не извлечён"
        elif g is None:
            status = "пропущен"
        elif values_match(t["value"], t["unit"], g[0], g[1]):
            status = "верно"
        else:
            status = "неверно"
        rows.append({
            "product_no": sample["product_no"], "title": sample["title"],
            "category_id": sample["category_id"],
            "category": sample["category"].get("category",
                                               sample["category"].get("category_name", "")),
            "attribute": a, "shown_in_text": a not in hidden,
            "true_value": f"{t['value']} {t['unit']}".strip(),
            "extracted_value": f"{g[0]} {g[1]}".strip() if g else "",
            "status": status, "catalog_param": t["param"],
        })
    # извлечено то, чего нет в эталоне вовсе - проверить нельзя, но покажем
    for a, g in got.items():
        if a not in sample["truth"]:
            rows.append({
                "product_no": sample["product_no"], "title": sample["title"],
                "category_id": sample["category_id"],
                "category": sample["category"].get("category", ""),
                "attribute": a, "shown_in_text": "",
                "true_value": "", "extracted_value": f"{g[0]} {g[1]}".strip(),
                "status": "нет эталона (не проверить)", "catalog_param": "",
            })
    return rows


# ---------------------------------------------------------------- метрики

def metrics(rows):
    c = defaultdict(int)
    for r in rows:
        c[r["status"]] += 1
    correct, wrong, missed = c["верно"], c["неверно"], c["пропущен"]
    halluc, ok_hidden = c["выдумка (атрибут спрятан)"], c["верно не извлечён"]
    extracted = correct + wrong + halluc
    shown = correct + wrong + missed
    hidden = halluc + ok_hidden
    p = correct / extracted if extracted else None
    rcl = correct / shown if shown else None
    f1 = 2 * p * rcl / (p + rcl) if p and rcl else None
    return {
        "precision": p, "recall": rcl, "f1": f1,
        "hallucination_rate": halluc / hidden if hidden else None,
        "correct": correct, "wrong": wrong, "missed": missed,
        "hallucinated": halluc, "shown_total": shown, "hidden_total": hidden,
    }


def pct(x):
    return "" if x is None else f"{x:.1%}"


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(rows)


def summary(rows, key):
    groups = defaultdict(list)
    for r in rows:
        if r["status"] != "нет эталона (не проверить)":
            groups[r[key]].append(r)
    out = []
    for k, rs in groups.items():
        m = metrics(rs)
        out.append({key: k, "precision": pct(m["precision"]), "recall": pct(m["recall"]),
                    "f1": pct(m["f1"]), "hallucination_rate": pct(m["hallucination_rate"]),
                    "correct": m["correct"], "wrong": m["wrong"], "missed": m["missed"],
                    "hallucinated": m["hallucinated"], "checked": len(rs)})
    return sorted(out, key=lambda r: -r["checked"])


# ---------------------------------------------------------------- запуск

def main():
    ap = argparse.ArgumentParser(description="Оценка извлечения атрибутов")
    ap.add_argument("--catalog", default=str(DATA_DIR / "catalog.sqlite"),
                    help="catalog.sqlite из analyse_mvd.py (нужна таблица params)")
    ap.add_argument("--n", type=int, default=100, help="сколько товаров проверить")
    ap.add_argument("--per-cat", type=int, default=3, help="товаров на категорию")
    ap.add_argument("--min-attrs", type=int, default=3,
                    help="брать товары, у которых хотя бы столько атрибутов схемы")
    ap.add_argument("--hide", type=float, default=0.3,
                    help="доля атрибутов, которые прячем (проверка на выдумки)")
    ap.add_argument("--workers", type=int, default=4, help="параллельных запросов")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(PROJECT_DIR / "reports" / "eval_extraction"))
    args = ap.parse_args()

    cat_path = Path(args.catalog)
    if not cat_path.is_file():
        raise SystemExit(f"Не найден каталог: {cat_path.resolve()}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "descriptions.json"
    cache = (json.loads(cache_path.read_text(encoding="utf-8"))
             if cache_path.exists() else {})

    con = sqlite3.connect(cat_path, check_same_thread=False)
    print("Подбираю товары с заполненными атрибутами схемы...", flush=True)
    samples = collect_samples(con, args.n, args.per_cat, args.min_attrs, args.seed)
    con.close()
    if not samples:
        raise SystemExit(
            "Не нашлось товаров: названия атрибутов схемы не совпали с характеристиками "
            "каталога. Пришлите кусок ideal_attribute_schema.json - поправлю сопоставление.")
    n_cats = len({s["category_id"] for s in samples})
    print(f"Товаров: {len(samples)} из {n_cats} категорий. "
          f"Запросов к LLM: ~{len(samples) * 2} (часть описаний может быть в кэше)",
          flush=True)

    all_rows, errors, t0 = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_one, s, args.hide, args.seed, cache): s
                   for s in samples}
        for i, fut in enumerate(as_completed(futures), 1):
            s = futures[fut]
            try:
                all_rows += fut.result()
            except Exception as e:
                errors += 1
                print(f"  ошибка на товаре {s['product_no']} ({s['title'][:40]}): {e!r}",
                      flush=True)
            if i % 10 == 0 or i == len(samples):
                print(f"  {i}/{len(samples)}  ({time.time() - t0:.0f} с)", flush=True)
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                                      encoding="utf-8")

    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    write_csv(out / "details.csv", all_rows)
    write_csv(out / "summary_attributes.csv", summary(all_rows, "attribute"))
    write_csv(out / "summary_categories.csv", summary(all_rows, "category"))

    m = metrics([r for r in all_rows if r["status"] != "нет эталона (не проверить)"])
    unverifiable = sum(r["status"] == "нет эталона (не проверить)" for r in all_rows)
    print("\n=== Качество извлечения (синтетический эталон из каталога) ===")
    print(f"Precision (из извлечённого верно):          {pct(m['precision'])}")
    print(f"Recall (из упомянутого в тексте найдено):   {pct(m['recall'])}")
    print(f"F1:                                          {pct(m['f1'])}")
    print(f"Доля выдумок (извлёк спрятанный атрибут):   {pct(m['hallucination_rate'])}")
    print(f"\nВерно: {m['correct']}, неверно: {m['wrong']}, пропущено: {m['missed']}, "
          f"выдумано: {m['hallucinated']}")
    print(f"Извлечено без эталона (не проверить): {unverifiable}")
    if errors:
        print(f"Товаров с ошибкой: {errors}")
    print(f"\nОтчёты: {out.resolve()}")


if __name__ == "__main__":
    main()
