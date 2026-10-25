# Поиск категории по векторам из rag.sqlite (его создаёт make_vectors.py).
#
# Команды (PowerShell, из папки со скриптами и rag.sqlite):
#   $env:OPENAI_API_KEY="ваш_ключ"
#   py -3.12 search.py eval
#   py -3.12 search.py predict --input partner.json --output result.csv
#
# eval    - проверка точности на наших товарах: берём товар из rag.sqlite,
#           "прячем" его и похожие варианты, ищем категорию заново.
# predict - товары продавца (JSON как в примере или CSV с колонками name,
#           description). Ищем по названию + началу описания, отдельно
#           по одному названию; если ответы разные - помечаем для проверки.
#
# Через API отправляются только искомые тексты (векторы каталога уже готовы).


import argparse
import json
import os
import re
import sqlite3
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from types import SimpleNamespace

from search.make_vectors import (API_BASE, API_MODEL, add_spend, make_text, norm,
                          request_embeddings)

INSTRUCT = ("Given a product name and description from an online store, "
            "retrieve catalog products of the same category")



class VectorBase:
    def __init__(self, path):
        if not Path(path).is_file():
            raise SystemExit(f"Не найден файл: {Path(path).resolve()}")
        self.con = sqlite3.connect(path)
        rows = self.con.execute("""
            SELECT i.item_id, i.kind, i.text, i.name, i.category_id,
                   i.category_path, v.vector, i.sku
            FROM index_items AS i JOIN item_vectors AS v USING (item_id)
            ORDER BY i.item_id
        """).fetchall()
        if not rows:
            raise SystemExit("В rag.sqlite нет векторов. Сначала: make_vectors.py embed")
        total = self.con.execute("SELECT COUNT(*) FROM index_items").fetchone()[0]
        if len(rows) < total:
            print(f"Внимание: векторизовано {len(rows):,} из {total:,} строк. "
                  "Поиск идёт только по готовым.", flush=True)
        self.item_id = np.array([r[0] for r in rows])
        self.kind = [r[1] for r in rows]
        self.text = [r[2] for r in rows]
        self.name = [r[3] for r in rows]
        self.cat = np.array([r[4] for r in rows])
        self.sku = [r[7] for r in rows]
        self.paths = {r[4]: r[5] for r in rows}
        self.matrix = np.vstack([np.frombuffer(r[6], dtype=np.float16) for r in rows])
        print(f"Загружено векторов: {len(rows):,} (размер {self.matrix.shape[1]})",
              flush=True)

    def path(self, cid):
        return self.paths.get(cid, f"[категория {cid}]")

    def search(self, qvecs, k, mask=None, chunk=20000):
        """Для каждого запроса: k ближайших строк (индексы и похожесть).
        mask[i] - набор индексов строк, которые для запроса i прятать."""
        n_q = qvecs.shape[0]
        sims = np.empty((n_q, self.matrix.shape[0]), dtype=np.float32)
        for start in range(0, self.matrix.shape[0], chunk):
            block = self.matrix[start:start + chunk].astype(np.float32)
            sims[:, start:start + chunk] = qvecs @ block.T
        if mask:
            for qi, hidden in mask.items():
                if hidden:
                    sims[qi, list(hidden)] = -1.0
        kk = min(k, sims.shape[1])
        part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        ps = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-ps, axis=1)
        return (np.take_along_axis(part, order, axis=1),
                np.take_along_axis(ps, order, axis=1))


def vote(idx_row, sim_row, cats, top_c=5, temp=0.03, k=None):
    """Соседи голосуют за свои категории.
    Вес соседа = exp((его похожесть - похожесть самого близкого) / temp):
    самый близкий весит 1, а кто похож на 0.1 меньше, при temp=0.03 весит ~0.04.
    Так решают ближайшие соседи, и большие категории не "перекрикивают"
    маленькие числом товаров. temp=0 - старый вариант: вес = похожесть².
    Возвращает [(category_id, доля голосов), ...] по убыванию."""
    if k:
        idx_row, sim_row = idx_row[:k], sim_row[:k]
    scores = defaultdict(float)
    top = float(sim_row[0]) if len(sim_row) else 0.0
    for i, s in zip(idx_row, sim_row):
        if s <= 0:
            continue
        w = np.exp((float(s) - top) / temp) if temp > 0 else float(s) ** 2
        scores[cats[i]] += w
    total = sum(scores.values()) or 1.0
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_c]
    return [(c, s / total) for c, s in ranked]


# ---------------------------------------------------------------- векторы запросов

def embed_queries(texts, args, con):
    key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Не задан API-ключ. В PowerShell выполните:\n"
                         '  $env:OPENAI_API_KEY="ваш_ключ"')
    full = [f"Instruct: {INSTRUCT}\nQuery: {t}" for t in texts]
    out, tokens_all = [], 0
    for start in range(0, len(full), args.batch):
        vecs, tokens = request_embeddings(full[start:start + args.batch], args, key)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        out.append(vecs)
        tokens_all += tokens
        add_spend(con, tokens)
        con.commit()
        if len(full) > 1000 and (start // args.batch) % 20 == 0:
            print(f"  векторизовано запросов: {start + len(vecs):,} / {len(full):,}",
                  flush=True)
    print(f"Векторизация запросов: {tokens_all:,} токенов = "
          f"{tokens_all * args.price / 1e6:.3f} ₽", flush=True)
    return np.vstack(out).astype(np.float32)


# ---------------------------------------------------------------- eval

def group_key(text):
    """Ключ 'одной модели': название без цифр и знаков.
    Матрас ... 160x200 и Матрас ... 180x200 -> один ключ."""
    return re.sub(r"[\d\W_]+", " ", text).strip()


def cmd_eval(args):
    vb = VectorBase(args.db)
    prod_idx = [i for i, k in enumerate(vb.kind) if k == "product"]

    # Группы похожих названий: при проверке прячем всю группу,
    # иначе скрипт "найдёт" почти тот же товар и результат будет завышен.
    groups = defaultdict(list)
    for i in prod_idx:
        groups[group_key(vb.text[i])].append(i)

    # Выборка для проверки: до N товаров с каждой категории
    by_cat = defaultdict(list)
    for i in prod_idx:
        by_cat[vb.cat[i]].append(i)
    rnd = np.random.default_rng(args.seed)
    test = []
    for cid in sorted(by_cat):
        items = by_cat[cid]
        take = rnd.choice(len(items), size=min(args.per_cat, len(items)), replace=False)
        test += [items[t] for t in take]
    if len(test) > args.max_test:
        test = list(rnd.choice(test, size=args.max_test, replace=False))
    print(f"Проверяем на {len(test):,} товарах ({args.per_cat} на категорию, "
          f"категорий {len(by_cat):,})", flush=True)

    qvecs = embed_queries([vb.text[i] for i in test], args, vb.con)
    mask = {qi: set(groups[group_key(vb.text[i])]) for qi, i in enumerate(test)}
    idx, sim = vb.search(qvecs, args.k, mask)

    rows = []
    for qi, i in enumerate(test):
        true = vb.cat[i]
        cands = vote(idx[qi], sim[qi], vb.cat, temp=args.temp)
        top1_vote = cands[0][0] if cands else ""
        top1_nn = vb.cat[idx[qi][0]]
        rows.append({
            "text": vb.text[i],
            "true_category": vb.path(true),
            "pred_vote": vb.path(top1_vote),
            "pred_nearest": vb.path(top1_nn),
            "confidence": round(cands[0][1], 3) if cands else 0.0,
            "best_sim": round(float(sim[qi][0]), 3),
            "ok_vote": top1_vote == true,
            "ok_nearest": top1_nn == true,
            "ok_top5": true in [c for c, _ in cands],
            "ok_root_vote": vb.path(top1_vote).split(" > ")[0] == vb.path(true).split(" > ")[0],
            "top5": " | ".join(f"{vb.path(c)} ({s:.2f})" for c, s in cands),
            "nearest_items": " | ".join(
                f"{vb.name[j]} ({sim[qi][n]:.2f})" for n, j in enumerate(idx[qi][:5])),
        })

    n = len(rows)
    acc = lambda key: sum(r[key] for r in rows) / n

    print("\n=== Сравнение способов выбрать категорию ===")
    print(" способ                                   | верная первой | верная в топ-5")
    trues = [vb.cat[i] for i in test]
    strategies = [("один самый похожий товар", dict(k=1, temp=0.03))]
    strategies += [(f"голосование, temp={t}", dict(temp=t)) for t in (0.01, 0.03, 0.05, 0.1)]
    strategies += [("голосование 30 соседей, вес = похожесть²", dict(temp=0))]
    for label, kw in strategies:
        ok1 = ok5 = 0
        for qi, true in enumerate(trues):
            c = [x for x, _ in vote(idx[qi], sim[qi], vb.cat, **kw)]
            ok1 += bool(c) and c[0] == true
            ok5 += true in c
        mark = "  <- сейчас" if kw.get("temp") == args.temp and "k" not in kw else ""
        print(f" {label:40s} |    {ok1 / n:6.1%}     |    {ok5 / n:6.1%}{mark}")

    print(f"\n=== Подробно для temp={args.temp} ===")
    print(f"Верная категория первой:            {acc('ok_vote'):6.1%}")
    print(f"Верная категория среди 5 вариантов: {acc('ok_top5'):6.1%}")
    print(f"Верный корневой раздел:             {acc('ok_root_vote'):6.1%}")

    print("\nЕсли брать без проверки только уверенные ответы (голосование):")
    print(" порог | доля таких товаров | точность у них")
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        sure = [r for r in rows if r["confidence"] >= thr]
        if sure:
            print(f"  {thr:.1f}  |      {len(sure) / n:6.1%}       |    "
                  f"{sum(r['ok_vote'] for r in sure) / len(sure):6.1%}")

    rep = write_csv(args.report, sorted(rows, key=lambda r: (r["ok_vote"], -r["confidence"])))
    print(f"\nВсе ответы (ошибки сверху): {rep.resolve()}")
    print("Важно: это наши чистые названия. На товарах продавцов точность будет ниже.")


# ---------------------------------------------------------------- predict

def full_description(text, max_len=6000):
    """Описание целиком (пробелы и переносы схлопываем).
    max_len - защита от очень длинных текстов: 6000 символов ~ 2000 токенов."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:max_len]


def first_sentences(text, min_len=40, max_len=300):
    """Первое предложение описания (если оно короче min_len - добавляем второе)."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = ""
    for p in parts:
        out = f"{out} {p}".strip()
        if len(out) >= min_len:
            break
    return out[:max_len]


def load_partner(path, name_col, desc_col):
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):     # {"offers": [...]} или один товар
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        items = []
        for o in data:
            params = o.get("param") or []
            if isinstance(params, dict):
                params = [params]
            items.append({
                "id": str(o.get("@id", o.get("id", ""))),
                "name": str(o.get(name_col) or ""),
                "vendor": str(o.get("vendor") or ""),
                "description": str(o.get(desc_col) or ""),
                "params": "; ".join(f"{x.get('@name')}: {x.get('#text')}"
                                    f"{(' ' + x['@unit']) if x.get('@unit') else ''}"
                                    for x in params if isinstance(x, dict)),
            })
        return items
    import csv
    with p.open(encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        return [{"id": r.get("id", r.get("@id", "")), "name": r.get(name_col, ""),
                 "vendor": r.get("vendor", ""), "description": r.get(desc_col, ""),
                 "params": ""} for r in csv.DictReader(f, dialect=dialect)]


def cmd_predict(args):
    items = load_partner(args.input, args.name_col, args.desc_col)
    if not items:
        raise SystemExit("В файле нет товаров.")
    if args.limit:
        print(f"В файле товаров: {len(items):,}, берём первые {args.limit}", flush=True)
        items = items[:args.limit]
    print(f"Товаров продавца: {len(items):,}", flush=True)
    vb = VectorBase(args.db)

    name_texts = [make_text(it["name"], it["vendor"]) for it in items]
    if args.desc_mode == "full":
        desc_starts = [full_description(it["description"], args.desc_max)
                       for it in items]
    else:
        desc_starts = [first_sentences(it["description"]) for it in items]
    print(f"Описание продавца: "
          f"{'целиком' if args.desc_mode == 'full' else 'только первое предложение'}",
          flush=True)
    full_texts = [norm(f"{n}. {d}") if d else n for n, d in zip(name_texts, desc_starts)]

    # два поиска: название + начало описания (основной) и только название
    q_full = embed_queries(full_texts, args, vb.con)
    q_name = embed_queries(name_texts, args, vb.con)
    idx_f, sim_f = vb.search(q_full, args.k)
    idx_n, sim_n = vb.search(q_name, args.k)

    rows, pairs, n_check, n_disagree = [], [], 0, 0
    for i, it in enumerate(items):
        cf = vote(idx_f[i], sim_f[i], vb.cat, temp=args.temp)
        cn = vote(idx_n[i], sim_n[i], vb.cat, temp=args.temp)
        best, conf = (cf[0] if cf else ("", 0.0))
        name_best = cn[0][0] if cn else ""
        disagree = bool(desc_starts[i]) and name_best != best
        reasons = []
        if conf < args.threshold:
            reasons.append(f"уверенность {conf:.2f} < {args.threshold}")
        if disagree:
            reasons.append("название и описание указывают на разные категории")
        n_check += bool(reasons)
        n_disagree += disagree
        rows.append({
            "id": it["id"],
            "name": it["name"],
            "description_used": desc_starts[i],
            "category_id": best,
            "category_path": vb.path(best) if best else "",
            "confidence": round(conf, 3),
            "needs_check": "да" if reasons else "нет",
            "why_check": "; ".join(reasons),
            "by_name_only_id": name_best,
            "by_name_only": vb.path(name_best) if name_best else "",
            "top5": " | ".join(f"[{c}] {vb.path(c)} ({s:.2f})" for c, s in cf),
            "nearest_items": " | ".join(
                f"{vb.name[j]} ({sim_f[i][n]:.2f})" for n, j in enumerate(idx_f[i][:5])),
            "params": it["params"],
        })

        # Пары "товар продавца - наш товар": по строке на каждого найденного соседа
        for search_name, idx, sim in (("название + описание", idx_f, sim_f),
                                      ("только название", idx_n, sim_n)):
            for rank, j in enumerate(idx[i][:args.pairs_n], start=1):
                pairs.append({
                    "partner_id": it["id"],
                    "partner_name": it["name"],
                    "partner_text": full_texts[i] if search_name.startswith("название +")
                                    else name_texts[i],
                    "search": search_name,
                    "rank": rank,
                    "similarity": round(float(sim[i][rank - 1]), 3),
                    "our_sku": vb.sku[j],
                    "our_name": vb.name[j],
                    "our_category_id": vb.cat[j],
                    "our_category": vb.path(vb.cat[j]),
                    "is_chosen_category": "да" if vb.cat[j] == best else "",
                })

    out_path = write_csv(args.output, rows)
    pairs_path = write_csv(
        Path(args.output).with_name(Path(args.output).stem + "_pairs.csv"), pairs)
    print(f"\nГотово: {len(rows):,} товаров")
    print(f"  нужно проверить вручную: {n_check:,} ({n_check / len(rows):.0%})")
    print(f"  из них название и описание расходятся: {n_disagree:,}")
    print(f"Результат: {out_path.resolve()}")
    print(f"Пары (товар продавца - наш товар): {pairs_path.resolve()}")


# ---------------------------------------------------------------- общее

def write_csv(path, rows):
    """Сохраняет CSV. Если файл занят (открыт в Excel) - пишет рядом
    под новым именем с временем и возвращает фактический путь."""
    import csv
    path = Path(path)
    if not rows:
        return path
    try:
        f = open(path, "w", encoding="utf-8-sig", newline="")
    except PermissionError:
        new = path.with_name(f"{path.stem}_{time.strftime('%H%M%S')}{path.suffix}")
        print(f"Файл {path.name} занят (открыт в Excel?). Сохраняю в {new.name}",
              flush=True)
        path, f = new, open(new, "w", encoding="utf-8-sig", newline="")
    with f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        w.writeheader()
        w.writerows(rows)
    return path

class RAGCategorySearcher:
    def __init__(self, db_path="rag.sqlite"):
        self.vb = VectorBase(db_path)

        self.args = SimpleNamespace(
            api_key=None,
            api_base=API_BASE,
            api_model=API_MODEL,
            batch=64,
            price=0.85,
        )

    def find_categories(self, title, description="", limit=5):
        description = re.sub(
            r"\s+", " ", str(description or "")
        ).strip()[:6000]

        query = norm(
            f"{title}. {description}" if description
            else str(title or "")
        )

        if not query.strip():
            return []

        qvec = embed_queries([query], self.args, self.vb.con)
        idx, sim = self.vb.search(qvec, k=30)

        ranked = vote(
            idx[0],
            sim[0],
            self.vb.cat,
            top_c=limit,
            temp=0.03,
        )

        return [
            {
                "category_id": int(cid),
                "category_path": self.vb.path(cid),
                "confidence": float(score),
            }
            for cid, score in ranked
        ]


# Один экземпляр на весь процесс
_rag_searcher = None


def find_best_categories(title, description="", limit=5):
    global _rag_searcher

    if _rag_searcher is None:
        # путь из config, а не "../data/...": так база находится
        # из какой угодно папки запуска (консоль, PyCharm, сайт)
        from config import DATA_DIR
        _rag_searcher = RAGCategorySearcher(str(DATA_DIR / "rag.sqlite"))

    return _rag_searcher.find_categories(
        title,
        description,
        limit,
    )



def main():
    ap = argparse.ArgumentParser(description="Поиск категории по векторам")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def api(p):
        p.add_argument("--db", default="rag.sqlite")
        p.add_argument("--k", type=int, default=30, help="сколько соседей голосуют")
        p.add_argument("--temp", type=float, default=0.03,
                       help="как быстро слабеет голос дальних соседей (меньше = быстрее)")
        p.add_argument("--api-base", default=API_BASE)
        p.add_argument("--api-model", default=API_MODEL)
        p.add_argument("--api-key", default=None)
        p.add_argument("--batch", type=int, default=64)
        p.add_argument("--price", type=float, default=0.85, help="₽ за 1 млн токенов")

    e = sub.add_parser("eval", help="проверить точность на наших товарах")
    api(e)
    e.add_argument("--per-cat", type=int, default=2, help="товаров на категорию")
    e.add_argument("--max-test", type=int, default=5000)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--report", default="eval_report.csv")

    p = sub.add_parser("predict", help="определить категории товаров продавца")
    api(p)
    p.add_argument("--input", required=True, help="JSON (как в примере) или CSV")
    p.add_argument("--output", default="result.csv")
    p.add_argument("--name-col", default="name")
    p.add_argument("--desc-col", default="description")
    p.add_argument("--desc-mode", choices=["full", "first"], default="full",
                   help="full - всё описание, first - только первое предложение")
    p.add_argument("--desc-max", type=int, default=6000,
                   help="обрезать описание длиннее стольких символов")
    p.add_argument("--pairs-n", type=int, default=10,
                   help="сколько наших похожих товаров показать на каждый товар продавца")
    p.add_argument("--limit", type=int, default=0,
                   help="взять только первые N товаров из файла (0 = все)")
    p.add_argument("--threshold", type=float, default=0.6,
                   help="ниже этой уверенности - на ручную проверку")

    args = ap.parse_args()
    {"eval": cmd_eval, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()