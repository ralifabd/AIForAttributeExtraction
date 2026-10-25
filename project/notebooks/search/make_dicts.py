import argparse
import csv
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------- нормализация

TOKEN_RE = re.compile(r"[0-9a-zа-я]+|\+")
TOKEN_ORIG_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+|\+")

# "бренды", которые на самом деле означают отсутствие бренда
NO_BRAND_KEYS = {"", "нетбренда", "безбренда", "noname", "nobrand", "безмарки",
                 "китай", "china", "другое", "прочее", "unknown", "generic"}

# слова, которые не могут быть линейкой бренда (слишком общие)
GENERIC_LINE_WORDS = {
    "pro", "max", "mini", "plus", "air", "lite", "light", "smart", "series", "classic",
    "home", "kids", "baby", "new", "set", "kit", "box", "mix", "color", "style",
    "premium", "comfort", "standard", "eco", "soft", "hard", "ultra", "super", "power",
    "black", "white", "grey", "gray", "silver", "gold", "blue", "red", "green", "pink",
    "the", "for", "and", "with", "led", "usb", "wifi", "bluetooth", "type",
}

# память / объём: всё, что после этого в модели - вариант (цвет, eSim и т.п.)
MEMORY_RE = re.compile(
    r"\b\d+\s*/\s*\d+\s*(gb|гб|tb|тб)\b|\b\d+\s*(gb|гб|tb|тб)\b"
    r"|\b\d{1,2}\s*/\s*(16|32|64|128|256|512|1024)\b",       # 8/256, 12/512
    re.IGNORECASE)

EN_COLORS = {
    "black", "white", "silver", "gold", "grey", "gray", "blue", "red", "green", "pink",
    "purple", "yellow", "orange", "graphite", "midnight", "starlight", "titanium",
    "natural", "desert", "lavender", "sage", "mist", "teal", "ultramarine", "coral",
    "cream", "violet", "mint", "navy", "beige", "brown", "rose", "space", "jet",
    "черный", "белый", "серый", "серебристый", "золотой", "золотистый", "синий",
    "красный", "зеленый", "розовый", "фиолетовый", "желтый", "оранжевый", "голубой",
    "бежевый", "коричневый", "мятный", "графитовый", "титановый", "бирюзовый",
}


def norm_tokens(s):
    return TOKEN_RE.findall(str(s or "").lower().replace("ё", "е"))


def key(s):
    """Ключ для сравнения: только буквы/цифры, нижний регистр.
    De'Longhi, DeLonghi, DELONGHI, De Longhi -> delonghi"""
    return "".join(norm_tokens(s))


def model_base(model, colors):
    """Базовая модель: без памяти, цвета, скобок.
    'iPhone 17 Pro 256GB Silver eSim (без RuStore)' -> 'iPhone 17 Pro'
    'Galaxy S25 Ultra 12/256GB Titanium Gray'      -> 'Galaxy S25 Ultra'"""
    m = re.sub(r"\(.*?\)", " ", str(model or ""))
    cut = MEMORY_RE.search(m)
    if cut:
        m = m[:cut.start()]
    words = m.split()
    while words and (words[-1].lower().strip(",.;") in colors or
                     words[-1].lower() in {"esim", "dual", "sim", "nano-sim", "5g", "4g"}):
        words.pop()
    return " ".join(words).strip(" ,.-/")


# ---------------------------------------------------------------- build

def load_colors(con):
    """Названия цветов: из характеристик каталога (если есть) + английские."""
    colors = set(EN_COLORS)
    try:
        for (v,) in con.execute(
                "SELECT DISTINCT value FROM params WHERE parameter IN ('Цвет','Цвет_search')"):
            for part in re.split(r"[/,;]", v or ""):
                part = part.strip().lower().replace("ё", "е")
                if part and len(part.split()) == 1:
                    colors.add(part)
    except sqlite3.OperationalError:
        pass  # в базе нет таблицы params - обойдёмся английскими
    return colors


def cmd_build(args):
    src = Path(args.db)
    if not src.is_file():
        raise SystemExit(f"Не найден файл: {src.resolve()}")
    con = sqlite3.connect(src)
    print("Читаю каталог...", flush=True)
    rows = con.execute("""
        SELECT p.vendor, p.model, p.name, pc.category_id
        FROM products AS p JOIN product_categories AS pc USING (product_no)
    """).fetchall()
    colors = load_colors(con)
    con.close()
    print(f"Товаров: {len(rows):,}", flush=True)

    # 1) бренды: группируем написания по ключу, единое написание = самое частое
    spellings = defaultdict(Counter)
    for vendor, _, _, _ in rows:
        k = key(vendor)
        if k not in NO_BRAND_KEYS:
            spellings[k][str(vendor).strip()] += 1
    brand_of_key = {}
    for k, cnt in spellings.items():
        brand_of_key[k] = max(cnt.items(), key=lambda x: (x[1], x[0] != x[0].upper()))[0]

    # ручные варианты (brand_aliases_manual.csv). Если вариант - это бренд из
    # каталога ("Xiaomi Redmi;Xiaomi"), его товары и модели переходят к целевому бренду.
    manual = []
    mpath = Path(args.manual)
    if mpath.is_file():
        with mpath.open(encoding="utf-8-sig") as f:
            for r in csv.reader(f, delimiter=";"):
                if len(r) >= 2 and r[0].strip() and r[1].strip() and r[0] != "alias":
                    target = brand_of_key.get(key(r[1]), r[1].strip())
                    manual.append((key(r[0]), target, r[0].strip()))
        for k, target, _ in manual:
            if k in brand_of_key:
                brand_of_key[k] = target
        print(f"Ручных вариантов из {mpath.name}: {len(manual)}", flush=True)

    brand_count = Counter()
    for k, cnt in spellings.items():
        brand_count[brand_of_key[k]] += sum(cnt.values())
    brands = [(b, key(b), n) for b, n in brand_count.items()]
    print(f"Брендов: {len(brands):,} (вариантов написания было "
          f"{sum(len(c) for c in spellings.values()):,})", flush=True)

    # 2) бренд x категория, модели, написание слов
    brand_cat = Counter()
    models = defaultdict(lambda: Counter())   # (brand, cat, base) -> Counter(full model)
    line_votes = defaultdict(Counter)         # line_key -> Counter(brand)
    line_case = defaultdict(Counter)
    case_by_brand = defaultdict(lambda: defaultdict(Counter))
    for vendor, model, name, cid in rows:
        k = key(vendor)
        if k not in brand_of_key:
            continue
        b = brand_of_key[k]
        brand_cat[(b, cid)] += 1
        model = str(model or "").strip()
        if not model or key(model) == k:
            continue
        base = model_base(model, colors) or model
        models[(b, cid, base)][model] += 1
        for t in TOKEN_ORIG_RE.findall(model):
            case_by_brand[b][t.lower().replace("ё", "е")][t] += 1
        first = TOKEN_ORIG_RE.findall(model)[:1]
        if first:
            f = first[0]
            fk = key(f)
            ntoks = norm_tokens(name)
            pos = ntoks.index(fk) if fk in ntoks else -1
            if pos > 0 and ntoks[pos - 1] in ("для", "for", "под"):
                continue                  # "Чехол для Galaxy S25" - не голос за бренд
            if (len(fk) >= 3 and re.search("[a-z]", fk) and not re.search(r"\d", fk)
                    and fk not in GENERIC_LINE_WORDS):
                line_votes[fk][b] += 1
                line_case[fk][f] += 1

    # 3) линейки, которые однозначно указывают на бренд (iphone -> Apple)
    line_aliases = []
    for lk, votes in line_votes.items():
        total = sum(votes.values())
        b, n = votes.most_common(1)[0]
        if (n >= args.line_min and n / total >= args.line_share
                and brand_of_key.get(lk) != b):
            line_aliases.append((lk, b, line_case[lk].most_common(1)[0][0], n, n / total))
    print(f"Линеек, однозначно указывающих на бренд: {len(line_aliases):,}", flush=True)


    # ---- запись
    out = sqlite3.connect(args.out)
    out.executescript("""
        DROP TABLE IF EXISTS brands;
        DROP TABLE IF EXISTS brand_aliases;
        DROP TABLE IF EXISTS brand_categories;
        DROP TABLE IF EXISTS models;
        DROP TABLE IF EXISTS token_case;
        CREATE TABLE brands (brand TEXT PRIMARY KEY, brand_key TEXT, product_count INTEGER);
        CREATE TABLE brand_aliases (
            alias_key TEXT, brand TEXT,
            alias_type TEXT,      -- написание / линейка / ручной
            example TEXT,         -- как это выглядит в каталоге
            support INTEGER,      -- сколько товаров подтверждают
            share REAL            -- доля товаров с этой линейкой у этого бренда
        );
        CREATE TABLE brand_categories (brand TEXT, category_id TEXT, product_count INTEGER);
        CREATE TABLE models (
            brand TEXT, category_id TEXT,
            model_base TEXT,      -- без памяти и цвета: iPhone 17 Pro
            model_base_key TEXT,
            model_example TEXT,   -- самый частый полный вариант из каталога
            variants INTEGER,     -- сколько разных полных вариантов
            product_count INTEGER
        );
        CREATE TABLE token_case (brand TEXT, token_key TEXT, token TEXT);
        DROP TABLE IF EXISTS colors;
        CREATE TABLE colors (color TEXT);
    """)
    out.executemany("INSERT INTO colors VALUES (?)", [(c,) for c in sorted(colors)])
    out.executemany("INSERT INTO brands VALUES (?, ?, ?)", brands)
    alias_rows = []
    for k, cnt in spellings.items():
        for sp, n in cnt.items():
            alias_rows.append((k, brand_of_key[k], "написание", sp, n, 1.0))
    alias_rows += [(lk, b, "линейка", ex, n, sh) for lk, b, ex, n, sh in line_aliases]
    alias_rows += [(k, b, "ручной", ex, 0, 1.0) for k, b, ex in manual]
    out.executemany("INSERT INTO brand_aliases VALUES (?, ?, ?, ?, ?, ?)", alias_rows)
    out.executemany("INSERT INTO brand_categories VALUES (?, ?, ?)",
                    [(b, c, n) for (b, c), n in brand_cat.items()])
    out.executemany("INSERT INTO models VALUES (?, ?, ?, ?, ?, ?, ?)", [
        (b, c, base, key(base), cnt.most_common(1)[0][0], len(cnt), sum(cnt.values()))
        for (b, c, base), cnt in models.items()])
    out.executemany("INSERT INTO token_case VALUES (?, ?, ?)", [
        (b, tk, c.most_common(1)[0][0])
        for b, d in case_by_brand.items() for tk, c in d.items()])
    out.executescript("""
        CREATE INDEX IF NOT EXISTS ba_key ON brand_aliases(alias_key);
        CREATE INDEX IF NOT EXISTS bc_brand ON brand_categories(brand, category_id);
        CREATE INDEX IF NOT EXISTS m_brand ON models(brand, model_base_key);
    """)
    out.commit()
    n_models = out.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    n_base = out.execute(
        "SELECT COUNT(DISTINCT brand || '|' || model_base_key) FROM models").fetchone()[0]
    out.close()
    print("\n=== Словари готовы ===")
    print(f"Брендов: {len(brands):,}")
    print(f"Вариантов написания бренда: {len(alias_rows) - len(line_aliases) - len(manual):,}")
    print(f"Линеек -> бренд: {len(line_aliases):,}  (примеры: " + ", ".join(
        f"{ex}->{b}" for lk, b, ex, n, sh in sorted(line_aliases, key=lambda x: -x[3])[:8]) + ")")
    print(f"Моделей (бренд + категория + базовая модель): {n_models:,}; "
          f"уникальных базовых моделей: {n_base:,}")
    print(f"Записано в: {Path(args.out).resolve()}")


# ---------------------------------------------------------------- определение

class BrandModelExtractor:
    """Находит бренд и модель в тексте товара по словарям из rag.sqlite."""

    def __init__(self, db):
        con = sqlite3.connect(db)
        try:
            con.execute("SELECT 1 FROM brand_aliases LIMIT 1")
        except sqlite3.OperationalError:
            raise SystemExit("В rag.sqlite нет словарей. Сначала: make_dicts.py build")
        self.alias = defaultdict(list)   # alias_key -> [(brand, type, support)]
        for k, b, t, sup in con.execute(
                "SELECT alias_key, brand, alias_type, support FROM brand_aliases"):
            self.alias[k].append((b, t, sup))
        # ручные варианты важнее всего: "Xiaomi Redmi;Xiaomi" перекрывает бренд Xiaomi Redmi
        for k, lst in self.alias.items():
            manual = [x for x in lst if x[1] == "ручной"]
            if manual:
                self.alias[k] = manual
        try:
            self.colors = {c for (c,) in con.execute("SELECT color FROM colors")}
        except sqlite3.OperationalError:
            self.colors = set(EN_COLORS)
        self.max_alias_tokens = 4
        self.brand_total = dict(con.execute("SELECT brand, product_count FROM brands"))
        self.brand_cat = defaultdict(dict)
        for b, c, n in con.execute("SELECT brand, category_id, product_count "
                                   "FROM brand_categories"):
            self.brand_cat[b][c] = n
        self.models = defaultdict(list)  # brand -> [(base_key, base, category, count)]
        for b, c, base, bk, n in con.execute(
                "SELECT brand, category_id, model_base, model_base_key, product_count "
                "FROM models"):
            if len(bk) >= 2:
                self.models[b].append((bk, base, c, n))
        self.model_index = {}            # brand -> {base_key: [(base, category, count)]}
        for b, lst in self.models.items():
            d = defaultdict(list)
            for bk, base, c, n in lst:
                d[bk].append((base, c, n))
            self.model_index[b] = d
        self.case = defaultdict(dict)
        for b, tk, t in con.execute("SELECT brand, token_key, token FROM token_case"):
            self.case[b][tk] = t
        con.close()

    # ---- бренд

    def _find_brands(self, text):
        """Все совпадения с вариантами написания: (start, end, brand, type, support)."""
        toks = norm_tokens(text)
        found = []
        for i in range(len(toks)):
            for L in range(self.max_alias_tokens, 0, -1):
                if i + L > len(toks):
                    continue
                k = "".join(toks[i:i + L])
                if k in self.alias:
                    for b, t, sup in self.alias[k]:
                        # очень короткие варианты (LG, ЭРА, Луч) - только целым словом
                        if len(k) <= 2 and L > 1:
                            continue
                        found.append((i, i + L, b, t, sup))
                    break
        return toks, found

    def brand(self, name, description="", category_id=None):
        candidates = []
        for where, text, base_conf in (("название", name, 0.95),
                                       ("описание", description, 0.75)):
            toks, found = self._find_brands(text)
            for s, e, b, t, sup in found:
                conf = base_conf - (0.05 if t == "линейка" else 0)
                notes = []
                if t == "написание" and len("".join(toks[s:e])) <= 3:
                    conf -= 0.15          # короткие бренды часто совпадают случайно
                in_cat = self.brand_cat[b].get(category_id, 0) if category_id else None
                if category_id and not in_cat:
                    # бренд ни разу не встречался в этой категории
                    conf -= 0.3 if t == "линейка" else 0.2
                    notes.append("бренд не встречался в этой категории")
                if s > 0 and toks[s - 1] in ("для", "for", "под"):
                    # "Чехол для iPhone 15": это совместимость, а не бренд товара
                    conf -= 0.45
                    notes.append(f"перед ним «{toks[s - 1]}» - скорее совместимость")
                candidates.append({"brand": b, "conf": conf, "where": where, "type": t,
                                   "start": s, "end": e, "in_category": in_cat,
                                   "tokens": toks, "notes": notes})
        if not candidates:
            return self._unknown_brand(name)
        # лучший: выше уверенность, затем больше товаров бренда в этой категории,
        # затем раньше в тексте. Если одно и то же слово - и бренд, и линейка
        # другого бренда (Galaxy), решает категория.
        snapshot = list(candidates)       # во время sort() сам список выглядит пустым
        def rank(c):
            same_span = [x for x in snapshot
                         if x["where"] == c["where"] and x["start"] == c["start"]]
            cat_bonus = 0.0
            if category_id and len(same_span) > 1:
                total = sum((x["in_category"] or 0) for x in same_span) or 1
                cat_bonus = 0.2 * (c["in_category"] or 0) / total
            return (-(c["conf"] + cat_bonus), -(c["in_category"] or 0), c["start"])
        candidates.sort(key=rank)
        best = candidates[0]
        best_spans = {(c["where"], c["start"]) for c in candidates
                      if c["brand"] == best["brand"]}
        others = {c["brand"] for c in candidates
                  if c["where"] == "название" and c["brand"] != best["brand"]
                  and c["conf"] > 0.5
                  and (c["where"], c["start"]) not in best_spans
                  and not (c["start"] < best["end"] and best["start"] < c["end"])}
        if others:
            best["conf"] -= 0.15          # в названии несколько разных брендов
            best["notes"].append("ещё в названии: " + ", ".join(sorted(others)))
        ambiguous = {c["brand"] for c in candidates if c["where"] == best["where"]
                     and c["start"] == best["start"] and c["brand"] != best["brand"]}
        if ambiguous:
            best["conf"] -= 0.05
            best["notes"].append("это слово есть и у: " + ", ".join(sorted(ambiguous)))
        best["conf"] = round(max(0.0, min(best["conf"], 0.99)), 2)
        return best

    def _unknown_brand(self, name):
        """Бренда нет в словаре. Кандидат: первое слово латиницей в названии
        ("Детские столовые приборы GSS" -> GSS). Уверенность низкая -> LLM."""
        orig = TOKEN_ORIG_RE.findall(str(name or ""))
        toks = [t.lower().replace("ё", "е") for t in orig]
        for i, t in enumerate(orig):
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", t) and len(t) >= 2 \
                    and t.lower() not in GENERIC_LINE_WORDS:
                if i > 0 and toks[i - 1] in ("для", "for", "под"):
                    continue
                return {"brand": t if not t.islower() else t.capitalize(),
                        "conf": 0.3, "where": "название", "type": "нет в словаре",
                        "start": i, "end": i + 1, "in_category": None, "tokens": toks,
                        "notes": ["похоже на новый бренд, проверить"]}
        return None

    # ---- модель

    def _pretty(self, brand, tokens):
        out = []
        for t in tokens:
            tk = t.lower().replace("ё", "е")
            out.append(self.case[brand].get(tk) or (t if not t.isupper() or len(t) <= 4
                                                    else t.capitalize()))
        s = " ".join(out)
        return re.sub(r"\s+\+", "+", s)

    def model(self, brand_info, name, category_id=None):
        b = brand_info["brand"]
        toks = norm_tokens(name)
        idx = self.model_index.get(b, {})
        # модель ищем только после бренда (для линейки - начиная с неё самой):
        # "Смартфон Apple iPhone 17 Pro" -> ищем в "iPhone 17 Pro"
        first = 0
        if brand_info["where"] == "название":
            first = brand_info["start"] if brand_info["type"] == "линейка" \
                else brand_info["end"]
        # 1) известная модель: ищем самую длинную последовательность слов из названия,
        #    которая совпадает с базовой моделью бренда
        best = None
        for i in range(first, len(toks)):
            for L in range(min(8, len(toks) - i), 0, -1):
                k = "".join(toks[i:i + L])
                if len(k) < 3 or k not in idx:
                    continue
                if not re.search(r"\d", k) and len(k) < 5:
                    continue              # "pro", "max" сами по себе не модель
                hits = idx[k]
                in_cat = [h for h in hits if h[1] == category_id]
                pick = max(in_cat or hits, key=lambda h: h[2])
                if best is None or len(k) > len(best[0]):
                    best = (k, pick, bool(in_cat))
                break
        if best:
            k, (base, cid, n), in_cat = best
            conf = 0.95 if in_cat or not category_id else 0.8
            return {"model": base, "conf": conf,
                    "status": "известная модель" if in_cat or not category_id
                    else "известная модель, но из другой категории"}

        # 2) неизвестная модель: берём слова после бренда/линейки в названии
        orig = TOKEN_ORIG_RE.findall(str(name or ""))
        if brand_info["where"] != "название" or len(orig) != len(brand_info["tokens"]):
            return {"model": "", "conf": 0.0, "status": "не найдена"}
        start = brand_info["start"] if brand_info["type"] == "линейка" else brand_info["end"]
        rest = orig[start:]
        cut = MEMORY_RE.search(" ".join(rest))
        if cut:
            rest = " ".join(rest)[:cut.start()].split()
        while rest and rest[-1].lower().replace("ё", "е") in self.colors:
            rest = rest[:-1]              # "S26+ черный" -> "S26+"
        if not rest or not any(re.search(r"[0-9A-Za-z]", t) for t in rest):
            return {"model": "", "conf": 0.0, "status": "не найдена"}
        pretty = self._pretty(b, rest)
        if brand_info["type"] == "линейка":
            return {"model": pretty, "conf": 0.6,
                    "status": "новая модель известной линейки"}
        if brand_info["type"] == "нет в словаре":
            return {"model": pretty, "conf": 0.3, "status": "бренд и модель не в словаре"}
        return {"model": pretty, "conf": 0.4, "status": "нет в словаре"}

    # ---- всё вместе

    def extract(self, name, description="", category_id=None, threshold=0.7):
        bi = self.brand(name, description, category_id)
        if not bi:
            return {"brand": "", "brand_confidence": 0.0, "brand_found_by": "",
                    "model": "", "model_confidence": 0.0, "model_status": "",
                    "brand_model_needs_llm": "да"}
        mi = self.model(bi, name, category_id)
        how = f"{bi['type']} в {'названии' if bi['where'] == 'название' else 'описании'}"
        if bi.get("notes"):
            how += "; " + "; ".join(bi["notes"])
        return {
            "brand": bi["brand"],
            "brand_confidence": bi["conf"],
            "brand_found_by": how,
            "model": mi["model"],
            "model_confidence": mi["conf"],
            "model_status": mi["status"],
            "brand_model_needs_llm":
                "да" if bi["conf"] < threshold or mi["conf"] < threshold else "нет",
        }


def cmd_test(args):
    ex = BrandModelExtractor(args.out)
    r = ex.extract(args.name, args.description, args.category_id)
    for k, v in r.items():
        print(f"{k:24s} {v}")


def main():
    ap = argparse.ArgumentParser(description="Словари брендов и моделей")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="собрать словари из catalog.sqlite в rag.sqlite")
    b.add_argument("--db", default="catalog_analysis/catalog.sqlite")
    b.add_argument("--out", default="rag.sqlite")
    b.add_argument("--manual", default="brand_aliases_manual.csv")
    b.add_argument("--line-min", type=int, default=10,
                   help="линейка должна встречаться хотя бы у стольких товаров")
    b.add_argument("--line-share", type=float, default=0.85,
                   help="и такая доля из них должна быть одного бренда")
    t = sub.add_parser("test", help="проверить на одном товаре")
    t.add_argument("--out", default="rag.sqlite")
    t.add_argument("--name", required=True)
    t.add_argument("--description", default="")
    t.add_argument("--category-id", default=None)
    args = ap.parse_args()
    {"build": cmd_build, "test": cmd_test}[args.cmd](args)


if __name__ == "__main__":
    main()
