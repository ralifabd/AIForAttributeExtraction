# Проверка извлечённых атрибутов и уверенность по каждому значению (без эталона).
#
# Для каждого атрибута проверяем:
#   1. Есть ли значение в тексте товара (название + описание).
#      Нет в тексте -> модель, возможно, выдумала значение.
#   2. Встречается ли такое значение у наших товаров этой категории
#      (словарь значений из category_values.csv).
#   3. Для чисел - попадает ли в диапазон, который бывает у наших товаров.
# По результатам ставим confidence (0..1) и пояснение check.
#
# Словарь значений нужно один раз загрузить в базу (пункты 2 и 3; без него
# работает только проверка по тексту). Из папки notebooks:
#   py -3.12 -m quality.attr_check build --values путь/к/category_values.csv

import argparse
import csv
import re
import sqlite3
from collections import defaultdict
from html import unescape
from pathlib import Path

from rapidfuzz import fuzz

UNITS = {
    "мм": ("len", 1), "см": ("len", 10), "м": ("len", 1000),
    "г": ("mass", 1), "кг": ("mass", 1000),
    "мл": ("vol", 1), "л": ("vol", 1000),
    "вт": ("pow", 1), "квт": ("pow", 1000),
    "мб": ("mem", 1), "гб": ("mem", 1024), "тб": ("mem", 1024 * 1024),
    "mb": ("mem", 1), "gb": ("mem", 1024), "tb": ("mem", 1024 * 1024),
    "мач": ("cap", 1), "mah": ("cap", 1), "гц": ("freq", 1), "кгц": ("freq", 1000),
}
YES = {"да", "есть", "yes", "true", "поддерживается", "имеется", "присутствует"}
NO = {"нет", "no", "false", "отсутствует", "не поддерживается"}
NUM_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*([a-zа-я]+)?")


def norm(s):
    s = unescape(str(s or ""))
    s = re.sub(r"<[^>]+>", " ", s).lower().replace("ё", "е")
    return re.sub(r"\s+", " ", s).strip(" .;,")


def to_base(num, unit):
    """(1.7, 'л') -> (1700, 'vol'); без известной единицы -> (num, None)."""
    u = norm(unit)
    if u in UNITS:
        kind, k = UNITS[u]
        return num * k, kind
    return num, None


def numbers_in_text(text):
    out = []
    for m in NUM_RE.finditer(text):
        out.append(to_base(float(m.group(1).replace(",", ".")), m.group(2) or ""))
    return out


def stem(w):
    """Грубая основа слова: отрезаем окончание.
    "нержавеющая" -> "нержавею" (есть в "нержавеющей"), "черный" -> "черн" ("черного")."""
    if len(w) >= 8:
        return w[:-3]
    if len(w) >= 5:
        return w[:-2]
    return w


def stem_words(s):
    return [stem(w) for w in re.findall(r"[a-zа-я0-9]+", s)]


def compound_in_text(word, text_words):
    """Составное слово, записанное в тексте раздельно:
    "автоотключение" ~ "автоматического отключения"."""
    if len(word) < 8:
        return False
    for i in range(3, len(word) - 4):
        left, right = word[:i], stem(word[i:])
        if len(right) < 4:
            continue
        if (any(t.startswith(left) for t in text_words)
                and any(right in t for t in text_words)):
            return True
    return False


# ---------------------------------------------------------------- проверка по тексту

def check_in_text(name, value, unit, text):
    """Возвращает (оценка 0..1, пояснение)."""
    v = norm(value)
    if not v:
        return 0.0, "пустое значение"

    # да / нет: в тексте должно упоминаться само свойство
    if v in YES or v in NO:
        name_words = [w for w in re.findall(r"[a-zа-я0-9]+", norm(name)) if len(w) >= 3]
        stems = [stem(w) for w in name_words]
        text_words = re.findall(r"[a-zа-я0-9]+", text)
        if stems and all(st in text for st in stems):
            return 0.85, "свойство упоминается в тексте"
        if name_words and all(stem(w) in text or compound_in_text(w, text_words)
                              for w in name_words):
            return 0.8, "свойство упоминается в тексте (другими словами)"
        if fuzz.partial_ratio(norm(name), text) >= 85:
            return 0.75, "свойство похоже упоминается в тексте"
        if any(st in text for st in stems if len(st) >= 4):
            return 0.55, "свойство частично упоминается в тексте"
        return 0.3, "свойство в тексте не упоминается"

    # числа: ищем то же число в тексте (с переводом единиц)
    m = NUM_RE.search(v)
    if m and len(m.group(0)) >= len(v) * 0.5:
        num, kind = to_base(float(m.group(1).replace(",", ".")), m.group(2) or unit or "")
        same_number = False
        for tn, tk in numbers_in_text(text):
            if abs(tn - num) <= max(abs(num) * 0.02, 1e-9):
                if kind and tk and kind == tk:
                    return 0.95, "число с единицей найдено в тексте"
                same_number = True
        if same_number:
            return 0.8, "число найдено в тексте"
        return 0.3, "числа нет в тексте"

    # текст: значение или его основа есть в тексте
    if v in text:
        return 0.9, "значение найдено в тексте"
    words = [w for w in stem_words(v) if len(w) >= 3]
    if words and all(w in text for w in words):
        return 0.85, "значение найдено в тексте (в другой форме)"
    if fuzz.partial_ratio(v, text) >= 85:
        return 0.7, "похожее значение есть в тексте"
    if words and any(w in text for w in words):
        return 0.55, "в тексте есть часть значения"
    return 0.3, "значения нет в тексте"


# ---------------------------------------------------------------- словарь значений

class AttributeChecker:
    def __init__(self, db_path=None):
        self.values = defaultdict(dict)   # (category_id, param) -> {norm value: count}
        self.ranges = {}                  # (category_id, param) -> (min, max, kind)
        self.has_dict = False
        if db_path and Path(db_path).is_file():
            con = sqlite3.connect(db_path)
            try:
                rows = con.execute(
                    "SELECT category_id, parameter, unit, value, product_count "
                    "FROM attr_values").fetchall()
            except sqlite3.OperationalError:
                rows = []
            con.close()
            nums = defaultdict(list)
            for cid, p, u, v, n in rows:
                key = (str(cid), p.casefold())
                self.values[key][norm(v)] = n
                m = NUM_RE.fullmatch(norm(v)) or NUM_RE.match(norm(v))
                if m:
                    nums[key].append(to_base(float(m.group(1).replace(",", ".")),
                                             m.group(2) or u or ""))
            for key, lst in nums.items():
                vals = [x for x, _ in lst]
                kind = next((k for _, k in lst if k), None)
                self.ranges[key] = (min(vals), max(vals), kind)
            self.has_dict = bool(rows)

    def _key(self, category_id, name):
        """Атрибут схемы -> характеристика каталога: "Имя_search", затем "Имя"."""
        cid = str(category_id)
        for p in (f"{name}_search", name):
            k = (cid, p.casefold())
            if k in self.values:
                return k
        return None

    def check(self, param, title, description, category_id=None):
        """param: {"@name", "#text", "@unit"} -> {"confidence", "check"}."""
        name, value, unit = param.get("@name", ""), param.get("#text", ""), param.get("@unit", "")
        text = norm(f"{title} {description}")
        score, reason = check_in_text(name, value, unit, text)
        notes = [reason]

        key = self._key(category_id, name) if (self.has_dict and category_id) else None
        if key:
            known = self.values[key]
            v = norm(value)
            if v in known:
                score += 0.05
                notes.append(f"значение встречается у наших товаров ({known[v]} шт.)")
            elif key in self.ranges:
                m = NUM_RE.search(v)
                if m:
                    num, kind = to_base(float(m.group(1).replace(",", ".")),
                                        m.group(2) or unit or "")
                    lo, hi, rkind = self.ranges[key]
                    if rkind and kind and rkind != kind:
                        score -= 0.2
                        notes.append("единица не такая, как у наших товаров")
                    elif not (lo * 0.5 <= num <= hi * 2):
                        score -= 0.3
                        notes.append("число вне обычного диапазона для категории")
            else:
                best = max((fuzz.ratio(v, k) for k in known), default=0)
                if best < 70:
                    score -= 0.1
                    notes.append("такое значение у наших товаров не встречалось")

        score = round(max(0.0, min(score, 0.99)), 2)
        return {"confidence": score, "check": "; ".join(notes)}


def level(score):
    if score >= 0.8:
        return "надёжно"
    if score >= 0.5:
        return "проверить"
    return "сомнительно"


def summarize(params, core_names=None):
    """Сводка по всем атрибутам товара - для вывода внизу результата.
    core_names - названия основных (core) атрибутов схемы категории."""
    scores = [p["confidence"] for p in params if isinstance(p.get("confidence"), (int, float))]
    levels = defaultdict(int)
    for p in params:
        if "confidence" in p:
            levels[level(p["confidence"])] += 1
    out = {
        "attributes_total": len(params),
        "avg_confidence": round(sum(scores) / len(scores), 2) if scores else None,
        "reliable": levels["надёжно"],
        "to_check": levels["проверить"],
        "doubtful": levels["сомнительно"],
        "doubtful_attributes": [p["@name"] for p in params
                                if "confidence" in p and level(p["confidence"]) == "сомнительно"],
    }
    if core_names:
        have = {str(p.get("@name", "")).strip().casefold() for p in params}
        core = {str(n).strip().casefold() for n in core_names}
        filled = len(core & have)
        out["core_total"] = len(core)
        out["core_filled"] = filled
        out["core_coverage"] = round(filled / len(core), 2) if core else None
        out["core_missing"] = sorted(n for n in core_names
                                     if str(n).strip().casefold() not in have)
    return out


# ---------------------------------------------------------------- загрузка словаря значений

def cmd_build(args):
    src, db = Path(args.values), Path(args.db)
    if not src.is_file():
        raise SystemExit(f"Не найден файл: {src.resolve()}")
    con = sqlite3.connect(db)
    con.executescript("""
        DROP TABLE IF EXISTS attr_values;
        CREATE TABLE attr_values (category_id TEXT, parameter TEXT, unit TEXT,
                                  value TEXT, product_count INTEGER);
    """)
    n = 0
    with src.open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        batch = []
        for row in r:
            batch.append((row["category_id"], row["parameter"], row.get("unit", ""),
                          row["value"], int(row.get("product_count") or 0)))
            if len(batch) >= 50000:
                con.executemany("INSERT INTO attr_values VALUES (?, ?, ?, ?, ?)", batch)
                n += len(batch)
                batch = []
        con.executemany("INSERT INTO attr_values VALUES (?, ?, ?, ?, ?)", batch)
        n += len(batch)
    con.execute("CREATE INDEX IF NOT EXISTS av_key ON attr_values(category_id, parameter)")
    con.commit()
    con.close()
    print(f"Загружено значений: {n:,} -> таблица attr_values в {db.resolve()}")


def main():
    try:
        from config import DATA_DIR
    except ImportError:
        DATA_DIR = Path("data")
    ap = argparse.ArgumentParser(description="Словарь значений атрибутов")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="загрузить category_values.csv в базу")
    b.add_argument("--values", default=str(DATA_DIR / "category_values.csv"))
    b.add_argument("--db", default=str(DATA_DIR / "rag.sqlite"))
    args = ap.parse_args()
    cmd_build(args)


if __name__ == "__main__":
    main()
