import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPORTS = DATA / "reports"
OUTPUT = DATA / "ideal_attribute_schema.json"
TEMP = DATA / "duckdb_tmp_ideal_schema"

sys.path.insert(0, str(ROOT / "notebooks"))
from normalization.dictionary.attribute_aliases import ATTRIBUTE_ALIASES

W_SEARCH = 0.45
W_COMPARISON = 0.35
W_EXTRACTION = 0.20

MAX_ATTRIBUTES_PER_CATEGORY = 80
MIN_COVERAGE = 0.01
MEMORY_LIMIT = "4GB"
THREADS = 2

# Поисковые формулировки — прокси пользовательского интереса.
# Это не клики и не конверсии.
INTENTS = {
    "dimensions": r"узк|ширин|глубин|высот|под\s+раковин",
    "capacity": r"\b[3-9]\s*(?:кг|килограмм)",
    "drying": r"с\s+сушк|сушильн|стирально.?сушильн",
    "loading_type": r"вертикальн(?:ая|ой)?\s+загрузк",
    "spin": r"оборотов|об/мин|скорост.{0,15}отжим",
    "energy": r"энергоэффектив|класс\s+[a-g]\b",
    "noise": r"уровень\s+шума|тихая|тихую",
    "motor": r"инверторн.{0,15}мотор|прямой\s+привод",
    "installation": r"встраиваем|под\s+столешниц",
}

# Служебные / коммерческие / не технические атрибуты.
EXCLUDE_PATTERNS = [
    r"^продавец$",
    r"^подборки?$",
    r"потратить бонус",
    r"топ коллекци",
    r"^sku$",
    r"^артикул$",
    r"^код товара$",
    r"^идентификатор$",
    r"^состояние$",
    r"^дата добавления$",
    r"^дата обновления$",
]

BOOL_VALUES = {
    "да", "нет", "true", "false", "есть", "отсутствует",
    "yes", "no", "1", "0",
}


def log(message):
    print(message, flush=True)


def clean(value):
    if value is None:
        return ""
    return str(value).strip()


def canonical_name(name):
    """Удаляем _search только как имя источника, не как правило эквивалентности."""
    original = clean(name)
    stripped = re.sub(r"_search$", "", original, flags=re.IGNORECASE).strip()

    aliases = {
        clean(k).casefold(): clean(v)
        for k, v in ATTRIBUTE_ALIASES.items()
    }
    return aliases.get(stripped.casefold(), stripped)


def infer_type(values, units):
    vals = [clean(v) for v in values if clean(v)]
    unit_set = {clean(u).casefold() for u in units if clean(u)}

    if vals and all(v.casefold() in BOOL_VALUES for v in vals[:100]):
        return "boolean"

    if unit_set:
        return "number_with_unit"

    numeric = [
        re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?", v.replace(" ", ""))
        for v in vals[:100]
    ]
    if vals and all(numeric):
        return "number"

    distinct = {v.casefold() for v in vals}
    if 0 < len(distinct) <= 30:
        return "enum"

    return "string"


def is_excluded(name):
    value = clean(name).casefold()
    return any(re.search(pattern, value) for pattern in EXCLUDE_PATTERNS)


def intent_for_attribute(name):
    """Возвращает поисковые намерения, к которым относится название поля."""
    n = clean(name).casefold()
    result = []

    if re.search(r"ширин|глубин|высот|габарит|размер", n):
        result.append("dimensions")
    if re.search(r"загрузк|вместимост", n):
        result.append("capacity")
    if re.search(r"сушк", n):
        result.append("drying")
    if re.search(r"тип загрузки|вертикальн", n):
        result.append("loading_type")
    if re.search(r"оборот|отжим", n):
        result.append("spin")
    if re.search(r"энергопотреб|энергоэффектив|класс энерг", n):
        result.append("energy")
    if re.search(r"шум", n):
        result.append("noise")
    if re.search(r"мотор|двигател|прямой привод", n):
        result.append("motor")
    if re.search(r"установк|встраиваем", n):
        result.append("installation")

    return result


def bounded_log_score(value, scale=100):
    if value <= 0:
        return 0.0
    return min(1.0, math.log1p(value) / math.log1p(scale))


def main():
    attrs_path = DATA / "attrs.parquet"
    catalog_path = REPORTS / "catalog_master.parquet"
    query_path = DATA / "query_clicks.parquet"

    for path in (attrs_path, catalog_path, query_path):
        if not path.exists():
            raise FileNotFoundError(f"Не найден файл: {path}")

    TEMP.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
    con.execute(f"SET threads = {THREADS}")
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET enable_progress_bar = true")
    con.execute("SET temp_directory = '" + str(TEMP) + "'")

    attrs_sql = "read_parquet(?)"
    catalog_sql = "read_parquet(?)"

    try:
        log("1/7. Подготовка списка исходных атрибутов...")

        # Здесь читается только небольшой список уникальных названий.
        source_names = con.execute(
            f"""
            SELECT DISTINCT trim(CAST(attr AS VARCHAR)) AS source_attr
            FROM {attrs_sql}
            WHERE attr IS NOT NULL
              AND trim(CAST(attr AS VARCHAR)) <> ''
            """,
            [str(attrs_path)],
        ).fetchall()

        mapping = []
        for (source_name,) in source_names:
            source_name = clean(source_name)
            if source_name:
                mapping.append((source_name, canonical_name(source_name)))

        con.execute("""
            CREATE TEMP TABLE attr_map (
                source_attr VARCHAR,
                canonical_attr VARCHAR
            )
        """)
        con.executemany(
            "INSERT INTO attr_map VALUES (?, ?)",
            mapping,
        )

        log(f"  Исходных названий: {len(mapping):,}")

        log("2/7. Подготовка категорий и числа товаров...")

        con.execute(f"""
            CREATE TEMP TABLE category_meta AS
            SELECT
                categoryId::BIGINT AS category_id,
                COUNT(DISTINCT sku_id) AS category_sku_count,
                first(category_name) AS category,
                first(category_path) AS category_path
            FROM {catalog_sql}
            WHERE sku_id IS NOT NULL
              AND categoryId IS NOT NULL
            GROUP BY categoryId
        """, [str(catalog_path)])

        # Один SKU может присутствовать в нескольких строках источника.
        # Для связи поисковых строк используем distinct пары SKU-категория.
        con.execute(f"""
            CREATE TEMP TABLE sku_category AS
            SELECT DISTINCT
                sku_id,
                categoryId::BIGINT AS category_id
            FROM {catalog_sql}
            WHERE sku_id IS NOT NULL
              AND categoryId IS NOT NULL
        """, [str(catalog_path)])

        log("3/7. Агрегация атрибутов в DuckDB...")

        # Не создаём списки всех значений в памяти.
        # Сначала получаем покрытие, количество разных значений и единиц.
        con.execute(f"""
            CREATE TEMP TABLE attr_stats AS
            SELECT
                a.categoryId::BIGINT AS category_id,
                m.canonical_attr AS name,
                COUNT(DISTINCT a.sku_id) FILTER (
                    WHERE a.value IS NOT NULL
                      AND trim(CAST(a.value AS VARCHAR)) <> ''
                ) AS sku_with_value,
                COUNT(DISTINCT trim(CAST(a.value AS VARCHAR))) FILTER (
                    WHERE a.value IS NOT NULL
                      AND trim(CAST(a.value AS VARCHAR)) <> ''
                ) AS distinct_values,
                COUNT(DISTINCT trim(CAST(a.unit AS VARCHAR))) FILTER (
                    WHERE a.unit IS NOT NULL
                      AND trim(CAST(a.unit AS VARCHAR)) <> ''
                ) AS distinct_units,
                COUNT(*) FILTER (
                    WHERE COALESCE(a.is_search, FALSE)
                ) AS search_rows,
                list(DISTINCT m.source_attr) AS source_fields
            FROM {attrs_sql} a
            JOIN attr_map m
              ON trim(CAST(a.attr AS VARCHAR)) = m.source_attr
            WHERE a.categoryId IS NOT NULL
            GROUP BY a.categoryId, m.canonical_attr
        """, [str(attrs_path)])

        log("4/7. Сбор примеров значений и единиц...")

        # Ограничиваем число образцов на исходное поле.
        # Это компактнее, чем list(value) по всему каталогу.
        samples = con.execute(f"""
            WITH value_counts AS (
                SELECT
                    a.categoryId::BIGINT AS category_id,
                    m.canonical_attr AS name,
                    m.source_attr,
                    trim(CAST(a.value AS VARCHAR)) AS value,
                    COUNT(*) AS n
                FROM {attrs_sql} a
                JOIN attr_map m
                  ON trim(CAST(a.attr AS VARCHAR)) = m.source_attr
                WHERE a.categoryId IS NOT NULL
                  AND a.value IS NOT NULL
                  AND trim(CAST(a.value AS VARCHAR)) <> ''
                GROUP BY 1, 2, 3, 4
            ),
            ranked AS (
                SELECT *,
                    row_number() OVER (
                        PARTITION BY category_id, name, source_attr
                        ORDER BY n DESC, value
                    ) AS rn
                FROM value_counts
            )
            SELECT category_id, name, source_attr, value
            FROM ranked
            WHERE rn <= 8
        """, [str(attrs_path)]).fetchall()

        unit_rows = con.execute(f"""
            SELECT DISTINCT
                a.categoryId::BIGINT AS category_id,
                m.canonical_attr AS name,
                m.source_attr,
                trim(CAST(a.unit AS VARCHAR)) AS unit
            FROM {attrs_sql} a
            JOIN attr_map m
              ON trim(CAST(a.attr AS VARCHAR)) = m.source_attr
            WHERE a.categoryId IS NOT NULL
              AND a.unit IS NOT NULL
              AND trim(CAST(a.unit AS VARCHAR)) <> ''
        """, [str(attrs_path)]).fetchall()

        samples_by_key = defaultdict(list)
        for cid, name, source, value in samples:
            key = (int(cid), clean(name))
            if clean(value):
                samples_by_key[key].append((clean(source), clean(value)))

        units_by_key = defaultdict(set)
        for cid, name, source, unit in unit_rows:
            if clean(unit):
                units_by_key[(int(cid), clean(name))].add(clean(unit))

        log("5/7. Оценка поискового спроса по текстам запросов...")

        # query_text в этом Parquet отображается DuckDB как
        # "toValidUTF8(query_text)" — используем фактическое имя поля.
        # Намерение определяется регулярными выражениями; это эвристика.
        intent_cases = "\n".join(
            f"WHEN regexp_matches(lower(coalesce(query_text, '')), '{pattern}') "
            f"THEN '{intent}'"
            for intent, pattern in INTENTS.items()
        )

        search_rows = con.execute(f"""
            WITH q AS (
                SELECT
                    sku_id,
                    "toValidUTF8(query_text)" AS query_text
                FROM read_parquet(?)
                WHERE sku_id IS NOT NULL
            ),
            classified AS (
                SELECT
                    sc.category_id,
                    CASE
                        {intent_cases}
                        ELSE NULL
                    END AS intent,
                    lower(trim(q.query_text)) AS query_text
                FROM q
                JOIN sku_category sc USING (sku_id)
                WHERE q.query_text IS NOT NULL
                  AND trim(q.query_text) <> ''
            )
            SELECT
                category_id,
                intent,
                COUNT(DISTINCT query_text) AS distinct_queries,
                COUNT(*) AS query_rows
            FROM classified
            WHERE intent IS NOT NULL
            GROUP BY category_id, intent
        """, [str(query_path)]).fetchall()

        demand = defaultdict(dict)
        for cid, intent, distinct_queries, query_rows in search_rows:
            demand[int(cid)][clean(intent)] = {
                "distinct_queries": int(distinct_queries or 0),
                "query_rows": int(query_rows or 0),
            }

        log("6/7. Расчёт оценок и приоритетов...")

        stat_rows = con.execute("""
            SELECT
                s.category_id,
                s.name,
                s.sku_with_value,
                s.distinct_values,
                s.distinct_units,
                s.search_rows,
                s.source_fields,
                c.category_sku_count,
                c.category,
                c.category_path
            FROM attr_stats s
            JOIN category_meta c USING (category_id)
        """).fetchall()

        # Сначала находим максимальный спрос по каждой категории,
        # чтобы сравнивать намерения в пределах одной категории.
        max_demand = {}
        for cid, intents in demand.items():
            max_demand[cid] = max(
                [v["query_rows"] for v in intents.values()] or [0]
            )

        records = []
        for row in stat_rows:
            (
                cid, name, sku_with_value, distinct_values, distinct_units,
                search_count, source_fields, sku_count, category, category_path
            ) = row

            cid = int(cid)
            sku_count = int(sku_count or 0)
            sku_with_value = int(sku_with_value or 0)
            distinct_values = int(distinct_values or 0)
            distinct_units = int(distinct_units or 0)

            if sku_count <= 0:
                continue

            coverage = min(1.0, sku_with_value / sku_count)
            if coverage < MIN_COVERAGE and not intent_for_attribute(name):
                continue

            intents = intent_for_attribute(name)
            intent_stats = demand.get(cid, {})

            # Search score основан на частоте запросов, соответствующих
            # семантической группе атрибута. Нормализуется в категории.
            search_raw = max(
                [intent_stats.get(i, {}).get("query_rows", 0) for i in intents]
                or [0]
            )
            denom = max_demand.get(cid, 0)
            search_score = (
                math.log1p(search_raw) / math.log1p(denom)
                if search_raw > 0 and denom > 0
                else 0.0
            )
            search_score = min(1.0, max(0.0, search_score))

            # Comparison proxy: покрытие плюс логарифмическое разнообразие.
            # Это не энтропия и не экспертная оценка полезности.
            diversity = (
                min(1.0, math.log1p(distinct_values) / math.log1p(100))
                if distinct_values > 0 else 0.0
            )
            comparison_score = 0.60 * coverage + 0.40 * diversity

            # Надёжность извлечения пока не измерена на размеченной выборке.
            extraction_score = None
            measured_extraction = False

            importance_score = (
                100 * (
                    W_SEARCH * search_score
                    + W_COMPARISON * comparison_score
                    # Не перераспределяем вес extraction на другие компоненты.
                    # Итоговый балл отражает неполноту оценки.
                )
            )

            excluded = is_excluded(name)
            if excluded:
                priority = "exclude"
                reason = "service_or_commercial_attribute"
            elif coverage >= 0.50 and (
                search_score >= 0.20 or comparison_score >= 0.55
            ):
                priority = "core"
                reason = "high_coverage_and_search_or_comparison_value"
            elif coverage >= 0.10 or search_score >= 0.20:
                priority = "optional"
                reason = "partial_coverage_or_search_signal"
            else:
                priority = "optional"
                reason = "low_evidence_keep_for_review"

            source_fields = sorted({
                clean(x) for x in (source_fields or []) if clean(x)
            })
            sample_pairs = samples_by_key.get((cid, clean(name)), [])
            sample_values = []
            for _, value in sample_pairs:
                if value not in sample_values:
                    sample_values.append(value)
                if len(sample_values) >= 8:
                    break

            source_roles = []
            for source in source_fields:
                role = (
                    "search_or_exact"
                    if source.casefold().endswith("_search")
                    else "base_or_categorical"
                )
                source_roles.append({"name": source, "role": role})

            all_units = sorted(units_by_key.get((cid, clean(name)), set()))
            values_for_type = sample_values[:]
            attr_type = infer_type(values_for_type, all_units)

            records.append({
                "category_id": cid,
                "category": clean(category) or None,
                "category_path": clean(category_path) or None,
                "category_sku_count": sku_count,
                "name": clean(name),
                "type": attr_type,
                "units": all_units[:30],
                "source_fields": source_roles,
                "aliases": [
                    x for x in source_fields
                    if x.casefold() != clean(name).casefold()
                ],
                "coverage": round(coverage, 6),
                "sku_with_value": sku_with_value,
                "distinct_values": distinct_values,
                "distinct_units": distinct_units,
                "sample_values": sample_values,
                "scores": {
                    "search_score": round(search_score, 4),
                    "comparison_score": round(comparison_score, 4),
                    "extraction_score": extraction_score,
                    "importance_score": round(importance_score, 2),
                    "score_status": "partial_extraction_unmeasured",
                    "weights": {
                        "search": W_SEARCH,
                        "comparison": W_COMPARISON,
                        "extraction": W_EXTRACTION,
                    },
                },
                "priority": priority,
                "priority_reason": reason,
                "search_intents": intents,
                "search_evidence": {
                    intent: intent_stats[intent]
                    for intent in intents if intent in intent_stats
                },
                "applicability": "unknown",
                "extract_from": ["title", "description"],
                "requires_validation": True,
            })

        log("7/7. Формирование JSON...")

        by_category = {}
        for item in records:
            cid = item["category_id"]
            if cid not in by_category:
                by_category[cid] = {
                    "category_id": cid,
                    "category": item["category"],
                    "category_path": item["category_path"],
                    "category_sku_count": item["category_sku_count"],
                    "attributes": [],
                }
            by_category[cid]["attributes"].append(item)

        for schema in by_category.values():
            # Core сначала, затем score. Служебные атрибуты остаются
            # в результате с priority=exclude для аудита.
            schema["attributes"].sort(
                key=lambda a: (
                    {"core": 0, "optional": 1, "exclude": 2}[a["priority"]],
                    -a["scores"]["importance_score"],
                    -a["coverage"],
                    a["name"].casefold(),
                )
            )
            schema["attributes"] = schema["attributes"][
                :MAX_ATTRIBUTES_PER_CATEGORY
            ]

        result = {
            "schema_version": "1.0-candidate",
            "generated_by": "notebooks/eda/build_ideal_schema.py",
            "description": (
                "Candidate category-specific schema for search, comparison "
                "and extraction. Search score is a query-text proxy; "
                "extraction score is intentionally null until evaluated "
                "against labeled product text. Validate before production."
            ),
            "scoring": {
                "weights": {
                    "search": W_SEARCH,
                    "comparison": W_COMPARISON,
                    "extraction": W_EXTRACTION,
                },
                "importance_score_note": (
                    "Partial score: weighted search and comparison components "
                    "only; extraction is unmeasured and its weight is not "
                    "redistributed. Do not interpret as calibrated probability."
                ),
                "comparison_score_note": (
                    "0.60 * SKU coverage + 0.40 * log-scaled value diversity; "
                    "a heuristic proxy, not an empirical buyer-preference model."
                ),
                "search_score_note": (
                    "Uses query-text intent patterns joined to catalog "
                    "categories by sku_id. Rows are not confirmed clicks "
                    "or conversions."
                ),
            },
            "validation_status": {
                "extraction_score": "not_measured",
                "applicability": "not_measured",
                "alias_equivalence": "requires_review",
            },
            "categories": sorted(
                by_category.values(),
                key=lambda c: c["category_sku_count"],
                reverse=True,
            ),
        }

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        temp_output = OUTPUT.with_suffix(".json.tmp")
        temp_output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_output.replace(OUTPUT)

        log(f"Сохранено: {OUTPUT}")
        log(f"Категорий: {len(result['categories']):,}")
        log(f"Атрибутов до лимита на категорию: {len(records):,}")
        log(
            "Extraction score: не измерен; в JSON сохранён null. "
            "Это намеренно."
        )

    finally:
        con.close()


if __name__ == "__main__":
    main()
