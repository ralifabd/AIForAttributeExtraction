import json

from config import DATA_DIR
from normalization.main import process_product
from quality.attr_check import AttributeChecker, level, summarize

from extract_ideal_attributes import (
    find_category_candidates,
    classify_category,
    extract_attributes,
    get_attribute_name,
    get_category_id,
    CATEGORIES,
    brand_model_extractor,
    clean_text,
)

attribute_checker = AttributeChecker(DATA_DIR / "rag.sqlite")


def finalize_product(product, params, title, description,
                     category_id=None, category=None):
    """Ставит уверенность атрибутам без неё (извлечённым LLM) и добавляет
    внизу результата сводку extraction_quality."""
    for p in params:
        if "confidence" not in p:          # у атрибутов нормализации она уже есть
            p.update(attribute_checker.check(p, title, description, category_id))

    core_names = [
        get_attribute_name(a)
        for a in (category or {}).get("attributes", [])
        if a.get("priority") == "core" and get_attribute_name(a)
    ]

    product["param"] = params
    product["extraction_quality"] = summarize(params, core_names or None)

    q = product["extraction_quality"]
    print(f"Качество: атрибутов {q['attributes_total']}, "
          f"средняя уверенность {q['avg_confidence']}, "
          f"надёжно {q['reliable']}, проверить {q['to_check']}, "
          f"сомнительно {q['doubtful']}", flush=True)
    for p in params:
        if "check" in p:
            print(f"  {p['@name']}: {p['#text']} {p.get('@unit', '')} -> "
                  f"{p['confidence']} ({level(p['confidence'])}): {p['check']}",
                  flush=True)
    return product


def enrich_product(sku_id, title, description):
    # 1. Сначала выполняем существующую нормализацию.
    product = process_product(
        sku_id,
        title,
        description,
    )

    # Не изменяем исходные параметры товара.
    existing_params = product.get("param", [])
    existing_params = [
        p for p in existing_params
        if isinstance(p, dict)
    ]

    # 2. Ищем категорию и её ideal_schema.
    source = {
        "sku_id": str(sku_id),
        "title": title,
        "description": description,
    }

    candidates = find_category_candidates(source)
    category = classify_category(source, candidates)

    print("Кандидаты категорий:", candidates[:5])
    print("Выбранная категория:", category.get("category") if category else None)
    print("ID категории:", get_category_id(category) if category else None)

     # Бренд и модель по словарям
    category_id = get_category_id(category) if category else None
    category_id_str = str(category_id) if category_id is not None else None

    brand_model = brand_model_extractor.extract(
        name=title,
        description=clean_text(description),
        category_id=category_id_str,
    )
    if brand_model["brand"]:
        product["vendor"] = brand_model["brand"]
    if brand_model["model"]:
        product["model"] = brand_model["model"]

    product["brand_model_check"] = {
        "brand_confidence": brand_model["brand_confidence"],
        "model_confidence": brand_model["model_confidence"],
        "model_status": brand_model["model_status"],
        "needs_llm": brand_model["brand_model_needs_llm"] == "да",
    }

    if category is None:
        return finalize_product(product, existing_params, title, description)
  
    if category_id is not None:
        product["categories"] = {
            "categoryId": str(category_id)
    }
    
    # Дальше идёт существующий код:
    # 3. Составляем список уже имеющихся атрибутов.
    existing_names = {
        str(p.get("@name", "")).strip().casefold()
        for p in existing_params
        if p.get("@name")
    }

    # 4. Оставляем только отсутствующие целевые атрибуты.
    missing_attributes = [
        a
        for a in category.get("attributes", [])
        if get_attribute_name(a)
        and a.get("priority") == "core"
        and get_attribute_name(a).strip().casefold()
            not in existing_names
    ]
    
    core_total = sum(
        1 for a in category.get("attributes", [])
        if a.get("priority") == "core"
    )
    
    core_filled = sum(
        1 for a in category.get("attributes", [])
        if a.get("priority") == "core"
        and get_attribute_name(a)
        and get_attribute_name(a).strip().casefold() in existing_names
    )
    
    print(f"Основных атрибутов в схеме: {core_total}")
    print(f"Основных атрибутов уже заполнено: {core_filled}")
    print(f"Основных атрибутов передаём LLM: {len(missing_attributes)}")

    if not missing_attributes:
        return finalize_product(product, existing_params, title, description,
                                category_id_str, category)

    # 5. Просим LLM извлечь только недостающие атрибуты.
    # extract_attributes использует атрибуты из category,
    # поэтому создаём копию категории с нужным подмножеством.
    category_for_extraction = dict(category)
    category_for_extraction["attributes"] = missing_attributes

    new_params = extract_attributes(
        source,
        category_for_extraction,
    )

    # 6. Дополнительная защита от повторов.
    known_names = set(existing_names)

    for param in new_params:
        name = str(param.get("@name", "")).strip()
        key = name.casefold()

        if not key or key in known_names:
            continue

        existing_params.append(param)
        known_names.add(key)

    # 7. Сохраняем старые параметры и добавляем новые + проверка качества.
    return finalize_product(product, existing_params, title, description,
                            category_id_str, category)
