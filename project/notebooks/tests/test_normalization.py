import pytest

from normalization.main import (
    clean_text,
    normalize_id,
    normalize_attribute_name,
    parse_seller_params,
    normalize_and_merge_attributes,
    extract_warranty,
)


from normalization.morphology import normalize_phrase
from normalization.dictionary_normalization.color_normalization import (
    extract_colors_from_words,
)
from normalization.dictionary_normalization.material_normalization import (
    find_materials_in_words,
)
from normalization.dictionary_normalization.country_normalization import (
    find_countries_in_words,
)
from normalization.dictionary_normalization.measurement_normalization import (
    extract_measurements,
    combine_dimensions,
)



# ---------- Очистка текста ----------

@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("  Зеркало   с подсветкой  ", "Зеркало с подсветкой"),
        ("Сталь &amp; стекло", "Сталь & стекло"),
        ("", ""),
        (None, ""),
    ],
)
def test_clean_text(source, expected):
    assert clean_text(source) == expected


# ---------- ID товара ----------

@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (123, 123),
        ("401054646", 401054646),
        (" 401054646 ", 401054646),
        ("sku-123", "sku-123"),
        (None, None),
    ],
)
def test_normalize_id(source, expected):
    assert normalize_id(source) == expected


# ---------- Названия характеристик ----------

@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("глубина", "Длина"),
        ("глубина изделия", "Длина"),
        ("width", "Ширина"),
        ("height", "Высота"),
        ("length", "Длина"),
        ("габариты", "Габаритные размеры (В*Ш*Д)"),
    ],
)
def test_normalize_attribute_name(source, expected):
    assert normalize_attribute_name(source) == expected


# ---------- Характеристики продавца ----------

def test_parse_seller_params_normalizes_names():
    params = [
        {"@name": "Глубина", "#text": "50", "@unit": "см"},
        {"@name": "Width", "#text": "40", "@unit": "см"},
    ]

    result = parse_seller_params(params)

    assert result[0]["@name"] == "Длина"
    assert result[0]["#text"] == "50"
    assert result[0]["@unit"] == "см"

    assert result[1]["@name"] == "Ширина"
    assert result[1]["#text"] == "40"


def test_parse_seller_params_skips_empty_values():
    params = [
        {"@name": "Цвет", "#text": ""},
        {"@name": "", "#text": "Серый"},
        {"@name": "Материал", "#text": "Сталь"},
    ]

    result = parse_seller_params(params)

    assert len(result) == 1
    assert result[0]["@name"] == "Материал"


def test_parse_seller_params_accepts_json_string():
    params = (
        '[{"@name": "Цвет", "#text": "Серый"}]'
    )

    result = parse_seller_params(params)

    assert len(result) == 1
    assert result[0]["@name"] == "Цвет"
    assert result[0]["#text"] == "Серый"


def test_parse_seller_params_handles_invalid_json():
    assert parse_seller_params("{invalid json") == []


# ---------- Объединение повторяющихся характеристик ----------

def test_normalize_and_merge_attributes_merges_duplicate_names():
    features = [
        {"@name": "Цвет", "#text": "Серый"},
        {"@name": "Цвет", "#text": "Серебристый"},
    ]

    result = normalize_and_merge_attributes(features)

    colors = [
        item for item in result
        if item["@name"] == "Цвет"
    ]

    assert len(colors) == 1
    assert "Серый" in colors[0]["#text"]
    assert "Серебристый" in colors[0]["#text"]


def test_normalize_and_merge_attributes_skips_empty_features():
    features = [
        {"@name": "Цвет", "#text": ""},
        {"@name": "", "#text": "Сталь"},
        {"@name": "Материал", "#text": "Нержавеющая сталь"},
    ]

    result = normalize_and_merge_attributes(features)

    assert len(result) == 1
    assert result[0]["@name"] == "Материал"


# ---------- Гарантия ----------

@pytest.mark.parametrize(
    ("source", "expected_value", "expected_unit"),
    [
        ("Гарантия: 2 года", "2", "года"),
        ("Гарантия 12 месяцев", "12", "месяцев"),
        ("Срок гарантии: 1 год", "1", "год"),
    ],
)
def test_extract_warranty(source, expected_value, expected_unit):
    result = extract_warranty(source)

    assert result is not None
    assert result["@name"] == "Гарантия"
    assert result["#text"] == expected_value
    assert result["@unit"] == expected_unit


def test_extract_warranty_returns_none_without_warranty():
    assert extract_warranty("Зеркало с подсветкой") is None





# ============================================================
# ЦВЕТА
# ============================================================

def test_extract_gray_color():
    words = normalize_phrase("Зеркало в серебристо-серой раме")
    colors = extract_colors_from_words(words)

    assert colors
    assert any("сер" in color.lower() for color in colors)


def test_extract_multiple_colors():
    words = normalize_phrase("Черная сумка с белыми вставками")
    colors = extract_colors_from_words(words)

    assert len(colors) >= 2


def test_extract_blue_color():
    words = normalize_phrase("Темно-синий чехол")
    colors = extract_colors_from_words(words)

    assert colors
    assert any("син" in color.lower() for color in colors)


def test_no_colors_in_text():
    words = normalize_phrase("Настольная лампа с сенсорным управлением")
    colors = extract_colors_from_words(words)

    assert colors == []


# ============================================================
# МАТЕРИАЛЫ
# ============================================================

def test_extract_stainless_steel():
    words = normalize_phrase("Полка изготовлена из нержавеющей стали")
    materials = find_materials_in_words(words)

    assert materials
    assert any(
        "нержав" in material.lower()
        for material in materials
    )


def test_extract_multiple_materials():
    words = normalize_phrase("Корпус выполнен из металла и пластика")
    materials = find_materials_in_words(words)

    assert len(materials) >= 2


def test_no_materials_in_text():
    words = normalize_phrase("Настольная лампа с регулировкой яркости")
    materials = find_materials_in_words(words)

    assert materials == []


# ============================================================
# СТРАНЫ
# ============================================================

def test_extract_china():
    words = normalize_phrase("Страна производства Китай")
    countries = find_countries_in_words(words)

    assert countries
    assert any("китай" in country.lower() for country in countries)


def test_extract_russia():
    words = normalize_phrase("Изготовлено в России")
    countries = find_countries_in_words(words)

    assert countries
    assert any("росси" in country.lower() for country in countries)


def test_no_countries_in_text():
    words = normalize_phrase("Складной стул с мягким сиденьем")
    countries = find_countries_in_words(words)

    assert countries == []


# ============================================================
# ИЗВЛЕЧЕНИЕ ИЗМЕРЕНИЙ
# ============================================================

def test_extract_width():
    measurements = extract_measurements("Ширина изделия 50 см")

    assert any(
        name == "Ширина" and value == "50" and unit == "см"
        for name, value, unit in measurements
    )


def test_extract_height():
    measurements = extract_measurements("Высота изделия 120 см")

    assert any(
        name == "Высота" and value == "120" and unit == "см"
        for name, value, unit in measurements
    )


def test_extract_weight():
    measurements = extract_measurements("Вес изделия 2 кг")

    assert any(
        name == "Вес" and value == "2" and unit == "кг"
        for name, value, unit in measurements
    )


def test_extract_decimal_measurement():
    measurements = extract_measurements("Длина изделия 12.5 см")

    assert any(
        name == "Длина" and value == "12,5" and unit == "см"
        for name, value, unit in measurements
    )


def test_empty_measurement_text():
    assert extract_measurements("") == []


# ============================================================
# ОБЪЕДИНЕНИЕ ГАБАРИТОВ
# ============================================================

def test_combine_three_dimensions():
    dimensions = [
        ("Длина", "100", "см"),
        ("Ширина", "50", "см"),
        ("Высота", "80", "см"),
    ]

    result = combine_dimensions(dimensions)

    assert ("Габаритные размеры (В*Ш*Д)", "80x50x100", "см") in result


def test_no_gabarites_if_height_missing():
    dimensions = [
        ("Длина", "100", "см"),
        ("Ширина", "50", "см"),
    ]

    result = combine_dimensions(dimensions)

    assert not any(
        name == "Габаритные размеры (В*Ш*Д)"
        for name, value, unit in result
    )


def test_dimensions_with_different_units_not_combined():
    dimensions = [
        ("Длина", "100", "см"),
        ("Ширина", "50", "см"),
        ("Высота", "80", "мм"),
    ]

    result = combine_dimensions(dimensions)

    assert not any(
        name == "Габаритные размеры (В*Ш*Д)"
        for name, value, unit in result
    )


def test_gabarites_not_duplicated():
    dimensions = [
        ("Длина", "100", "см"),
        ("Ширина", "50", "см"),
        ("Высота", "80", "см"),
    ]

    first_result = combine_dimensions(dimensions)
    second_result = combine_dimensions(first_result)

    assert second_result.count(
        ("Габаритные размеры (В*Ш*Д)", "80x50x100", "см")
    ) == 1


