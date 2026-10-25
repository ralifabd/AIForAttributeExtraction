"""Тесты проверки извлечённых атрибутов (без базы и без LLM).
Запуск из папки notebooks:  py -3.12 -m pytest tests/test_attr_check.py -q
"""
from quality.attr_check import AttributeChecker, level, summarize

TITLE = "Электрочайник Delta Lux DL-1285"
DESC = (
    "Резервуар выполнен из нержавеющей стали, ручка из черного ABS пластика. "
    "Потребляет 2200 Вт. Рассчитан на 1700 мл воды. Есть опция автоматического "
    "отключения при закипании. Управление механическое. Высота — 23 см."
)
checker = AttributeChecker(None)   # без словаря значений: только проверка по тексту


def conf(name, value, unit=""):
    p = {"@name": name, "#text": value}
    if unit:
        p["@unit"] = unit
    return checker.check(p, TITLE, DESC)["confidence"]


def test_numbers_found_with_unit_conversion():
    assert level(conf("Мощность", "2200", "Вт")) == "надёжно"
    assert level(conf("Объем", "1.7", "л")) == "надёжно"        # в тексте 1700 мл
    assert level(conf("Мощность", "2,2", "кВт")) == "надёжно"


def test_text_values_in_other_word_form():
    assert level(conf("Цвет", "черный")) == "надёжно"            # в тексте "черного"
    assert level(conf("Материал", "нержавеющая сталь")) == "надёжно"
    assert level(conf("Тип управления", "механическое")) == "надёжно"


def test_yes_no_properties():
    assert level(conf("Автоотключение", "да")) == "надёжно"     # "автоматического отключения"
    assert level(conf("Подсветка", "да")) == "сомнительно"


def test_hallucinations_are_doubtful():
    assert level(conf("Мощность", "22000", "Вт")) == "сомнительно"
    assert level(conf("Объем", "2.5", "л")) == "сомнительно"
    assert level(conf("Цвет", "белый")) == "сомнительно"


def test_summary_counts_and_core_coverage():
    params = [
        {"@name": "Мощность", "#text": "2200", "confidence": 0.95},
        {"@name": "Цвет", "#text": "белый", "confidence": 0.3},
    ]
    q = summarize(params, ["Мощность", "Цвет", "Объем"])
    assert q["reliable"] == 1 and q["doubtful"] == 1
    assert q["core_filled"] == 2 and q["core_missing"] == ["Объем"]
