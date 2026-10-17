import html
import json
import re

import pandas as pd

from color_dictionary import COLOR_DICTIONARY, COLOR_NORMALIZATION
from country_dictionary import COUNTRY_DICTIONARY
from material_dictionary import MATERIAL_DICTIONARY

INPUT_FILE = "sku_desc.parquet"
OUTPUT_FILE = "normalized.json"

ATTRIBUTE_ALIASES = {
    "гарантия": "Гарантийный срок",
    "срок гарантии": "Гарантийный срок",
    "гарантийный период": "Гарантийный срок",
    "гарантия производителя": "Гарантийный срок",
    "гарантийный срок производителя": "Гарантийный срок",
    "страна происхождения": "Страна производства",
    "страна изготовления": "Страна производства",
    "происхождение": "Страна производства",
    "origin country": "Страна производства",
    "country of origin": "Страна производства",
    "material": "Материал",
    "item material": "Материал",
    "product material": "Материал",
    "age restrictions": "Возрастные ограничения",
    "возрастные ограничения по использованию": "Возрастные ограничения",
    "рекомендуемый возраст": "Возрастные ограничения",
    "возрастные рекомендации": "Возрастные ограничения",
    "возрастная категория": "Возрастные ограничения",
    "colour": "Цвет",
    "color": "Цвет",
    "цвет изделия": "Цвет",
    "объём": "Объем",
    "вид застежки": "Тип застежки",
    "kind of fastener": "Тип застежки",
    "подарочное назначение": "Назначение подарка",
    "gift purpose": "Назначение подарка",
    "повод для подарка": "Повод",
    "halter": "Повод",
    "movement mechanism": "Механизм движения",
    "toy material": "Материал игрушки",
    "toy features": "Особенности игрушки",
    "kind of toy car": "Тип игрушечного автомобиля",
    "car type": "Тип автомобиля",
    "effects": "Эффекты",
    "pattern": "Рисунок",
    "plot": "Сюжет",
    "gift type": "Тип подарка",
    "package": "Комплектация",
    "power cell type": "Тип элемента питания",
    "cable cross-section": "Форма сечения",
}

COUNTRY_EXTRA_ALIASES = {
    alias.casefold().replace("ё", "е"): country
    for country, aliases in COUNTRY_DICTIONARY.items()
    for alias in [country, *aliases]
}
MATERIAL_ALIASES = {
    alias.casefold().replace("ё", "е"): material
    for material, aliases in MATERIAL_DICTIONARY.items()
    for alias in [material, *aliases]
}
# Для каждого варианта сохраняем и конкретный цвет, и группу цвета.
# Например: «серебристый» -> («серебристый», «серый»).
COLOR_ALIASES = {}
for color_group, aliases in COLOR_DICTIONARY.items():
    normalized_group = color_group.casefold().replace("ё", "е")
    for alias in aliases:
        key = alias.casefold().replace("ё", "е")
        specific_color = COLOR_NORMALIZATION.get(key, key)
        COLOR_ALIASES[key] = (specific_color, normalized_group)


def clean_text(value):
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""

    text = html.unescape(str(value))
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = re.sub(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF\uFE0F\u200D]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s*[-–—]+\s*$", "", text).strip()


def normalize_key(value):
    value = clean_text(value).casefold().replace("ё", "е")
    return re.sub(r"\s+", " ", value).strip(" :.-")


def compile_alias_pattern(aliases):
    ordered_aliases = sorted(aliases, key=len, reverse=True)
    if not ordered_aliases:
        return None
    alternatives = "|".join(re.escape(alias) for alias in ordered_aliases)
    return re.compile(r"(?<!\w)(?:" + alternatives + r")(?!\w)")


COUNTRY_PATTERN = compile_alias_pattern(COUNTRY_EXTRA_ALIASES)
MATERIAL_PATTERN = compile_alias_pattern(MATERIAL_ALIASES)
COLOR_PATTERN = compile_alias_pattern(COLOR_ALIASES)


def normalize_attribute_name(name):
    cleaned = clean_text(name).strip(" :.-")
    return ATTRIBUTE_ALIASES.get(normalize_key(cleaned), cleaned)


def normalize_country(value):
    key = normalize_key(value).strip(' .,:;"\'')
    return COUNTRY_EXTRA_ALIASES.get(key, clean_text(value))


def find_country_in_text(text):
    normalized = normalize_key(text)
    if not normalized or COUNTRY_PATTERN is None:
        return None
    match = COUNTRY_PATTERN.search(normalized)
    return COUNTRY_EXTRA_ALIASES.get(match.group(0)) if match else None


def find_materials_in_text(text):
    normalized = normalize_key(text)
    if not normalized or MATERIAL_PATTERN is None:
        return []

    found = []
    occupied_spans = []
    for match in MATERIAL_PATTERN.finditer(normalized):
        if any(match.start() < end and match.end() > start for start, end in occupied_spans):
            continue
        occupied_spans.append(match.span())
        material = MATERIAL_ALIASES[match.group(0)]
        if material not in found:
            found.append(material)
    return found


def extract_color(text):
    """Находит все цвета в тексте и возвращает оттенок вместе с его группой."""
    normalized = normalize_key(text)
    if not normalized or COLOR_PATTERN is None:
        return []

    found = []
    for match in COLOR_PATTERN.finditer(normalized):
        alias = match.group(0).casefold().replace("ё", "е")
        colors = COLOR_ALIASES.get(alias)
        if colors is None:
            continue

        # Сначала конкретный оттенок, затем общая группа.
        # dict.fromkeys убирает повторы, например «серый, серый».
        for color in colors:
            if color and color not in found:
                found.append(color)
    return found


def add_text_attribute(features, name, values, separator="; "):
    if not values:
        return features

    target = next((item for item in features if normalize_key(item.get("@name", "")) == normalize_key(name)), None)
    if target is None:
        features.append({"id": len(features) + 1, "@name": name, "#text": separator.join(values)})
        return features

    existing = [part.strip() for part in re.split(r"\s*[;,]\s*", target.get("#text", "")) if part.strip()]
    target["#text"] = separator.join(dict.fromkeys(existing + values))
    return features


def normalize_and_merge_attributes(features):
    merged = []
    lookup = {}

    for feature in features:
        name = normalize_attribute_name(feature.get("@name", ""))
        value = clean_text(feature.get("#text", ""))
        unit = clean_text(feature.get("@unit", ""))

        if normalize_key(name) == "страна производства":
            value = normalize_country(value)
        if not name or not value:
            continue

        key = (normalize_key(name), normalize_key(unit))
        if key not in lookup:
            item = {"id": len(merged) + 1, "@name": name, "#text": value}
            if unit:
                item["@unit"] = unit
            lookup[key] = item
            merged.append(item)
            continue

        item = lookup[key]
        existing = [part.strip() for part in item["#text"].split("; ") if part.strip()]
        if value not in existing:
            item["#text"] = "; ".join(existing + [value])

    for index, feature in enumerate(merged, start=1):
        feature["id"] = index
    return merged


def html_to_lines(value):
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return []

    text = html.unescape(str(value))
    text = re.sub(r"<br\s*/?>|</p\s*>|</div\s*>|</li\s*>|<li\b[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
    return [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]


UNITS = sorted({
    "кВт·ч", "кВт/ч", "кВт", "МВт", "мА·ч", "А·ч", "мА", "А", "мкм", "мм", "см", "км", "м",
    "мг", "кг", "г", "мл", "л", "Вт", "мВ", "кВ", "В", "°C", "°F", "℃", "мм²", "см²", "м²",
    "км²", "мм³", "см³", "м³", "Гц", "кГц", "МГц", "ГГц", "дБ", "дюйм", "дюйма", "дюймов",
    "секунд", "секунда", "секунды", "с", "минут", "минута", "минуты", "мин", "часов", "часа", "час", "ч",
    "суток", "сутки", "дней", "день", "м/с", "км/ч", "об/мин", "шт.", "шт", "%",
}, key=len, reverse=True)
UNIT_PATTERNS = [(unit, re.compile(r"^(.*?)\s*" + re.escape(unit) + r"\s*$", re.IGNORECASE)) for unit in UNITS]
NUMBER_PATTERN = re.compile(r"[\d\s.,+\-–—/×xX]+")
HEADER_PATTERN = re.compile(r"характеристик\w*(?:\s+[а-яёa-z0-9№-]+){0,5}\s*:", re.IGNORECASE)
SECTION_PATTERN = re.compile(r"^\s*(Комплектация|Упаковка|Инструкция|Гарантия|Описание|Примечание|Подключение)(?:\s+[^:.;!?]*)?\s*:", re.IGNORECASE)
FEATURE_PATTERN = re.compile(r"(?:^|[.;]\s*|•\s*)([^:;•\n]+?)\s*:\s*")


def extract_unit(value):
    value = clean_text(value).strip()
    for unit, pattern in UNIT_PATTERNS:
        match = pattern.match(value)
        if not match or not re.search(r"\d", match.group(1)):
            continue
        number = match.group(1).strip()
        if NUMBER_PATTERN.fullmatch(number):
            return number, unit
    return value, None


def parse_product_description(raw_description):
    lines = html_to_lines(raw_description)
    features = []
    in_features = False

    for line in lines:
        header = HEADER_PATTERN.search(line)
        if header and not in_features:
            in_features = True
            line = line[header.end():].strip()
            if not line:
                continue
        if not in_features:
            continue

        next_section = SECTION_PATTERN.search(line)
        if next_section:
            line = line[:next_section.start()].strip()
            in_features = False
        if not line:
            continue

        matches = list(FEATURE_PATTERN.finditer(line))
        for index, match in enumerate(matches):
            name = normalize_attribute_name(clean_text(match.group(1)).strip(" .;:-"))
            end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
            value = clean_text(line[match.end():end]).strip(" .;:-")
            value, unit = extract_unit(value)
            if not name or not value:
                continue

            feature = {"id": len(features) + 1, "@name": name, "#text": value}
            if unit:
                feature["@unit"] = unit
            features.append(feature)

    return clean_text(raw_description), features


def normalize_id(value):
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    return int(value) if re.fullmatch(r"\d+", value) else value


def process_product(sku_id, title, raw_description):
    title = clean_text(title)
    description, features = parse_product_description(raw_description)
    features = normalize_and_merge_attributes(features)
    full_text = f"{title} {description}"

    features = add_text_attribute(features, "Материал", find_materials_in_text(full_text))

    if not any(normalize_key(item.get("@name", "")) == "страна производства" for item in features):
        country = find_country_in_text(full_text)
        if country:
            features.append({"id": len(features) + 1, "@name": "Страна производства", "#text": country})

    colors = extract_color(full_text)
    if colors:
        features = add_text_attribute(features, "Цвет", colors, separator=", ")

    return {
        "@id": normalize_id(sku_id),
        "name": title,
        "description": description,
        "param": normalize_and_merge_attributes(features),
    }


def main():
    dataframe = pd.read_parquet(INPUT_FILE).head(500)
    required_columns = {"sku_id", "title", "description"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Отсутствуют столбцы: {sorted(missing_columns)}. Найдены: {dataframe.columns.tolist()}")

    products = [
        process_product(*row)
        for row in dataframe[["sku_id", "title", "description"]].itertuples(index=False, name=None)
    ]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
        json.dump(products, output_file, ensure_ascii=False, indent=2)

    print(f"Обработано товаров: {len(products)}")
    print(f"JSON сохранён в: {OUTPUT_FILE}")
    if products:
        print("\nПервый товар:")
        print(json.dumps(products[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
