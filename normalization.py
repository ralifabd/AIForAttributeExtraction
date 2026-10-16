
import json
import re
import html
import pandas as pd


INPUT_FILE = "sku_desc.parquet"
OUTPUT_FILE = "normalized.json"


def html_to_lines(value):
    if pd.isna(value):
        return []

    text = html.unescape(str(value))
    text = re.sub(
        r"<br\s*/?>|</p\s*>|</div\s*>|</li\s*>",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<li\b[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]*>", " ", text)

    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)

    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)

    return lines


def clean_text(value):
    if pd.isna(value):
        return ""

    text = html.unescape(str(value))
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)

    # Убираем эмодзи
    text = re.sub(
        r"[\U0001F000-\U0001FAFF"
        r"\U00002600-\U000027BF"
        r"\uFE0F\u200D]+",
        "",
        text,
    )

    text = re.sub(r"\s+", " ", text).strip()

    text = re.sub(r"\s*[-–—]+\s*$", "", text)

    return text



def parse_product_description(raw_description):
    lines = html_to_lines(raw_description)
    features = []
    in_features = False

    header_re = re.compile(
        r"характеристик\w*(?:\s+[а-яёa-z0-9№-]+){0,5}\s*:",
        re.IGNORECASE,
    )

    section_re = re.compile(
        r"^\s*(Комплектация|Упаковка|Инструкция|Гарантия|"
        r"Описание|Примечание|Подключение)"
        r"(?:\s+[^:.;!?]*)?\s*:",
        re.IGNORECASE,
    )

    for line in lines:
        header = header_re.search(line)

        if header and not in_features:
            in_features = True
            line = line[header.end():].strip()

            if not line:
                continue

        if not in_features:
            continue

        next_section = section_re.search(line)
        if next_section:
            line = line[:next_section.start()].strip()
            in_features = False

        if not line:
            continue

        matches = list(re.finditer(
            r"(?:^|[.;]\s*|•\s*)"
            r"([^:;•\n]+?)\s*:\s*",
            line,
        ))

        for i, match in enumerate(matches):
            name = clean_text(match.group(1)).strip(" .;:-")

            value_start = match.end()
            value_end = (
                matches[i + 1].start()
                if i + 1 < len(matches)
                else len(line)
            )

            value = clean_text(line[value_start:value_end])
            value = value.strip(" .;:-")

            if name and value:
                features.append({
                    "id": len(features) + 1,
                    "@name": name,
                    "#text": value,
                })

    description = clean_text(raw_description)
    return description, features




def normalize_id(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if re.fullmatch(r"\d+", value):
        return int(value)

    return value


def main():
    df = pd.read_parquet(INPUT_FILE)

    required_columns = {"sku_id", "title", "description"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Отсутствуют столбцы: {sorted(missing)}. "
            f"Найдены: {df.columns.tolist()}"
        )

    products = []

    for _, row in df.iterrows():
        description, features = parse_product_description(
            row["description"]
        )

        products.append({
            "@id": normalize_id(row["sku_id"]),
            "name": clean_text(row["title"]),
            "description": description,
            "param": features,
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            products,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Обработано товаров: {len(products)}")
    print(f"JSON сохранён в: {OUTPUT_FILE}")

    if products:
        print("\nПервый товар:")
        print(json.dumps(products[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
