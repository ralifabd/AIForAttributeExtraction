from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
print(BASE_DIR)
PROJECT_DIR = BASE_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
PATH_SKUS_RAW = DATA_DIR / "skus.pkl"
PATH_TEXT = DATA_DIR / "sku_desc.parquet"
PATH_SEARCH = DATA_DIR / "query_clicks.parquet"
PATH_CATEGORIES = DATA_DIR / "cat_df.parquet"
PATH_SKUS = DATA_DIR / "skus_meta.parquet"
PATH_ATTRIBUTES = DATA_DIR / "attrs.parquet"
PATH_ATTRIBUTE_SCHEMA = DATA_DIR / "ideal_attribute_schema.json"
SEARCH_DICTIONARIES_DIR = DATA_DIR / "search_dictionaries"

OUTPUT_DIR = DATA_DIR / "reports"

SERVICE_ATTRIBUTES = {
    "подборки",
    "топ коллекция",
    "продавец",
    "потратить бонусы",
    "акции",
    "гарантия",
    "гарантия предоставляется",
    # как понять системный атрибут???
}

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)