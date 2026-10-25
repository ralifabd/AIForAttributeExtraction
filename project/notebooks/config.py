from pathlib import Path

# Папка notebooks/
NOTEBOOKS_DIR = Path(__file__).resolve().parent

# Корень проекта/
PROJECT_DIR = NOTEBOOKS_DIR.parent

# Папка с данными
DATA_DIR = PROJECT_DIR / "data"

#llm
BASE_URL="https://hub.gigaschool.ru/api/v1/llm/v1"
API_KEY="hub-0cdXYQX_Ie6d-oeL2bn5MfZVeAZLk3xOvc2eWyr6JgY"
MODEL = "GigaChat/GigaChat-3-Pro"

#вот типо этих штук сделай из eda и используй их в eda

PATH_SKUS_RAW = DATA_DIR / "skus.pkl"
PATH_TEXT = DATA_DIR / "sku_desc.parquet"
PATH_SEARCH = DATA_DIR / "query_clicks.parquet"
PATH_CATEGORIES = DATA_DIR / "cat_df.parquet"
PATH_SKUS = DATA_DIR / "skus_meta.parquet"
PATH_ATTRIBUTES = DATA_DIR / "attrs.parquet"
PATH_ATTRIBUTE_SCHEMA = DATA_DIR / "ideal_attribute_schema.json"

OUTPUT_DIR = DATA_DIR / "reports"
SEARCH_DICTIONARIES_DIR = DATA_DIR / "search_dictionaries"

#normalization
INPUT_FILE = DATA_DIR / "sku_desc.parquet"
OUTPUT_FILE = DATA_DIR / "normalized.json"

