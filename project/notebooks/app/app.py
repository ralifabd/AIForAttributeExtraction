
import json
import sys
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

# Пути к проекту
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Добавляем корень проекта в путь импорта Python
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Импортируем функцию из корневого main.py
from main import enrich_product

app = FastAPI(title="Обогащение товаров")

# Шаблоны и статические файлы
templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)


@app.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.post("/submit", response_class=HTMLResponse)
async def submit_form(
    request: Request,
    title: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
):
    temp_id = int(str(uuid.uuid4().int)[:9])

    try:
        # LLM и нормализация могут работать долго,
        # поэтому выполняем функцию в отдельном потоке.
        result = await run_in_threadpool(
            enrich_product,
            temp_id,
            title,
            description,
        )

        json_str = json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )

        error = None

    except Exception as e:
        json_str = ""
        error = f"{type(e).__name__}: {e}"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "title": title,
            "description": description,
            "json_str": json_str,
            "error": error,
        },
    )
