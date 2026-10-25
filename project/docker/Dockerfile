# Веб-сервис обогащения карточек товаров (FastAPI + uvicorn).
# Собирается из КОРНЯ проекта (контекст ..), см. docker/docker-compose.yml.
# Код - внутри образа, данные (data/: rag.sqlite, схема, каталог) - подключаются томом.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/app/data

WORKDIR /app/notebooks

# Сначала только зависимости: при правке кода этот слой не пересобирается
COPY notebooks/requirements.txt .
RUN pip install -r requirements.txt

COPY notebooks/ .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=5)"

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
