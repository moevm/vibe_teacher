# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY forgejo_pr_llm.py ./

RUN useradd --create-home --uid 1000 appuser
USER appuser

# Рабочий каталог для промпта и результатов выгрузки
WORKDIR /data
VOLUME ["/data"]

ENTRYPOINT ["python3", "/app/forgejo_pr_llm.py"]
CMD ["--help"]
