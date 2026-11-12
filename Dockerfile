FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY deep_research ./deep_research

RUN pip install --upgrade pip && \
    pip install -e ".[ddg,web]"

EXPOSE 8000

VOLUME ["/app/reports", "/app/.deep_research"]

CMD ["deep-research-web", "--host", "0.0.0.0", "--port", "8000"]
