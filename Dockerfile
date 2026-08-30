FROM python:3.12-slim

WORKDIR /app

RUN pip install poetry --quiet

COPY pyproject.toml poetry.lock README.md ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --quiet