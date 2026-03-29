GREEN := "\u{001b}[32m"
YELLOW := "\u{001b}[33m"
RESET := "\u{001b}[0m"

init dev='--dev':
    @if [ ! -f .env ]; then \
        echo "{{GREEN}}.env does not exist. Creating from example...{{RESET}}"; \
        cp .env.example .env; \
    else \
        echo "{{YELLOW}}.env already exists. Skipping creation.{{RESET}}"; \
    fi

dev:
    poetry run uvicorn src.__main__:app --host 0.0.0.0 --port 8000 --reload

start:
    poetry run python -m src.__main__

install:
    poetry install

test:
    poetry run pytest -s

lint:
    poetry run ruff check .

format:
    poetry run ruff format .

fix:
    poetry run ruff check --fix .