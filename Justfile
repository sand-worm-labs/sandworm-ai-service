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
    docker compose up --build

start:
    docker compose up

install package:
    docker compose run --rm ai pip install {{package}}
    docker compose down

test:
    docker compose run --rm ai pytest -s

lint:
    docker compose run --rm ai ruff check .

format:
    docker compose run --rm ai ruff format .

fix:
    docker compose run --rm ai ruff check --fix .