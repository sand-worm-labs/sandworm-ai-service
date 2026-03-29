dev:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
install:
    poetry install
test:
    poetry run pytest
lint:
    poetry run ruff check .
format:
    poetry run ruff format .
fix:
    poetry run ruff check --fix .