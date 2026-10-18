.PHONY: install test lint data

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

data:
	tsl download --from 2015-01-01 && tsl build && tsl stats
