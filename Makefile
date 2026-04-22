.PHONY: install lint typecheck test fix ci install-hooks

install:
	uv sync --group dev

lint:
	uv run ruff format --check . && uv run ruff check .

typecheck:
	uv run mypy .

test:
	uv run pytest --cov --cov-report=term-missing

fix:
	uv run ruff format . && uv run ruff check --fix .

ci: lint typecheck test

install-hooks:
	./scripts/install-git-hooks.sh
