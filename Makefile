.PHONY: install lint typecheck test fix ci docker-image install-hooks

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

docker-image:
	uv build --wheel --out-dir sandbox-wheel/dist sandbox-wheel
	docker build -f docker/Dockerfile -t adk-code-mode:local .

install-hooks:
	./scripts/install-git-hooks.sh
