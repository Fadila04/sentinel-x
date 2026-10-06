.PHONY: install test test-fast lint format train evaluate replay clean

CONFIG ?= configs/default.yaml

install:      ## Installe le package en mode éditable + outils de dev
	pip install -e ".[dev]"

test:         ## Tous les tests (dont non-régression, ~45 s)
	pytest

test-fast:    ## Tests unitaires uniquement
	pytest -m "not slow"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff check --fix src tests
	ruff format src tests

train:
	sentinel-pm -c $(CONFIG) -v train

evaluate:
	sentinel-pm -c $(CONFIG) evaluate --figures reports/figures --report reports/evaluation.json

replay:
	sentinel-pm -c $(CONFIG) replay

clean:
	rm -rf .pytest_cache .ruff_cache src/*.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} +
