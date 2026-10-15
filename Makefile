.PHONY: install install-all test lint format info autoresearch-smoke clean

install:
	uv pip install -e ".[dev]"

install-all:
	uv pip install -e ".[all]"

test:
	pytest

lint:
	ruff check src tests autoresearch scripts

format:
	ruff format src tests autoresearch scripts
	ruff check --fix src tests autoresearch scripts

info:
	bridge-lc info

autoresearch-smoke:
	python autoresearch/prepare.py --synthetic
	python autoresearch/train.py

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info autoresearch/cache autoresearch/run.log
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
