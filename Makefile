.PHONY: test smoke env

test:
	python -m pytest tests

smoke:
	python -m src.smoke

env:
	@echo "environment placeholder - see reports/environment.md"
